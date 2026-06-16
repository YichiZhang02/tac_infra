#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
模型推理入口 (policy 驱动) + 录像。

最小命令 (--match-policy 默认开, 硬件与任务自动对齐 checkpoint):
    python -m deployment.inference \
        --robot.type=realman_ugripper_dual \
        --policy.path=playground/results/models/xxx/checkpoints/005000/pretrained_model \
        --dataset.repo_id=eval_pen

--match-policy 会自动:
    - 触觉: 模型用触觉则 use_tactile=true, 否则 false (不连触觉)
    - 相机: 只保留模型实际消费的本地相机 (如模型 wrist_only 则丢掉 cam_top, 不连它)
    - single_task: 从 checkpoint 的 train_config.json -> 训练集 meta/tasks.parquet 自动取
关掉自动对齐: --match-policy=false (此时需手动 --robot.use_tactile / --dataset.single_task 等)

其余默认: 录像存 playground/eval/<repo_id> (video=True); num_episodes 用默认; 不推 hub。
"""

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from deployment._record_engine import RecordConfig, run_record  # noqa: E402 (引擎含 X11/注册初始化)
from vtla.engine.configs import parser

logger = logging.getLogger(__name__)


class StickyHint:
    """把一行提示钉在终端最底行: 后台线程每 0.5s 重绘, 被其他输出刷掉也会马上回来。

    用 ANSI: 保存光标 -> 跳到底行 -> 清行 -> 写提示 -> 恢复光标。非 tty 则空操作。
    """

    def __init__(self, text: str):
        self.text = text
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            # \0337 存光标; \033[999;1H 到底行; \033[K 清行; 写提示; \0338 恢复光标
            sys.stdout.write(f"\0337\033[999;1H\033[K{self.text}\0338")
            sys.stdout.flush()
            self._stop.wait(0.5)

    def __enter__(self):
        if sys.stdout.isatty():
            self._t = threading.Thread(target=self._loop, daemon=True, name="StickyHint")
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
        if sys.stdout.isatty():
            sys.stdout.write("\033[999;1H\033[K")  # 退出时清掉提示行
            sys.stdout.flush()


@dataclass
class InferenceConfig(RecordConfig):
    # 自动把机器人硬件 + 任务描述对齐到 checkpoint (默认开)
    match_policy: bool = True


def _task_from_checkpoint(pretrained_path: str) -> str | None:
    """从 checkpoint 的 train_config.json 找训练集, 读其 meta/tasks.parquet 的任务文字。

    仅当训练集恰好单任务时可靠; 多任务返回 None (需手动指定)。任何一步缺失也返回 None。
    """
    try:
        tc = Path(pretrained_path) / "train_config.json"
        if not tc.is_file():
            logger.warning(f"[match-policy] 未找到 {tc}, 无法自动获取 single_task")
            return None
        info = json.load(open(tc))
        ds = info.get("dataset", {}) or {}
        repo_id = ds.get("repo_id")
        # 候选根目录: train_config 里的 root (可能是训练机绝对路径) + 本地约定 playground/data/<repo_id>
        candidates = []
        if ds.get("root"):
            candidates.append(Path(ds["root"]))
        if repo_id:
            candidates.append(Path("playground/data") / repo_id)
        tasks_pq = next((c / "meta" / "tasks.parquet"
                         for c in candidates if (c / "meta" / "tasks.parquet").is_file()), None)
        if tasks_pq is None:
            logger.warning(f"[match-policy] 未找到训练集任务表 (试过: {[str(c) for c in candidates]})")
            return None
        import pandas as pd
        df = pd.read_parquet(tasks_pq)
        if len(df) != 1:
            logger.warning(f"[match-policy] 训练集有 {len(df)} 个任务, 无法自动选定, 请手动 --dataset.single_task")
            return None
        # 任务文字: 优先 'task' 列, 否则用索引 (LeRobot tasks.parquet 以任务文字为索引)
        task = str(df["task"].iloc[0]) if "task" in df.columns else str(df.index[0])
        return task
    except Exception as e:
        logger.warning(f"[match-policy] 解析训练集任务失败: {e}")
        return None


def _apply_match_policy(cfg: InferenceConfig) -> None:
    """按 checkpoint 把机器人硬件 + single_task 对齐到模型实际所需。"""
    in_feats = set(cfg.policy.input_features or {})
    uses = lambda sub: any(sub in k for k in in_feats)  # noqa: E731

    # 触觉: 模型 input_features 里有触觉键才开
    cfg.robot.use_tactile = uses("finger") or uses("tactile")

    # 本地相机 (cameras dict, 一般是 cam_top): 只留模型消费的
    if hasattr(cfg.robot, "cameras") and cfg.robot.cameras:
        kept = {n: c for n, c in cfg.robot.cameras.items()
                if f"observation.images.{n}" in in_feats}
        dropped = set(cfg.robot.cameras) - set(kept)
        cfg.robot.cameras = kept
        if dropped:
            logger.info(f"[match-policy] 模型不消费, 已不连相机: {sorted(dropped)}")

    # 任务描述: 仅当未手动指定时自动取。
    # 优先用 checkpoint 自带的 single_task (训练写入 config.json, 自包含, 不依赖数据集);
    # 取不到再回退到「从 train_config -> 训练集 tasks.parquet」查找 (兼容旧 checkpoint)。
    if cfg.dataset.single_task is None:
        task = getattr(cfg.policy, "single_task", None)
        if task:
            cfg.dataset.single_task = task
            logger.info(f"[match-policy] single_task <- checkpoint(config.json): {task!r}")
        else:
            task = _task_from_checkpoint(cfg.policy.pretrained_path)
            if task is not None:
                cfg.dataset.single_task = task
                logger.info(f"[match-policy] single_task <- 训练集 tasks.parquet: {task!r}")

    logger.info(f"[match-policy] 对齐结果: use_tactile={cfg.robot.use_tactile}, "
                f"cameras={list(getattr(cfg.robot, 'cameras', {}) or {})}")


@parser.wrap()
def inference(cfg: InferenceConfig):
    # 注意: policy 由 RecordConfig.__post_init__ 从 --policy.path 加载, 解析后即就绪
    if cfg.policy is None:
        raise ValueError("inference 需要模型: 请指定 --policy.path=.../pretrained_model")
    if cfg.teleop is not None:
        raise ValueError("inference 是纯推理入口, 不要 --teleop.*; 采数据请用 `python -m deployment.collect`")

    if cfg.match_policy:
        _apply_match_policy(cfg)

    if cfg.dataset.single_task is None:
        raise ValueError(
            "缺少 single_task: --match-policy 未能自动获取 (训练集缺失/多任务?), "
            "请手动 --dataset.single_task=\"...\""
        )

    # 录像默认存到 playground/eval/<repo_id 末段> (时间戳命名由调用方/bash 负责)
    if cfg.dataset.root is None:
        cfg.dataset.root = f"playground/eval/{cfg.dataset.repo_id.split('/')[-1]}"

    # 钉在底行的保存提示 (推理时容易被日志刷掉)
    hint = " \033[30;43m 推理中 按 → 保存 | ← 重录 | ESC 退出 \033[0m"
    with StickyHint(hint):
        return run_record(cfg)


def main():
    import faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)
    inference()


if __name__ == "__main__":
    main()
