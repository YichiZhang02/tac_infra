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
采数据入口 (遥操作 teleop 驱动)。

最小命令:
    python -m deployment.collect \
        --robot.type=realman_ugripper_dual \
        --teleop.type=bi_realman_ugripper_leader \
        --dataset.repo_id=pick_pen \
        --dataset.single_task="抓笔" \
        --dataset.num_episodes=20

默认行为:
    - 数据存到 playground/data/<repo_id> (不传 --dataset.root 时)
    - 触觉随 robot 配置, 默认开; 不要触觉加 --robot.use_tactile=false
    - 不推 HuggingFace hub (需要才加 --dataset.push_to_hub=true)
    - 其余硬件参数 (IP/串口等) 走各自 config 默认, 需要时照样可 --robot.xxx / --teleop.xxx 覆盖
"""

import sys

from deployment._record_engine import RecordConfig, run_record  # noqa: E402 (引擎含 X11/注册初始化)
from vtla.engine.configs import parser


@parser.wrap()
def collect(cfg: RecordConfig):
    if cfg.policy is not None:
        raise ValueError("collect 是采数据入口, 不接受 --policy.*; 模型推理请用 `python -m deployment.inference`")
    if cfg.teleop is None:
        raise ValueError("collect 需要遥操作器: 请指定 --teleop.type=... (如 bi_realman_ugripper_leader)")
    if cfg.dataset.single_task is None:
        raise ValueError("collect 需要任务描述: 请指定 --dataset.single_task=\"...\"")

    # 默认存到 playground/data/<repo_id 末段> (时间戳命名由调用方/bash 负责)
    if cfg.dataset.root is None:
        cfg.dataset.root = f"playground/data/{cfg.dataset.repo_id.split('/')[-1]}"
    return run_record(cfg)


def main():
    import faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)
    collect()


if __name__ == "__main__":
    main()
