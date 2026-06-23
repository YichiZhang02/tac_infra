# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
真机策略评估 (eval-inference) 脚本 —— 基于 lerobot-record, 但**完全独立**, 不修改
任何现有录制/训练逻辑。

设计目标:
    - 让策略在**同一个起始姿态**反复开始, 便于多次公平评估;
    - 评估过程中随时按 [空格] 或 [r], 立即打断策略输出, 机械臂平滑回到"启动时
      捕获的初始姿态", 静置若干秒后, 策略自动重新接管开始下一次;
    - 按 [ESC] 或 [q] 退出。

与 lerobot-record 的关系:
    - 复用 RecordConfig / DatasetRecordConfig 以及 stream 视频保存逻辑 (StreamVideoWriter
      等), 因此现有的命令行参数 (--robot.type / --policy.path / --dataset.save=stream ...)
      原样可用;
    - 但本脚本不调用 lerobot_record.record(), 而是用自己的评估主循环, 因此对现有
      lerobot-record 行为零影响。

home (初始姿态) 定义:
    脚本连上机器人后, 读一次观测, 取出 action_features 对应的关节+夹爪值作为 home。
    放哪开始就回哪, 无需手工配置。

复位运动:
    从当前姿态在 ~home_move_s 秒内分多帧线性插值, 经 robot.send_action 平滑移动到 home,
    避免一步到位的猛冲; 到位后静置 reset_settle_s 秒, 再让策略重新接管。
"""

import sys
import os


def _init_x11_threads():
    """与 lerobot_record 一致: 在导入任何 X11 相关库之前初始化 X11 多线程。"""
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        try:
            import ctypes

            x11 = ctypes.CDLL("libX11.so.6")
            if x11.XInitThreads() == 0:
                import logging

                logging.warning("XInitThreads() 返回 0，X11 多线程初始化可能失败")
        except OSError:
            pass
        except Exception as e:
            import logging

            logging.debug(f"X11 线程初始化跳过: {e}")


_init_x11_threads()

import logging
import threading
import time
from dataclasses import dataclass
from pprint import pformat
from typing import Any

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import is_headless, predict_action, sanity_check_dataset_name
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# 复用 lerobot_record 中的配置与 stream 保存组件, 不重复实现, 也不改动原文件
from lerobot.scripts.lerobot_record import (
    RecordConfig,
    StreamPolicyMeta,
    StreamVideoWriter,
    resolve_compute_stats,
    resolve_dataset_root,
    resolve_stream_camera_keys,
)


@dataclass
class EvalInferenceConfig(RecordConfig):
    """评估推理配置: 继承 RecordConfig, 仅追加两个复位相关参数。"""

    # 复位时从当前姿态平滑插值回 home 的运动时长 (秒)
    home_move_s: float = 2.0
    # 回到 home 后静置等待、再让策略接管的时长 (秒)
    reset_settle_s: float = 3.0

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class EvalKeyboardListener:
    """评估专用键盘监听: [空格]/[r] 请求复位, [ESC]/[q] 退出。

    同时支持有显示环境 (pynput) 与 SSH headless 环境 (终端 stdin raw 模式),
    与现有 init_keyboard_listener 互不影响 (本类完全独立)。
    """

    def __init__(self):
        self.events = {"reset_request": False, "stop": False}
        self._pynput_listener = None
        self._term_thread = None
        self._term_running = False
        self._old_settings = None

    def start(self):
        if is_headless():
            self._start_terminal()
        else:
            self._start_pynput()

    # ---------- pynput (有显示) ----------
    def _start_pynput(self):
        try:
            from pynput import keyboard
        except Exception as e:
            logging.warning(f"pynput 不可用, 回退到终端监听: {e}")
            self._start_terminal()
            return

        def on_press(key):
            try:
                if key == keyboard.Key.space or (
                    hasattr(key, "char") and key.char in ("r", "R")
                ):
                    print("\n[评估] 收到复位请求 (空格/r): 打断策略, 回初始位置...")
                    self.events["reset_request"] = True
                elif key == keyboard.Key.esc or (
                    hasattr(key, "char") and key.char in ("q", "Q")
                ):
                    print("\n[评估] 收到退出请求 (ESC/q)...")
                    self.events["stop"] = True
            except Exception as e:
                print(f"按键处理出错: {e}")

        self._pynput_listener = keyboard.Listener(on_press=on_press)
        self._pynput_listener.start()
        logging.info("键盘监听已启动 (pynput): [空格]/[r]=复位, [ESC]/[q]=退出")

    # ---------- 终端 stdin (SSH headless) ----------
    def _start_terminal(self):
        self._term_running = True
        self._term_thread = threading.Thread(target=self._terminal_loop, daemon=True)
        self._term_thread.start()
        logging.info("键盘监听已启动 (终端): [空格]/[r]=复位, [ESC]/[q]=退出")

    def _terminal_loop(self):
        import select
        import termios
        import tty

        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            fd = sys.stdin.fileno()
            while self._term_running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    data = os.read(fd, 1024).decode("utf-8", errors="replace")
                    if " " in data or "r" in data or "R" in data:
                        print("\n[评估] 收到复位请求 (空格/r): 打断策略, 回初始位置...")
                        self.events["reset_request"] = True
                    elif "\x1b" in data or "q" in data or "Q" in data:
                        print("\n[评估] 收到退出请求 (ESC/q)...")
                        self.events["stop"] = True
        except Exception as e:
            logging.debug(f"终端键盘监听出错: {e}")
        finally:
            if self._old_settings is not None:
                import termios

                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
                except Exception:
                    pass

    def stop(self):
        self._term_running = False
        if self._term_thread is not None:
            self._term_thread.join(timeout=0.5)
        if self._pynput_listener is not None:
            self._pynput_listener.stop()


def capture_home_action(robot, action_features: dict[str, Any]) -> dict[str, float]:
    """读一次观测, 取出 action_features 对应键作为 home 姿态 (关节 + 夹爪)。"""
    obs = robot.get_observation()
    home = {}
    for key in action_features:
        if key not in obs:
            raise KeyError(
                f"捕获 home 失败: 观测中缺少动作键 '{key}'。"
                f"可用观测键示例: {list(obs.keys())[:8]} ..."
            )
        home[key] = float(obs[key])
    return home


def move_to_home(
    robot,
    home_action: dict[str, float],
    action_features: dict[str, Any],
    fps: int,
    duration_s: float,
):
    """从当前姿态在 duration_s 内分多帧线性插值平滑移动到 home。"""
    obs = robot.get_observation()
    start = {k: float(obs[k]) for k in action_features if k in obs}

    steps = max(1, int(duration_s * fps))
    for i in range(1, steps + 1):
        t0 = time.perf_counter()
        alpha = i / steps
        target = {
            k: start.get(k, home_action[k]) * (1 - alpha) + home_action[k] * alpha
            for k in home_action
        }
        robot.send_action(target)
        busy_wait(1 / fps - (time.perf_counter() - t0))


def eval_segment(
    robot,
    listener: EvalKeyboardListener,
    fps: int,
    robot_observation_processor,
    robot_action_processor,
    record_features: dict[str, dict[str, Any]],
    policy,
    preprocessor,
    postprocessor,
    single_task: str,
    stream_writer: StreamVideoWriter | None,
    control_time_s: float,
    display_data: bool,
) -> str:
    """运行一段策略推理, 直到: 用户请求复位 / 用户退出 / 到达时间上限。

    返回结束原因: "reset" | "stop" | "timeout"。
    """
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_t = time.perf_counter()
    while True:
        loop_t = time.perf_counter()

        if listener.events["stop"]:
            return "stop"
        if listener.events["reset_request"]:
            listener.events["reset_request"] = False
            return "reset"
        if (time.perf_counter() - start_t) >= control_time_s:
            return "timeout"

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(record_features, obs_processed, prefix=OBS_STR)

        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )
        act_processed_policy = make_robot_action(action_values, record_features)
        action_to_send = robot_action_processor((act_processed_policy, obs))
        robot.send_action(action_to_send)

        if stream_writer is not None:
            stream_writer.add_observation(obs_processed)

        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed_policy)

        busy_wait(1 / fps - (time.perf_counter() - loop_t))


@parser.wrap()
def eval_inference(cfg: EvalInferenceConfig):
    init_logging()
    logging.info(pformat({k: v for k, v in vars(cfg).items()}))

    if cfg.policy is None:
        raise ValueError("评估脚本必须提供 --policy.path。")

    save_mode = cfg.dataset.save
    if save_mode not in ("stream", "episode"):
        raise ValueError(f"不支持的 save 模式: {save_mode}")
    # 评估只支持纯推理(不存)或 stream 存视频; episode 完整数据集保存交给 lerobot-record
    if save_mode == "episode":
        logging.warning(
            "eval-inference 仅支持 save=stream(存视频) 或纯推理; 已将 episode 视为纯推理(不保存)。"
        )
        save_mode = "none"

    resolve_compute_stats(cfg.dataset.save, cfg.dataset.compute_stats)

    if cfg.display_data:
        init_rerun(session_name="eval_inference")

    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    record_features = dataset_features

    # 策略加载 (与 lerobot-record stream 路径一致: 轻量 ds_meta 占位)
    sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
    ds_meta_for_policy = StreamPolicyMeta(features=record_features, stats=None)
    dataset_stats = rename_stats({}, cfg.dataset.rename_map)
    policy = make_policy(cfg.policy, ds_meta=ds_meta_for_policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
        },
    )

    robot.connect()
    listener = EvalKeyboardListener()
    listener.start()

    # 捕获 home 姿态 (启动时双臂关节 + 夹爪)
    home_action = capture_home_action(robot, robot.action_features)
    logging.info(f"已捕获初始 home 姿态 ({len(home_action)} 个自由度), 复位将回到此处。")

    fps = cfg.dataset.fps

    stream_writer = None
    stream_ctx = None
    if save_mode == "stream":
        stream_root = resolve_dataset_root(cfg.dataset) / "stream"
        stream_camera_keys = resolve_stream_camera_keys(robot, cfg.dataset.stream_camera_keys)
        if stream_camera_keys:
            logging.info(f"Stream 模式将保存相机: {list(stream_camera_keys)}")
        else:
            logging.warning("Stream 模式未解析到 RGB 相机键, 不会保存视频。")
        stream_ctx = StreamVideoWriter(
            stream_root=stream_root, fps=fps, camera_keys=stream_camera_keys
        )
        stream_writer = stream_ctx.__enter__()
        # resume: 接着已有 episode 编号
        episode_index = 0
        if cfg.resume and stream_root.exists():
            existing = sorted(stream_root.glob("episode-*"))
            if existing:
                try:
                    episode_index = int(existing[-1].name.split("-", 1)[1]) + 1
                except (ValueError, IndexError):
                    episode_index = len(existing)
    else:
        episode_index = 0

    try:
        while not listener.events["stop"]:
            log_say(
                f"策略接管中 (eval #{episode_index})。按 [空格]/[r] 复位, [ESC]/[q] 退出。",
                cfg.play_sounds,
            )
            if stream_writer is not None:
                stream_writer.start_episode(episode_index)

            reason = eval_segment(
                robot=robot,
                listener=listener,
                fps=fps,
                robot_observation_processor=robot_observation_processor,
                robot_action_processor=robot_action_processor,
                record_features=record_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                single_task=cfg.dataset.single_task,
                stream_writer=stream_writer,
                control_time_s=cfg.dataset.episode_time_s,
                display_data=cfg.display_data,
            )

            if stream_writer is not None:
                # 时间到 / 复位 / 退出, 当前段都算一条评估视频保存
                stream_writer.close_episode(discard=False)
            episode_index += 1

            if reason == "stop":
                break

            # reset / timeout: 打断策略输出, 平滑回到 home, 静置后重新接管
            log_say("打断策略, 平滑回到初始位置...", cfg.play_sounds)
            move_to_home(robot, home_action, robot.action_features, fps, cfg.home_move_s)

            log_say(f"已回到初始位置, 静置 {cfg.reset_settle_s:.0f} 秒...", cfg.play_sounds)
            settle_t = time.perf_counter()
            while time.perf_counter() - settle_t < cfg.reset_settle_s:
                if listener.events["stop"]:
                    break
                # 静置期间持续保持 home, 防止漂移
                robot.send_action(home_action)
                busy_wait(1 / fps)
    finally:
        if hasattr(policy, "stop"):
            policy.stop()
        if stream_ctx is not None:
            stream_ctx.__exit__(None, None, None)
        log_say("评估结束", cfg.play_sounds, blocking=True)
        robot.disconnect()
        listener.stop()


def main():
    import faulthandler

    faulthandler.enable(file=sys.stderr, all_threads=True)
    register_third_party_devices()
    eval_inference()


if __name__ == "__main__":
    main()
