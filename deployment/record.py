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
采集数据集 / 用策略控制机械臂 (tac_infra deployment, 自包含)

动作可来自遥操作 (主臂) 或策略 (模型推理)。触觉传感器以 uint8 (TactileSensorFeat) 保存。

示例 - 遥操作采数据 (睿尔曼 RM75b + 触觉):
```shell
python -m deployment.record \
    --robot.type=realman_tactile_shandd_hd \
    --robot.id=realman_right \
    --teleop.type=realman_rm75b_leader \
    --teleop.port=/dev/ttyLeaderR \
    --dataset.repo_id=local/my_dataset \
    --dataset.num_episodes=20 \
    --dataset.single_task="Grab the cube" \
    --dataset.push_to_hub=false
```

示例 - 用策略控制 (模型推理):
```shell
python -m deployment.record \
    --robot.type=realman_tactile_shandd_hd \
    --robot.id=realman_right \
    --policy.path=playground/results/models/xxx/checkpoints/last/pretrained_model \
    --dataset.repo_id=local/eval_dataset \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab the cube" \
    --dataset.push_to_hub=false
```
"""

# ============================================================================
# X11 线程安全初始化 - 必须在任何其他导入之前执行
# 解决 pynput (Xlib) + 多线程导致的 futex 崩溃问题
# ============================================================================
import sys
import os


def _init_x11_threads():
    """初始化 X11 多线程支持，防止 Xlib 在多线程环境下崩溃。"""
    if sys.platform.startswith('linux') and os.environ.get('DISPLAY'):
        try:
            import ctypes
            x11 = ctypes.CDLL('libX11.so.6')
            result = x11.XInitThreads()
            if result == 0:
                import logging
                logging.warning("XInitThreads() 返回 0，X11 多线程初始化可能失败")
        except OSError:
            pass
        except Exception as e:
            import logging
            logging.debug(f"X11 线程初始化跳过: {e}")


_init_x11_threads()
# ============================================================================

import logging
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

# ---- 硬件层 (deployment 自包含) ----
from deployment.cameras import CameraConfig  # noqa: F401
from deployment.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from deployment.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from deployment.robots import Robot, RobotConfig, make_robot_from_config
from deployment.robots.realman_tactile_shandd_hd import RealmanTactileShanddHd  # noqa: F401  注册 config 选项
from deployment.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from deployment.teleoperators.realman_rm75b_leader import RealmanRM75bLeader  # noqa: F401  注册 config 选项

# ---- 策略 / 数据集 / 处理管线 (复用本仓库 vtla) ----
from vtla.engine.configs import parser
from vtla.engine.configs.policies import PreTrainedConfig
from vtla.datasets.image_writer import safe_stop_image_writer
from vtla.datasets.lerobot_dataset import LeRobotDataset
from vtla.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from vtla.datasets.video_utils import VideoEncodingManager
from vtla.engine.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from vtla.frameworks.factory import make_policy, make_pre_post_processors
from vtla.frameworks.pretrained import PreTrainedPolicy
from vtla.frameworks.utils import make_robot_action
from vtla.engine.types import PolicyAction, RobotAction, RobotObservation
from vtla.engine.processor.pipeline import PolicyProcessorPipeline, RobotProcessorPipeline
from vtla.engine.processor.factory import make_default_processors
from vtla.engine.processor.rename_processor import rename_stats
from vtla.engine.utils.constants import ACTION, OBS_STR, HF_LEROBOT_HOME
from vtla.engine.common.control_utils import (
    init_keyboard_listener,
    is_headless,  # noqa: F401
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from vtla.engine.utils.device_utils import get_safe_torch_device
from vtla.engine.utils.utils import init_logging, log_say
from vtla.engine.utils.visualization_utils import init_rerun, log_rerun_data


def busy_wait(seconds: float) -> None:
    """精确控频等待。Mac/Windows 上 time.sleep 不够精准，用忙等。"""
    if platform.system() in ("Darwin", "Windows"):
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        if seconds > 0:
            time.sleep(seconds)


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}'.
    repo_id: str
    # A short but accurate description of the task performed during the recording.
    single_task: str
    # Root directory where the dataset will be stored.
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    video_encoding_batch_size: int = 1
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    # 保存模式：episode (标准 LeRobot 数据集) / stream (仅保存 stream/ 下视频 sidecar)
    save: str = "episode"
    # 统一统计开关 (stream 模式默认 False, episode 默认 True)
    compute_stats: bool | None = None

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")
        if self.save not in {"episode", "stream"}:
            raise ValueError(f"`dataset.save` must be one of ['episode', 'stream'], got: {self.save}")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@dataclass
class StreamPolicyMeta:
    """用于 stream 模式下 policy 初始化的轻量 ds_meta 占位对象。"""
    features: dict[str, dict[str, Any]]
    stats: dict[str, Any] | None = None


def resolve_compute_stats(save_mode: str, compute_stats: bool | None) -> bool:
    """统一解析统计开关：stream 默认 False，episode 默认 True。"""
    if compute_stats is not None:
        return compute_stats
    return save_mode == "episode"


def resolve_dataset_root(dataset_cfg: DatasetRecordConfig) -> Path:
    """解析数据集根目录，保证 stream 与 episode 的 root 规则一致。"""
    return Path(dataset_cfg.root) if dataset_cfg.root is not None else HF_LEROBOT_HOME / dataset_cfg.repo_id


class StreamVideoWriter:
    """stream 模式视频写入器：仅写固定相机，不生成任何 meta 文件。"""

    def __init__(
        self,
        stream_root: Path,
        fps: int,
        camera_keys: tuple[str, ...] = ("cam_top", "cam_right_wrist"),
    ):
        self.stream_root = Path(stream_root)
        self.fps = fps
        self.camera_keys = camera_keys
        self.stream_root.mkdir(parents=True, exist_ok=True)

        self._episode_dir: Path | None = None
        self._writers: dict[str, Any] = {}
        self._frame_sizes: dict[str, tuple[int, int]] = {}
        self._resize_warned_keys: set[str] = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def start_episode(self, episode_index: int) -> None:
        self.close_episode(discard=False)
        self._episode_dir = self.stream_root / f"episode-{episode_index:06d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)

    def _make_writer(self, camera_key: str, frame: np.ndarray):
        import cv2

        if self._episode_dir is None:
            raise RuntimeError("Stream episode is not started")

        height, width = int(frame.shape[0]), int(frame.shape[1])
        self._frame_sizes[camera_key] = (width, height)
        video_path = self._episode_dir / f"{camera_key}.mp4"

        for fourcc_tag in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*fourcc_tag),
                float(self.fps),
                (width, height),
            )
            if writer.isOpened():
                return writer
            writer.release()

        raise RuntimeError(f"Failed to create stream video writer for {camera_key}: {video_path}")

    def _normalize_frame(self, frame: Any, camera_key: str) -> np.ndarray | None:
        import cv2

        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = frame.transpose(1, 2, 0)

        if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
            logging.warning(
                f"Skip stream frame for {camera_key}: unsupported shape {getattr(frame, 'shape', None)}"
            )
            return None

        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame, 0.0, 1.0)
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        expected_size = self._frame_sizes.get(camera_key)
        if expected_size is not None:
            width, height = expected_size
            if frame.shape[1] != width or frame.shape[0] != height:
                if camera_key not in self._resize_warned_keys:
                    logging.warning(
                        f"Resize stream frame for {camera_key} from {frame.shape[1]}x{frame.shape[0]} "
                        f"to {width}x{height}"
                    )
                    self._resize_warned_keys.add(camera_key)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        return frame

    def add_observation(self, obs: dict[str, Any]) -> None:
        import cv2

        if self._episode_dir is None:
            raise RuntimeError("Stream episode is not started")

        for camera_key in self.camera_keys:
            frame = obs.get(camera_key)
            if frame is None:
                logging.warning(f"Missing frame for stream camera '{camera_key}', skipping this frame.")
                continue

            normalized_frame = self._normalize_frame(frame, camera_key)
            if normalized_frame is None:
                continue

            writer = self._writers.get(camera_key)
            if writer is None:
                writer = self._make_writer(camera_key, normalized_frame)
                self._writers[camera_key] = writer

            bgr_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

    def close_episode(self, discard: bool = False) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        self._frame_sizes.clear()
        self._resize_warned_keys.clear()

        if discard and self._episode_dir is not None and self._episode_dir.exists():
            shutil.rmtree(self._episode_dir, ignore_errors=True)

        self._episode_dir = None

    def close(self) -> None:
        self.close_episode(discard=False)


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    record_features: dict[str, dict[str, Any]] | None = None,
    stream_writer: StreamVideoWriter | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        # stream + policy 场景下 dataset 可能为 None，因此统一使用 features_for_frame
        features_for_frame = dataset.features if dataset is not None else record_features

        observation_frame = None
        if policy is not None or dataset is not None:
            if features_for_frame is None:
                raise ValueError(
                    "record_features is required when policy is enabled or dataset is None in record_loop."
                )
            observation_frame = build_dataset_frame(features_for_frame, obs_processed, prefix=OBS_STR)

        # Get action from either policy or teleop
        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None
        if policy is not None and preprocessor is not None and postprocessor is not None:
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

            if features_for_frame is None:
                raise ValueError("record_features is required for policy action conversion.")
            act_processed_policy = make_robot_action(action_values, features_for_frame)

        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor
        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot. Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            if features_for_frame is None:
                raise ValueError("record_features is required for dataset frame construction.")
            action_frame = build_dataset_frame(features_for_frame, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
        elif stream_writer is not None:
            # stream 模式仅保存视频 sidecar，不写标准 dataset 帧
            stream_writer.add_observation(obs_processed)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset | None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    # 解析保存模式与统计开关 (stream 默认不计算统计)
    save_mode = cfg.dataset.save
    compute_stats_enabled = resolve_compute_stats(save_mode, cfg.dataset.compute_stats)

    if save_mode == "stream" and compute_stats_enabled:
        logging.info("`dataset.save=stream` currently does not compute dataset stats, forcing stats off.")

    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

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

    dataset: LeRobotDataset | None = None
    if save_mode == "episode":
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # episode 模式创建标准 LeRobotDataset
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )
    else:
        # stream 模式不创建 LeRobotDataset，只做 repo_id 合法性检查
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)

    # 按当前模式加载策略；stream 模式使用轻量 ds_meta 占位
    if cfg.policy is not None:
        if dataset is not None:
            ds_meta_for_policy = dataset.meta
            dataset_stats = rename_stats(dataset.meta.stats, cfg.dataset.rename_map)
        else:
            ds_meta_for_policy = StreamPolicyMeta(features=record_features, stats=None)
            dataset_stats = rename_stats({}, cfg.dataset.rename_map)
        policy = make_policy(cfg.policy, ds_meta=ds_meta_for_policy)
    else:
        policy = None

    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
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
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    try:
        if save_mode == "episode":
            if dataset is None:
                raise RuntimeError("Episode mode requires a valid dataset instance.")

            with VideoEncodingManager(dataset):
                recorded_episodes = 0
                while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                    log_say(f"正在记录中,..., 按<-方向键复位重新开始, 按->方向键保存 , episode {dataset.num_episodes}", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=dataset,
                        record_features=record_features,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                    # 跳过最后一集的 reset
                    if not events["stop_recording"] and (
                        (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                    ):
                        log_say("复位中, 进行桌面场景还原.,Reset the environment", cfg.play_sounds)
                        record_loop(
                            robot=robot,
                            events=events,
                            fps=cfg.dataset.fps,
                            teleop_action_processor=teleop_action_processor,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            teleop=teleop,
                            record_features=record_features,
                            control_time_s=cfg.dataset.reset_time_s,
                            single_task=cfg.dataset.single_task,
                            display_data=cfg.display_data,
                        )

                    if events["rerecord_episode"]:
                        log_say("复位中..., 可以开始操作,Re-record episode", cfg.play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        dataset.clear_episode_buffer()
                        continue

                    dataset.save_episode()
                    recorded_episodes += 1
        else:
            # stream 模式使用 sidecar 视频写入，不触碰 LeRobotDataset 计数逻辑
            stream_root = resolve_dataset_root(cfg.dataset) / "stream"
            with StreamVideoWriter(stream_root=stream_root, fps=cfg.dataset.fps) as stream_writer:
                episode_start = 0
                if cfg.resume and stream_root.exists():
                    existing = sorted(stream_root.glob("episode-*"))
                    if existing:
                        last_name = existing[-1].name
                        try:
                            episode_start = int(last_name.split("-", 1)[1]) + 1
                        except (ValueError, IndexError):
                            episode_start = len(existing)
                        logging.info(f"Stream resume: found {len(existing)} existing episodes, starting from episode {episode_start}")

                recorded_episodes = 0
                episode_index = episode_start
                while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                    log_say(f"Recording stream episode {episode_index}", cfg.play_sounds)
                    stream_writer.start_episode(episode_index)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=None,
                        record_features=record_features,
                        stream_writer=stream_writer,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                    if not events["stop_recording"] and (
                        (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment", cfg.play_sounds)
                        record_loop(
                            robot=robot,
                            events=events,
                            fps=cfg.dataset.fps,
                            teleop_action_processor=teleop_action_processor,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            teleop=teleop,
                            record_features=record_features,
                            control_time_s=cfg.dataset.reset_time_s,
                            single_task=cfg.dataset.single_task,
                            display_data=cfg.display_data,
                        )

                    if events["rerecord_episode"]:
                        log_say("Re-record episode", cfg.play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        stream_writer.close_episode(discard=True)
                        continue

                    stream_writer.close_episode(discard=False)
                    recorded_episodes += 1
                    episode_index += 1
    finally:
        # 优先停止策略推理线程
        if policy is not None and hasattr(policy, "stop"):
            policy.stop()

        log_say("Stop recording", cfg.play_sounds, blocking=True)

        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

    if cfg.dataset.push_to_hub:
        if save_mode == "episode" and dataset is not None:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        else:
            logging.info("Skipping push_to_hub in stream mode (stream outputs are non-standard sidecar videos).")

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    import faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)
    record()


if __name__ == "__main__":
    main()
