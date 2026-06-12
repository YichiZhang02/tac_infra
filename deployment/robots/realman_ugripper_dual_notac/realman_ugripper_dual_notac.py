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
睿尔曼 RM75b 双臂 (ugripper) — 无触觉版 (notac) LeRobot 适配器。

继承 RealmanUGripperDual, 去掉 2 路触觉传感器 (cam_finger0 / cam_finger1):
    - observation.images 只保留 {side}_cam_wrist (鱼眼) + cam_top
    - 数据流只启动鱼眼, 不启动触觉进程

其余逻辑 (从臂关节/夹爪/连接/断开) 全部继承复用。
_ArmDevices.tactile0 / tactile1 保持为 None, arm.receivers 自动只含鱼眼,
因此 connect / disconnect / is_connected 等基于 receivers 的逻辑无需修改。
"""

import logging
import threading
from functools import cached_property
from typing import Any

from vtla.engine.utils.errors import DeviceNotConnectedError

from ..realman_ugripper_dual.realman_ugripper_dual import (
    RealmanUGripperDual,
    _ArmDevices,
)
from ..realman_ugripper_dual.stream_receivers import (
    StreamReceiver,
    _fisheye_worker,
    decode_fisheye_jpeg,
)
from .config_realman_ugripper_dual_notac import RealmanUGripperDualNotacConfig

logger = logging.getLogger(__name__)


class RealmanUGripperDualNotac(RealmanUGripperDual):
    """睿尔曼 RM75b 双臂 (ugripper) 无触觉版。"""

    config_class = RealmanUGripperDualNotacConfig
    name = "realman_ugripper_dual_notac"

    # ==================== 特征定义 ====================

    @property
    def _stream_ft(self) -> dict[str, tuple]:
        # 只保留手腕鱼眼, 去掉触觉
        ft: dict[str, tuple] = {}
        for side in self.config.arms:
            ft[f"{side}_cam_wrist"] = (self.config.fisheye_height, self.config.fisheye_width, 3)
        return ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._stream_ft, **self._cameras_ft}

    # ==================== 连接数据流 (仅鱼眼) ====================

    def _connect_streams(self, side: str, arm: _ArmDevices) -> None:
        cfg = self.config
        board_ip = self._board_ip(side)
        udp_port = self._fisheye_udp_port(side)

        arm.fisheye = StreamReceiver(
            name=f"{side}_cam_wrist",
            target=_fisheye_worker,
            args=(board_ip, cfg.fisheye_grpc_port, udp_port,
                  cfg.fisheye_width, cfg.fisheye_height, cfg.fisheye_max_datagram,
                  cfg.stream_max_fps, cfg.stream_debug_fps),
            empty_frame=self._empty_wrist,
            first_frame_timeout=cfg.stream_first_frame_timeout,
            decode_fn=decode_fisheye_jpeg,   # 队列传 JPEG 字节, 消费端解码
        )
        # tactile0 / tactile1 保持 None (无触觉)
        stream_errors: dict[str, Exception] = {}

        def _connect_stream(r: StreamReceiver):
            try:
                logger.info(f"[{side}] 正在启动数据流 {r.name}...")
                r.connect()
            except Exception as e:  # noqa: BLE001
                stream_errors[r.name] = e

        ts = [threading.Thread(target=_connect_stream, args=(r,), name=f"stream-{r.name}")
              for r in arm.receivers]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if stream_errors:
            name, err = next(iter(stream_errors.items()))
            raise ConnectionError(f"数据流 {name} 启动失败: {err}") from err

    # ==================== 读取观测 (无触觉) ====================

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        obs: dict[str, Any] = {}

        # 1. 关节 + 夹爪 (state)
        for side in self.config.arms:
            arm = self._arms[side]
            joints = arm.state_reader.get_state() if arm.state_reader is not None else [0.0] * self.DOF
            for i, joint in enumerate(self.JOINT_NAMES):
                obs[f"{side}_{joint}"] = joints[i]
            if arm.state_reader is not None and arm.state_reader.get_state_age() > 0.1:
                logger.warning(f"[{side}] 从臂状态过时: {arm.state_reader.get_state_age()*1000:.0f}ms")

            obs[f"{side}_{self.GRIPPER_NAME}"] = (
                arm.gripper.read_norm() if arm.gripper is not None else 0.0
            )

        # 2. 数据流图像 (仅鱼眼)
        for side in self.config.arms:
            arm = self._arms[side]
            obs[f"{side}_cam_wrist"] = arm.fisheye.async_read()

        # 3. 额外本地相机
        for cam_key, cam in self.cameras.items():
            img = cam.async_read()
            if cam_key in self.config.crop_4_3_cameras:
                img = self._center_crop_4_3(img)
            obs[cam_key] = img

        return obs
