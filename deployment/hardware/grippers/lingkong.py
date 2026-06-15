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
领控电爪 (ugripper) 封装器 —— 新 SDK 版本

封装 ugripper/zxd_test_ugrp/lingkong_grip_api.py 的 LingkongGrip。
参考实测脚本: ugripper/zxd_test_ugrp/test_grip_new_api.py。

新 SDK 语义 (与旧版方向相反, 已统一):
    move_to_pos(0)    = 夹紧 (close)
    move_to_pos(1000) = 张开 (open)
    read_pos()        与之一致 (0=夹紧, 1000=张开)
    grip_init() 会实测行程并自适配左/右镜像。

⚠️ 关键: grip_init 之前必须先发 0x9B(清错误) + 0x88(使能), 否则电机若处于
   失能/错误态会卡在 "Read open position" 失败。

本类对外采用 **归一化 [0, 1]** 语义 (1.0=张开, 0.0=夹紧), 与机器人 observation /
action 中的 *_main_gripper 字段保持同一坐标系, 便于训练。
"""

import logging
import time

from .._sdk_paths import ensure_lingkong_sdk
from .base import GripperBase

logger = logging.getLogger(__name__)


class LingkongGripper(GripperBase):
    """领控电爪封装 (新 SDK)。

    Attributes:
        server_address: 夹爪 gRPC 服务地址, 如 "192.168.1.10:55551"
        can_interface:  CAN 接口名, 默认 "can0"
        can_bitrate:    CAN 波特率, 默认 1_000_000
        speed:          运动速度 10~100
        torque:         力矩限制 10~100
    """

    CAN_ID = 0x141

    def __init__(
        self,
        server_address: str,
        can_interface: str = "can0",
        can_bitrate: int = 1_000_000,
        speed: int = 40,
        torque: int = 50,
    ):
        self.server_address = server_address
        self.can_interface = can_interface
        self.can_bitrate = can_bitrate
        self.speed = speed
        self.torque = torque

        self._grip = None
        self._connected = False
        self._initialized = False
        self._last_good_pos = 0  # read_pos 失败时回退

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def connect(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """连接夹爪 gRPC/CAN 服务 (带重试)。"""
        if self._connected:
            logger.warning(f"夹爪 {self.server_address} 已连接")
            return True

        ensure_lingkong_sdk()
        try:
            from dm_lingkong_grip_sdk import LingkongGrip
        except ImportError as e:
            logger.error(f"无法导入 LingkongGrip (dm_lingkong_grip_sdk): {e}")
            return False

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"正在连接夹爪 {self.server_address} (尝试 {attempt}/{max_retries})..."
                )
                grip = LingkongGrip(
                    server_address=self.server_address,
                    interface=self.can_interface,
                    bitrate=self.can_bitrate,
                )
                if grip.init_status:
                    self._grip = grip
                    self._connected = True
                    logger.info(f"✅ 夹爪 {self.server_address} CAN 通讯初始化成功")
                    return True
                logger.warning(f"夹爪 CAN 通讯初始化失败 (尝试 {attempt}/{max_retries})")
            except Exception as e:
                logger.warning(f"连接夹爪出错 (尝试 {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                time.sleep(retry_delay)

        logger.error(f"夹爪 {self.server_address} 连接失败, 已重试 {max_retries} 次")
        return False

    def _clear_and_enable(self) -> None:
        """清错误 + 使能闭环 —— grip_init 前的恢复步骤 (见 test_grip_new_api.py)。"""
        self._grip.client.recv_can_async(self._grip._on_message_received, 1000)
        time.sleep(0.3)
        self._grip.client.send_can(self.CAN_ID, [0x9B, 0, 0, 0, 0, 0, 0, 0])  # 清错误标志
        time.sleep(0.2)
        self._grip.client.send_can(self.CAN_ID, [0x88, 0, 0, 0, 0, 0, 0, 0])  # 电机使能
        time.sleep(0.2)

    def init_gripper(self, timeout: int = 6000, itinerary_override: int | None = None) -> bool:
        """初始化夹爪 (会先夹紧实测行程)。

        Args:
            timeout: grip_init 超时 (ms)
            itinerary_override: 真实满行程编码器计数。SDK grip_init 会把 max_itinerary
                写死成 25000/90000, 但左右爪传动比不同时该值不准 (见 measure_gripper_stroke.py)。
                传入实测值则按 open_pos = clamp_pos - itinerary 重算位置映射。
        """
        if not self._connected or self._grip is None:
            logger.error("夹爪未连接, 无法初始化")
            return False
        try:
            self._clear_and_enable()
            if not self._grip.grip_init(time_out=timeout):
                logger.error("夹爪 grip_init 失败")
                return False
            self._grip.set_torque_limit(self.torque)
            self._grip.set_speed(self.speed)

            if itinerary_override is not None:
                self._grip._max_itinerary = int(itinerary_override)
                self._grip._open_pos = self._grip._clamp_pos - int(itinerary_override)
                logger.info(f"夹爪行程已覆盖为实测值 max_itinerary={itinerary_override}")

            self._initialized = True
            logger.info(
                f"✅ 夹爪初始化成功 clamp={self._grip._clamp_pos} "
                f"open={self._grip._open_pos} itinerary={self._grip._max_itinerary}"
            )
            return True
        except Exception as e:
            logger.error(f"初始化夹爪出错: {e}")
            return False

    def move_to_pos(self, position: int) -> bool:
        """移动到 0~1000 (0=夹紧, 1000=张开)。"""
        if not self._initialized or self._grip is None:
            return False
        position = int(max(0, min(1000, position)))
        try:
            self._grip.move_to_pos(position)
            return True
        except Exception as e:
            logger.warning(f"移动夹爪出错: {e}")
            return False

    def move_norm(self, value: float) -> bool:
        """归一化移动: value∈[0,1], 1.0=张开, 0.0=夹紧。"""
        value = max(0.0, min(1.0, float(value)))
        return self.move_to_pos(int(round(value * 1000)))

    def send_norm(self, value: float) -> None:
        """GripperBase 接口: 下发归一化目标 [0,1]。等价于 move_norm。"""
        self.move_norm(value)

    def read_pos(self) -> int:
        """读取当前位置 0~1000 (后台线程已在轮询缓存)。失败时返回上次有效值。"""
        if self._grip is None:
            return self._last_good_pos
        try:
            pos = self._grip.read_pos()
            if pos is None or pos < 0:
                return self._last_good_pos
            self._last_good_pos = int(pos)
            return self._last_good_pos
        except Exception:
            return self._last_good_pos

    def read_norm(self) -> float:
        """归一化位置: [0,1], 1.0=张开。"""
        return self.read_pos() / 1000.0

    def disconnect(self) -> None:
        if self._grip is not None:
            # 厂商 SDK grip_init 起的 _send_request 后台线程是 while True 且无停止开关,
            # close() 关掉连接后它仍每 20ms 发 CAN -> 刷屏 "Not connected to the server"。
            # 线程停不掉, 这里把该 CanClient 自己的 logger 压成 CRITICAL 静音其 teardown 噪声。
            try:
                self._grip.client.logger.setLevel(logging.CRITICAL)
            except Exception:
                pass
            try:
                self._grip.close()
            except Exception as e:
                logger.warning(f"关闭夹爪出错: {e}")
            self._grip = None
        self._connected = False
        self._initialized = False
        logger.info(f"夹爪 {self.server_address} 已断开")
