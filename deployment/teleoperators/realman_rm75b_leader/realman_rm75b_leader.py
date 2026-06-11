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
睿尔曼 RM75b 主臂遥操作器

用于遥操作控制从臂，通过 USB 串口读取主臂关节位置作为动作目标
"""

import logging
import os
from functools import cached_property
from typing import Any

import numpy as np

from deployment.motors import MotorCalibration
from deployment.teleoperators.teleoperator import Teleoperator
from vtla.engine.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_realman_rm75b_leader import RealmanRM75bLeaderConfig

# 添加 Robotic_Arm SDK 路径 (deployment/sdk，使仓库自包含)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "sdk"))

logger = logging.getLogger(__name__)


class RealmanRM75bLeader(Teleoperator):
    """
    睿尔曼 RM75b 主臂遥操作器
    
    通过 USB 串口读取主臂位置，用于遥操作控制从臂
    
    数据格式:
    - action: 8D (7关节 + 1夹爪，弧度)
    """

    config_class = RealmanRM75bLeaderConfig
    name = "realman_rm75b_leader"
    
    # RM75b 是 7 自由度机械臂
    DOF = 7
    
    # 关节名称 (与已采集数据 dm_right_only 保持一致)
    JOINT_NAMES = [
        "main_joint1",
        "main_joint2", 
        "main_joint3",
        "main_joint4",
        "main_joint5",
        "main_joint6",
        "main_joint7",
    ]
    
    # 夹爪名称
    GRIPPER_NAME = "main_gripper"

    def __init__(self, config: RealmanRM75bLeaderConfig):
        super().__init__(config)
        self.config = config

        # 主臂通信接口
        self._leader_arm = None
        self._connected = False

        # 异步读取器 (async_read=True 时使用)
        self._async_reader = None

    # ==================== 特征定义 ====================
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征：7个关节 + 1个夹爪"""
        features = {joint: float for joint in self.JOINT_NAMES}
        features[self.GRIPPER_NAME] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征：主臂关节位置"""
        return self._motors_ft

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """反馈特征：无需反馈"""
        return {}

    # ==================== 连接状态 ====================
    
    @property
    def is_connected(self) -> bool:
        """检查主臂是否已连接"""
        return self._connected and self._leader_arm is not None

    @property
    def is_calibrated(self) -> bool:
        """主臂出厂已校准"""
        return True

    # ==================== 连接/断开 ====================
    
    def connect(self, calibrate: bool = True) -> None:
        """连接主臂"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已连接")

        # 检查串口是否存在
        if not os.path.exists(self.config.port):
            raise ConnectionError(
                f"主臂串口 {self.config.port} 不存在。"
                "请检查串口连接和 udev 规则配置。"
            )

        # 导入并创建主臂通信类
        from deployment.robots.realman_rm75b.leader_arm import LeaderArm
        
        logger.info(f"正在连接主臂 {self.config.port}...")
        self._leader_arm = LeaderArm(
            port=self.config.port,
            baudrate=self.config.baudrate,
            hex_data=self.config.hex_data,
        )
        self._leader_arm.connect()
        self._connected = True
        logger.info(f"主臂连接成功!")

        # 启动异步读取线程 (60Hz 全局采集模式)
        if self.config.async_read:
            self._start_async_reader()

        # 校准
        if not self.is_calibrated and calibrate:
            self.calibrate()

        # 配置
        self.configure()

    def disconnect(self) -> None:
        """断开主臂连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        # 停止异步读取线程
        if self._async_reader is not None:
            self._async_reader.stop()
            self._async_reader.join(timeout=1.0)
            logger.info(f"主臂异步读取器已停止 (FPS: {self._async_reader.current_fps:.1f})")
            self._async_reader = None

        if self._leader_arm is not None:
            self._leader_arm.disconnect()
            self._leader_arm = None

        self._connected = False
        logger.info(f"主臂已断开")

    # ==================== 校准 ====================
    
    def calibrate(self) -> None:
        """校准主臂（通常不需要）"""
        logger.info(f"主臂出厂已校准，跳过")
        
        # 创建默认校准数据
        self.calibration = {}
        for i, joint in enumerate(self.JOINT_NAMES):
            self.calibration[joint] = MotorCalibration(
                id=i + 1,
                drive_mode=0,
                homing_offset=0,
                range_min=-180,
                range_max=180,
            )
        self.calibration["gripper"] = MotorCalibration(
            id=8,
            drive_mode=0,
            homing_offset=0,
            range_min=0,
            range_max=100,
        )
        self._save_calibration()

    def configure(self) -> None:
        """配置主臂"""
        pass

    # ==================== 异步读取 ====================

    def _start_async_reader(self) -> None:
        """启动异步主臂读取线程"""
        from deployment.robots.realman_rm75b.async_readers import AsyncLeaderArmReader

        if self._async_reader is not None:
            self._async_reader.stop()
            self._async_reader.join(timeout=1.0)

        self._async_reader = AsyncLeaderArmReader(
            leader_arm=self._leader_arm,
            history_size=4,
        )
        self._async_reader.start()
        import time
        time.sleep(0.1)  # 等待首帧数据
        logger.info(f"主臂异步读取器已启动 (用于 60Hz 全局采集)")

    # ==================== 读取动作 ====================

    def get_action(self) -> dict[str, Any]:
        """
        读取主臂当前位置作为动作

        当 async_read=True 时，从异步缓存读取 (~0ms)；
        否则同步串口读取 (~22ms)。

        Returns:
            dict: 包含 7 个关节位置 + 1 个夹爪位置的字典
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        # 异步模式: 从缓存读取; 同步模式: 直接串口读取
        if self._async_reader is not None:
            positions = self._async_reader.get_position()
        else:
            positions = self._leader_arm.read_position()

        # 构建动作字典
        action = {}
        for i, joint in enumerate(self.JOINT_NAMES):
            if self.config.use_degrees:
                action[joint] = np.rad2deg(positions[i])
            else:
                action[joint] = positions[i]

        # 夹爪
        if self.config.use_degrees:
            action[self.GRIPPER_NAME] = np.rad2deg(positions[7])
        else:
            action[self.GRIPPER_NAME] = positions[7]

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        发送反馈到主臂（无需实现）
        
        主臂是被动读取设备，不需要发送反馈
        """
        pass
