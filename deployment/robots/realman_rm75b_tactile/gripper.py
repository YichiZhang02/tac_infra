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
睿尔曼 RM75b 夹爪控制模块

使用 Feetech STS3215 舵机控制夹爪。
传动比与之前 ROS-LeRobot 框架保持一致。
"""

import logging
import os
import threading
import time
from typing import Any

import numpy as np

from deployment.motors import Motor, MotorNormMode
from deployment.motors.feetech import FeetechMotorsBus

logger = logging.getLogger(__name__)


# ==================== 传动比转换函数 ====================
# 与之前 ROS-LeRobot 框架保持一致
# 注意：这里有两套不同的转换关系
# - convert_degrees_to_steps: 用于发送命令，SCALE = 75.0
# - step2degree: 用于读取位置，使用标准 180.0

GRIPPER_SCALE = 75.0


def degrees_to_steps(degrees: float) -> int:
    """
    角度转步数 (用于发送命令)
    
    与 ROS 框架中的 convert_degrees_to_steps 保持一致：
    def convert_degrees_to_steps(degrees):
        SCALE = 75.0  
        return int(-degrees / SCALE * 2048 + 2048)
    """
    return int(-degrees / GRIPPER_SCALE * 2048 + 2048)


def steps_to_degrees(steps: int) -> float:
    """
    步数转角度 (用于读取位置)
    
    与 ROS 框架中的 step2degree 保持一致：
    def step2degree(steps):
        return (steps - 2048) / 2048.0 * 180.0
    """
    return (steps - 2048) / 2048.0 * 180.0


def radians_to_steps(radians: float) -> int:
    """弧度转步数 (用于发送命令)"""
    degrees = np.rad2deg(radians)
    return degrees_to_steps(degrees)


def steps_to_radians(steps: int) -> float:
    """步数转弧度 (用于读取位置)"""
    degrees = steps_to_degrees(steps)
    return np.deg2rad(degrees)


class Gripper:
    """
    夹爪控制类
    
    使用 Feetech STS3215 舵机，传动比与 ROS-LeRobot 框架保持一致。
    """
    
    def __init__(
        self,
        port: str,
        motor_id: int = 1,
        motor_model: str = "sts3215",
        baudrate: int = 115200,
    ):
        """
        初始化夹爪
        
        Args:
            port: 串口路径，如 /dev/ttyFollowerR
            motor_id: 舵机 ID，默认 1
            motor_model: 舵机型号，默认 sts3215
            baudrate: 波特率，默认 115200
        """
        self.port = port
        self.motor_id = motor_id
        self.motor_model = motor_model
        self.baudrate = baudrate
        
        self._bus: FeetechMotorsBus | None = None
        self._is_connected = False
        self._last_valid_position: float = 0.0
        self._read_fail_count: int = 0

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._is_connected and self._bus is not None and self._bus.is_connected

    def connect(self) -> None:
        """连接夹爪舵机"""
        if self.is_connected:
            logger.warning(f"夹爪 {self.port} 已连接")
            return
        
        if not os.path.exists(self.port):
            raise FileNotFoundError(f"夹爪串口 {self.port} 不存在")
        
        logger.info(f"正在连接夹爪 {self.port}...")
        
        try:
            # 创建 FeetechMotorsBus 实例
            self._bus = FeetechMotorsBus(
                port=self.port,
                motors={
                    "gripper": Motor(
                        id=self.motor_id,
                        model=self.motor_model,
                        norm_mode=MotorNormMode.RANGE_0_100,  # 夹爪使用 0-100 范围
                    ),
                },
            )
            
            # 连接
            self._bus.connect()
            
            # 配置舵机
            self._configure()
            
            self._is_connected = True
            logger.info(f"夹爪连接成功!")
            
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"夹爪连接失败: {e}")

    def _configure(self) -> None:
        """配置夹爪舵机"""
        if self._bus is None:
            return
        
        try:
            # 设置为位置模式 (Mode = 0)
            self._bus.write("Operating_Mode", "gripper", 0)
            # 设置加速度
            self._bus.write("Acceleration", "gripper", 254)
            logger.debug("夹爪配置完成")
        except Exception as e:
            logger.warning(f"夹爪配置失败: {e}")

    def disconnect(self) -> None:
        """断开连接"""
        if self._bus is not None:
            try:
                self._bus.disconnect()
            except Exception as e:
                logger.warning(f"断开夹爪时出错: {e}")
            self._bus = None
        
        self._is_connected = False
        logger.info(f"夹爪 {self.port} 已断开")

    def read_position(self) -> float:
        """
        读取夹爪当前位置

        Returns:
            float: 夹爪位置 (弧度)，失败时返回上一次有效值
        """
        if not self.is_connected or self._bus is None:
            return self._last_valid_position

        try:
            # 读取原始步数 (不使用归一化，直接读取原始值)
            result = self._bus.sync_read("Present_Position", normalize=False)
            raw_steps = result.get("gripper", 2048)

            # 转换为弧度 (使用与 ROS 框架一致的传动比)
            radians = steps_to_radians(raw_steps)
            self._last_valid_position = float(radians)
            return self._last_valid_position

        except Exception as e:
            self._read_fail_count += 1
            if self._read_fail_count <= 3 or self._read_fail_count % 100 == 0:
                logger.warning(f"读取夹爪位置失败 (第{self._read_fail_count}次): {e}")
            return self._last_valid_position

    def write_position(self, position: float) -> None:
        """
        设置夹爪目标位置
        
        Args:
            position: 目标位置 (弧度)
        """
        if not self.is_connected or self._bus is None:
            return
        
        try:
            # 弧度转步数 (使用与 ROS 框架一致的传动比)
            steps = radians_to_steps(position)
            
            # 限制范围 [0, 4096]
            steps = max(0, min(4096, steps))
            
            # 写入目标位置 (不使用归一化，直接写入原始值)
            self._bus.sync_write("Goal_Position", {"gripper": steps}, normalize=False)
            
        except Exception as e:
            logger.warning(f"设置夹爪位置失败: {e}")

    def set_torque(self, enable: bool) -> None:
        """
        设置夹爪力矩使能
        
        Args:
            enable: True 启用力矩，False 禁用
        """
        if not self.is_connected or self._bus is None:
            return
        
        try:
            self._bus.write("Torque_Enable", "gripper", 1 if enable else 0)
        except Exception as e:
            logger.warning(f"设置夹爪力矩失败: {e}")

    def __del__(self):
        """析构时断开连接"""
        self.disconnect()


class AsyncGripperHandler(threading.Thread):
    """
    异步夹爪管理器

    在后台线程中持续读取夹爪位置，主循环通过 get_position() 获取缓存值（零延迟）。
    写入操作通过 _bus_lock 与后台读取互斥，避免半双工串口冲突。

    用法:
        handler = AsyncGripperHandler(gripper, read_interval=0.02)
        handler.start()
        pos = handler.get_position()       # 主循环读取（不碰串口）
        handler.set_position(target_rad)   # 主循环写入（通过锁协调）
        handler.stop()
    """

    def __init__(self, gripper: Gripper, read_interval: float = 0.02):
        """
        Args:
            gripper: 已连接的 Gripper 实例
            read_interval: 后台读取间隔 (秒)，默认 0.02 = 50Hz
        """
        super().__init__(daemon=True, name="AsyncGripperHandler")
        self.gripper = gripper
        self.read_interval = read_interval
        self.running = True
        self._bus_lock = threading.Lock()
        self._cached_position: float = gripper._last_valid_position
        self._last_update: float = 0.0
        self._read_count: int = 0

    def run(self):
        """后台持续读取夹爪位置"""
        while self.running:
            try:
                with self._bus_lock:
                    pos = self.gripper.read_position()
                self._cached_position = pos
                self._last_update = time.time()
                self._read_count += 1
            except Exception as e:
                logger.debug(f"异步读取夹爪位置错误: {e}")
            time.sleep(self.read_interval)

    def get_position(self) -> float:
        """获取缓存的夹爪位置（非阻塞，不碰串口）"""
        return self._cached_position

    def set_position(self, position: float) -> None:
        """写入夹爪目标位置（通过锁与后台读取互斥）"""
        with self._bus_lock:
            self.gripper.write_position(position)

    def get_state_age(self) -> float:
        """获取缓存数据的年龄 (秒)"""
        if self._last_update == 0.0:
            return float('inf')
        return time.time() - self._last_update

    def stop(self):
        """停止后台线程"""
        self.running = False
