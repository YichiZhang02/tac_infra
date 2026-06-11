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
睿尔曼 RM75b 机械臂配置类 (Shear+Depth 触觉传感器 HD 高清版)

该配置支持主从控制模式，并集成 Shear+Depth 触觉传感器：
- 主臂（Leader）: 通过 USB 连接到电脑
- 从臂（Follower）: 通过网线 TCP/IP 连接（默认 192.168.1.201:8080）
- 夹爪: 舵机通过 USB 单独连接
- 相机: 手腕相机和全景相机通过 USB 连接
- 触觉传感器: 两个触觉传感器，输出 Shear+Depth RGB 图像

HD 版本特性:
- cam_top 全景相机使用 1920x1080 分辨率
- ROI 裁剪输出 480x360 (与标清版相同)

触觉传感器输出格式:
- 通道 0 (B): shear_x (X方向剪切力), 归一化范围 [-5, +5] -> [0, 255]
- 通道 1 (G): shear_y (Y方向剪切力), 归一化范围 [-5, +5] -> [0, 255]
- 通道 2 (R): depth (接触深度), 归一化范围 [0, 4] -> [0, 255]
"""

from dataclasses import dataclass, field

from deployment.cameras import CameraConfig
from deployment.cameras.opencv import OpenCVCameraConfig

from ..config import RobotConfig
from ..realman_tactile_shandd.tactile_sensor_feat import TactileSensorFeatConfig


@RobotConfig.register_subclass("realman_tactile_shandd_hd")
@dataclass
class RealmanTactileShanddHdConfig(RobotConfig):
    """睿尔曼 RM75b 机械臂配置类 (Shear+Depth 触觉传感器 HD 高清版)
    
    Attributes:
        leader_port: 主臂 USB 串口路径
        follower_ip: 从臂 IP 地址
        follower_port: 从臂端口号
        gripper_port: 夹爪舵机 USB 串口路径
        gripper_baudrate: 夹爪舵机波特率
        disable_torque_on_disconnect: 断开连接时是否禁用力矩
        max_relative_target: 相对目标位置的最大值（安全限制）
        cameras: 相机配置字典
        tactile_sensors: 触觉传感器配置字典 (Shear+Depth 模式)
    """
    
    # ============ 主臂配置（Leader - USB 串口） ============
    leader_port: str = "/dev/ttyLeaderR"
    leader_baudrate: int = 460800
    leader_hex_data: str = "55 AA 02 00 00 67"
    connect_leader: bool = False
    
    # ============ 从臂配置（Follower - TCP/IP 网线） ============
    follower_ip: str = "192.168.1.201"
    follower_tcp_port: int = 8080
    
    # ============ 夹爪配置（Gripper - USB 舵机 STS3215） ============
    gripper_port: str = "/dev/ttyGripperFollowerR"
    gripper_baudrate: int = 115200
    gripper_motor_id: int = 1
    gripper_motor_model: str = "sts3215"
    gripper_open: float = -2.3562
    gripper_close: float = -0.198
    
    # ============ 安全与控制配置 ============
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    
    # ============ 相机配置 (HD 版本) ============
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            # 手腕相机 (D405)
            "cam_right_wrist": OpenCVCameraConfig(
                index_or_path=4,  # 注意: 索引可能因 USB 重插而变化
                fps=30,
                width=640,
                height=480,
            ),
            # 全景相机 (D515 RGB流) - HD 1920x1080
            # 代码中会做 ROI 裁剪，输出仍为 480x360
            "cam_top": OpenCVCameraConfig(
                index_or_path=16,
                fps=30,
                width=1920,    # HD 分辨率
                height=1080,   # HD 分辨率
            ),
        }
    )
    
    # ============ 触觉传感器配置 (Shear+Depth 模式) ============
    # 输出 RGB 图像: (H, W, 3), 其中 B=shear_x, G=shear_y, R=depth
    tactile_sensors: dict[str, TactileSensorFeatConfig] = field(
        default_factory=lambda: {
            # 触觉传感器 0 - 序列号 M2505150275
            "cam_finger0": TactileSensorFeatConfig(
                device_id=6,  # 使用 index
                fps=30,
                width=320,    # getFeat 输出尺寸
                height=240,
                # 归一化参数
                shear_min=-5.0,
                shear_max=5.0,
                depth_min=0.0,
                depth_max=4.0,
            ),
            # 触觉传感器 1 - 序列号 M2505150108
            "cam_finger1": TactileSensorFeatConfig(
                device_id=8,  # 使用 index
                fps=30,
                width=320,
                height=240,
                shear_min=-5.0,
                shear_max=5.0,
                depth_min=0.0,
                depth_max=4.0,
            ),
        }
    )
    
    # ============ 兼容性配置 ============
    use_degrees: bool = False
