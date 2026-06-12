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
睿尔曼 RM75b 机械臂配置类 (HD 高清版，无触觉传感器)

基于 realman_tactile_shandd_hd，但去掉触觉传感器。

特性:
- 主臂（Leader）: 通过 USB 连接到电脑
- 从臂（Follower）: 通过网线 TCP/IP 连接（默认 192.168.1.201:8080）
- 夹爪: 舵机通过 USB 单独连接
- 相机: 手腕相机和全景相机通过 USB 连接

HD 版本特性:
- cam_top 全景相机使用 1920x1080 分辨率，ROI 裁剪输出 896x896 正方形
- cam_right_wrist 使用 640x480 分辨率，ROI 裁剪输出 480x480 正方形
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import Cv2Rotation

from ..config import RobotConfig


@RobotConfig.register_subclass("realman_rm75b_hd")
@dataclass
class RealmanRM75bHdConfig(RobotConfig):
    """睿尔曼 RM75b 机械臂配置类 (HD 高清版，无触觉传感器)
    
    Attributes:
        leader_port: 主臂 USB 串口路径
        follower_ip: 从臂 IP 地址
        follower_port: 从臂端口号
        gripper_port: 夹爪舵机 USB 串口路径
        gripper_baudrate: 夹爪舵机波特率
        disable_torque_on_disconnect: 断开连接时是否禁用力矩
        max_relative_target: 相对目标位置的最大值（安全限制）
        cameras: 相机配置字典
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

    # ============ 被夹物体配置 ============
    object: str = ""
    gripper_object_min_open: dict[str, float] = field(
        default_factory=lambda: {
            "eraser": 0.5800,
            "plug":   0.3300,
            "gear1":  0.1800,
            "gear2":  0.1400,
            "pen":    0.0400,
            "usb":    0.0100,
        }
    )

    # ============ 安全与控制配置 ============
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    
    # ============ 相机配置 (HD 版本) ============
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            # 手腕相机 (D405)
            "cam_right_wrist": OpenCVCameraConfig(
                index_or_path=12,  
                fps=30,
                width=640,
                height=480,
                rotation=Cv2Rotation.ROTATE_180,  # 相机倒装，旋转180度
            ),
            # 全景相机 (D515 RGB流) - HD 1920x1080
            # 代码中会做 ROI 裁剪，输出 896x896 正方形
            "cam_top": OpenCVCameraConfig(
                index_or_path=6,
                fps=30,
                width=1920,    # HD 分辨率
                height=1080,   # HD 分辨率
            ),
        }
    )
    
    # ============ 兼容性配置 ============
    use_degrees: bool = False
