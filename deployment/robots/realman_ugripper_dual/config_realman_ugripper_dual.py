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
睿尔曼 RM75b 双臂 (ugripper 集成版) 配置类

每条手臂的设备:
    - 从臂 (Follower): TCP/IP 网线  (左 192.168.1.200 / 右 192.168.1.201, 端口 8080)
    - 领控电爪 (Gripper): gRPC/CAN  (左 192.168.1.10:55551 / 右 192.168.1.11:55551)
    - 手腕鱼眼相机: fish_camera gRPC(50088) + UDP, 原生 1920x1080
    - 触觉传感器 x2: dmrobotics Flux gRPC(50051/50052), 输出 (240,320,3) uint16

主臂 (Leader) 不在本机器人内, 由 bi_realman_ugripper_leader 遥操作器负责。

observation / action 字段统一加 left_ / right_ 前缀:
    observation.state : left_main_joint1..7, left_main_gripper, right_...
    observation.images: left_cam_wrist, left_cam_finger0/1, right_...
    action            : left_main_joint1..7, left_main_gripper, right_...  (夹爪归一化 [0,1])
"""

from dataclasses import dataclass, field

from deployment.hardware.top_cameras import OpenCVTopCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("realman_ugripper_dual")
@dataclass
class RealmanUGripperDualConfig(RobotConfig):
    """睿尔曼 RM75b 双臂 (ugripper) 配置。"""

    # ============ 启用的手臂 ============
    # 可选 ["left"], ["right"], 或 ["left", "right"]
    arms: list[str] = field(default_factory=lambda: ["left", "right"])

    # ============ 硬件存在性开关 (物理 rig 有什么就开什么) ============
    # 触觉传感器: True=连接并产出 *_cam_finger0/1 观测; False=完全跳过 (原 _notac 变体)。
    # 顶部相机: 由下方 cameras 配置控制 (空字典 = 无 top, 原 _notop 变体)。
    # 关节 state 永远产出, 不设开关 (控制/安全都需要; 模型是否消费由 checkpoint 决定)。
    use_tactile: bool = True

    # ============ 从臂 (Follower - TCP/IP) ============
    left_follower_ip: str = "192.168.1.200"
    right_follower_ip: str = "192.168.1.201"
    follower_tcp_port: int = 8080

    # ============ 每臂板子 IP (鱼眼/触觉/夹爪代理) ============
    left_board_ip: str = "192.168.1.10"
    right_board_ip: str = "192.168.1.11"

    # ============ 领控电爪 (gRPC/CAN via 板子) ============
    gripper_grpc_port: int = 55551
    gripper_can_interface: str = "can0"
    gripper_can_bitrate: int = 1_000_000
    gripper_speed: int = 40    # 10~100
    gripper_torque: int = 50   # 10~100
    # 真实满行程覆盖 (编码器计数): 解决左右爪传动比不同。
    # SDK grip_init 把 max_itinerary 写死成 25000/90000, 但实测左右爪不同
    # (measure_gripper_stroke.py: 左≈92218, 右≈113972)。None 表示用 SDK 自动值。
    left_gripper_itinerary: int | None = 92218
    right_gripper_itinerary: int | None = 113972

    # ============ 手腕鱼眼相机 (fish_camera gRPC + UDP) ============
    fisheye_grpc_port: int = 50088
    # 每臂占用一个本机 UDP 接收端口 (必须各不相同)
    left_fisheye_udp_port: int = 50100
    right_fisheye_udp_port: int = 50101
    fisheye_width: int = 1920
    fisheye_height: int = 1080
    fisheye_max_datagram: int = 1200

    # ============ 触觉传感器 (dmrobotics Flux gRPC) ============
    pc_host: str = "192.168.1.120"          # 本机 IP, 用于 UDP 回传
    tactile0_grpc_port: int = 50051
    tactile1_grpc_port: int = 50052
    tactile0_dev_id: int = 0
    tactile1_dev_id: int = 2
    # 每路触觉占用一个本机 UDP 接收端口 (必须各不相同)
    left_tactile0_pc_port: int = 60000
    left_tactile1_pc_port: int = 60001
    right_tactile0_pc_port: int = 60002
    right_tactile1_pc_port: int = 60003
    # 输出尺寸 (须与 dmrobotics getDepth/getDeformation2D 实际输出一致)
    # 实测这批 Flux 传感器输出 384x288 (见 record 日志首帧 shape=(288,384,3))
    tactile_width: int = 384
    tactile_height: int = 288
    # 触觉 uint8 归一化编码 (float32 -> uint8): 每通道按 [min,max] 线性映射到 [0,255]。
    # 通道: B=depth, G=deform_x, R=deform_y (见 _tactile_worker)。
    # ⚠️ 以下范围需与传感器实测量级 / 训练时一致, 否则会饱和或丢失动态范围。
    tactile_depth_min: float = 0.0
    tactile_depth_max: float = 4.0
    tactile_deform_min: float = -1.0
    tactile_deform_max: float = 1.0

    # ============ 数据流首帧等待超时 ============
    stream_first_frame_timeout: float = 5.0

    # ============ 数据流限速 ============
    # ⚠️ 跳帧式限速的"量化陷阱": 从 ~50fps 的鱼眼源按 N 限速, 会被量化成源帧率的整数
    # 分之一。设 30~49 之间任意值都会变成"每隔一帧"= ~25fps (< 30Hz 消费) -> 手腕重复帧
    # 卡顿。实测 (analyze 鱼眼视频): 设 45 时唯一帧只有 25fps。
    # 因此只能取: 0 = 不限速(鱼眼跑满 ~50-59fps, 触觉 ~110fps), 或 >= 源帧率(>=60)。
    # 保证产出 > 录制 30fps, 主循环每帧都有新帧。28 核机器满速 CPU 仅 ~24%, 充裕。
    # 若想省 CPU 又不卡, 用 640x360/480 鱼眼(满速也很轻), 别回到 30~49 的限速。
    stream_max_fps: float = 0.0

    # ============ 数据流诊断 ============
    # True 时每个 worker 每 ~2s 打印产出fps/解码耗时/UDP到达率, 用于排查瓶颈
    # (板子+网线 vs PC解码)。正常录制请保持 False。
    stream_debug_fps: bool = False

    # ============ 安全与控制 ============
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = False  # False = 弧度制

    # ============ 额外本地 USB 相机 ============
    # 顶部全景相机 (cam_top), 参考 realman_tactile_shandd_hd_tac16
    cameras: dict[str, OpenCVTopCameraConfig] = field(
        default_factory=lambda: {
            "cam_top": OpenCVTopCameraConfig(
                index_or_path=6,
                fps=30,
                width=1920,
                height=1080,
            ),
        }
    )
    # 需要做"中心裁剪到 4:3"的相机名 (居中裁出 宽:高 = 4:3 的区域)。
    # cam_top 1920x1080 -> 1440x1080。
    crop_4_3_cameras: list[str] = field(default_factory=lambda: ["cam_top"])
