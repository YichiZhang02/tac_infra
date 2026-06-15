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
deployment.hardware —— 所有物理硬件封装的唯一来源。

设计目标: robots/ 与 teleoperators/ 不再各自藏一份硬件代码, 而是直接
从这里 import 具体硬件类、new 出来、调用。每类硬件一个子包, 子包内可以
有多种实现 (不同厂商/接口), 全部实现该类的 base.py 抽象接口, 可互换。

子包 (按"角色"划分):
    leader_arms/      1 主臂   (遥操作输入设备, 只读)
    follower_arms/    2 从臂   (被控机械臂本体, 读关节 + 下发目标)
    grippers/         3 夹爪   (读/写归一化开合度)
    tactile_sensors/  4 触觉   (异步读最新帧)
    wrist_cameras/    5 wrist 相机 (异步读最新帧)
    top_cameras/      6 top 相机   (异步读最新帧)

约定:
    - 硬件类只接收"原始参数"(ip/port/baudrate 等), 不依赖 RobotConfig /
      TeleoperatorConfig, 以保持与上层解耦。
    - 硬件类不得反向 import robots/ 或 teleoperators/ (避免循环依赖)。
    - 厂商 SDK 的 sys.path 接入统一走 _sdk_paths.py。
"""
