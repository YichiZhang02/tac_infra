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
厂商 SDK 的 sys.path 统一接入点。

历史上每个硬件封装都自己写一段 `sys.path.insert(...)` 去 deployment/sdk/
下找厂商库, 路径写法各异、还出过指错目录的 bug。这里集中成幂等的 ensure_*
函数, 各硬件实现在 import 厂商库之前调用对应函数即可。

SDK 目录布局 (deployment/sdk/):
    Robotic_Arm/              睿尔曼机械臂 SDK      -> import Robotic_Arm.*
    dm_lingkong_grip/         领控电爪客户端        -> import dm_lingkong_grip_sdk.*
    fish_camera_client/       鱼眼相机 gRPC 客户端  -> 扁平 import
    dmrobotics/               Flux 触觉传感器 SDK   -> import dmrobotics.*
"""

import sys
from pathlib import Path

# deployment/hardware/_sdk_paths.py -> parents[1] = deployment/
SDK_ROOT: Path = Path(__file__).resolve().parents[1] / "sdk"


def _ensure_on_path(directory: Path) -> None:
    """幂等地把 directory 加到 sys.path 最前 (目录不存在则静默跳过, 留给调用方报错)。"""
    s = str(directory)
    if directory.is_dir() and s not in sys.path:
        sys.path.insert(0, s)


def ensure_realman_sdk() -> None:
    """睿尔曼机械臂 SDK: from Robotic_Arm.rm_robot_interface import RoboticArm"""
    _ensure_on_path(SDK_ROOT)


def ensure_lingkong_sdk() -> None:
    """领控电爪: from dm_lingkong_grip_sdk import LingkongGrip"""
    _ensure_on_path(SDK_ROOT / "dm_lingkong_grip")


def ensure_fisheye_sdk() -> None:
    """鱼眼相机 gRPC 客户端 (扁平 import)。"""
    _ensure_on_path(SDK_ROOT / "fish_camera_client")


def ensure_dmrobotics_sdk() -> None:
    """Flux 触觉传感器: import dmrobotics.*"""
    _ensure_on_path(SDK_ROOT)
