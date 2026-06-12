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
睿尔曼 RM75b 双臂 (ugripper) — 无顶部相机 + 无触觉版 (notop_notac) 配置类。

与 realman_ugripper_dual 相比同时去掉:
    - 顶部全景相机 cam_top (cameras 置空)
    - 2 路触觉 (cam_finger0 / cam_finger1)
保留: 从臂 + 领控电爪 + 手腕鱼眼。
"""

from dataclasses import dataclass, field

from deployment.cameras import CameraConfig

from ..config import RobotConfig
from ..realman_ugripper_dual.config_realman_ugripper_dual import RealmanUGripperDualConfig


@RobotConfig.register_subclass("realman_ugripper_dual_notop_notac")
@dataclass
class RealmanUGripperDualNotopNotacConfig(RealmanUGripperDualConfig):
    """睿尔曼 RM75b 双臂 (ugripper) 无顶部相机 + 无触觉版配置。"""

    # 去掉顶部全景相机 cam_top
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    crop_4_3_cameras: list[str] = field(default_factory=list)
