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
睿尔曼 RM75b 机械臂 - Shear+Depth 触觉版本 (HD 高清版)

该模块将触觉传感器的 shear 和 depth 数据合并为 RGB 图像，
以便与 LeRobot 的图像处理流程兼容。

HD 版本: stop相机使用 1920x1080 分辨率，ROI 输出 480x360
"""

from .config_realman_tactile_shandd_hd import RealmanTactileShanddHdConfig
from .realman_tactile_shandd_hd import RealmanTactileShanddHd

# 复用父模块的触觉传感器类
from ..realman_tactile_shandd.tactile_sensor_feat import TactileSensorFeat, TactileSensorFeatConfig

__all__ = [
    "RealmanTactileShanddHdConfig",
    "RealmanTactileShanddHd",
    "TactileSensorFeat",
    "TactileSensorFeatConfig",
]
