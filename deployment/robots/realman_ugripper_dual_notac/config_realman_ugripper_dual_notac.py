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
睿尔曼 RM75b 双臂 (ugripper) — 无触觉版 (notac) 配置类。

与 realman_ugripper_dual 完全一致, 仅去掉 2 路触觉传感器:
    - 保留: 从臂 + 领控电爪 + 手腕鱼眼 + 顶部相机 cam_top
    - 去掉: dmrobotics Flux 触觉 (cam_finger0 / cam_finger1)

触觉相关配置字段仍保留 (继承自父类, 不使用), 以保持与父类一致, 无副作用。
"""

from dataclasses import dataclass

from ..config import RobotConfig
from ..realman_ugripper_dual.config_realman_ugripper_dual import RealmanUGripperDualConfig


@RobotConfig.register_subclass("realman_ugripper_dual_notac")
@dataclass
class RealmanUGripperDualNotacConfig(RealmanUGripperDualConfig):
    """睿尔曼 RM75b 双臂 (ugripper) 无触觉版配置。"""

    pass
