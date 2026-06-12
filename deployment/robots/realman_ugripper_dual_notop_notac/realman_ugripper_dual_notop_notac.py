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
睿尔曼 RM75b 双臂 (ugripper) — 无顶部相机 + 无触觉版 (notop_notac) LeRobot 适配器。

继承 RealmanUGripperDualNotac (已去掉触觉, 数据流只跑鱼眼), 再通过配置把 cameras 置空
去掉顶部相机 cam_top。因此 observation.images 只剩 {side}_cam_wrist。
其余逻辑全部继承复用。
"""

from ..realman_ugripper_dual_notac.realman_ugripper_dual_notac import RealmanUGripperDualNotac
from .config_realman_ugripper_dual_notop_notac import RealmanUGripperDualNotopNotacConfig


class RealmanUGripperDualNotopNotac(RealmanUGripperDualNotac):
    """睿尔曼 RM75b 双臂 (ugripper) 无顶部相机 + 无触觉版。"""

    config_class = RealmanUGripperDualNotopNotacConfig
    name = "realman_ugripper_dual_notop_notac"
