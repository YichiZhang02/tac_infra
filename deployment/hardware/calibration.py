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
关节/电机标定数据结构。

原属 deployment/motors, 但 Robot / Teleoperator 基类的标定文件读写 (draccus dump/load
dict[str, MotorCalibration]) 都依赖它, 而 motors 子系统其余部分 (舵机总线) 本仓库未使用。
故把这个轻量数据类抽到 hardware 下作为共享类型, 让 motors/ 可以整体移除。
"""

from dataclasses import dataclass


@dataclass
class MotorCalibration:
    """单个关节/电机的标定参数 (draccus 可序列化)。"""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int
