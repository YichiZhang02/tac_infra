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

"""主臂 (Leader Arm) 统一接口。"""

import abc

import numpy as np


class LeaderArmBase(abc.ABC):
    """
    主臂 (遥操作输入设备) 抽象基类。

    主臂只读不写: 读取人手操作的关节位置, 由 teleoperator 转成 action 下发给从臂。
    实现示例: RealmanLeader (USB 串口)。
    """

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """是否已连接。"""
        ...

    @abc.abstractmethod
    def connect(self) -> None:
        """建立连接 (打开串口 / 握手等)。"""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接并清理资源。"""
        ...

    @abc.abstractmethod
    def read_position(self) -> np.ndarray:
        """
        读取当前关节位置 (含夹爪)。

        Returns:
            np.ndarray: 长度 = 关节数 + 夹爪数 的一维数组, 单位由实现决定 (通常弧度)。
        """
        ...
