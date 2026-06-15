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

"""从臂 (Follower Arm) 统一接口。"""

import abc
from collections.abc import Sequence

import numpy as np


class FollowerArmBase(abc.ABC):
    """
    从臂 (被控机械臂本体) 抽象基类。

    读当前关节状态 + 下发关节目标。建议实现内部用后台线程异步缓存关节状态,
    使 read_joints() 非阻塞。
    实现示例: RealmanTcpFollower (睿尔曼 SDK over TCP)。
    """

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """是否已连接。"""
        ...

    @abc.abstractmethod
    def connect(self) -> None:
        """建立连接。"""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接并清理资源。"""
        ...

    @abc.abstractmethod
    def read_joints(self) -> np.ndarray:
        """
        读取当前关节位置。

        Returns:
            np.ndarray: 长度 = 关节数 的一维数组, 单位由实现决定 (通常弧度)。
        """
        ...

    @abc.abstractmethod
    def send_joints(self, positions: Sequence[float]) -> None:
        """
        下发关节目标位置。

        Args:
            positions: 长度 = 关节数 的目标位置, 单位需与 read_joints 一致。
        """
        ...
