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

"""夹爪 (Gripper) 统一接口。"""

import abc


class GripperBase(abc.ABC):
    """
    夹爪抽象基类。

    读/写统一用归一化开合度 [0, 1] (约定 1 = 完全张开, 0 = 完全夹紧), 由各实现
    负责与真实编码器行程之间的换算 (左右爪行程不同等)。
    实现示例: LingkongGripper (gRPC/CAN), Rm75bGripper。
    """

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """是否已连接。"""
        ...

    @abc.abstractmethod
    def connect(self) -> None:
        """连接夹爪 (可能包含夹紧自标定)。"""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接并清理资源。"""
        ...

    @abc.abstractmethod
    def read_norm(self) -> float:
        """读取当前开合度, 归一化到 [0, 1]。"""
        ...

    @abc.abstractmethod
    def send_norm(self, value: float) -> None:
        """
        下发目标开合度。

        Args:
            value: 归一化目标 [0, 1], 实现应自行 clip 到合法范围。
        """
        ...
