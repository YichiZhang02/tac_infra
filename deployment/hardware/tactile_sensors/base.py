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

"""触觉传感器 (Tactile Sensor) 统一接口。"""

import abc

from numpy.typing import NDArray


class TactileSensorBase(abc.ABC):
    """
    触觉传感器抽象基类。

    异步读取最新一帧触觉图 (形如 (H, W, C))。建议实现内部用独立进程/线程持续收流,
    async_read() 只取最新帧、非阻塞。
    实现示例: DmroboticsFlux (gRPC + UDP)。
    """

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """是否已连接 (收流进程是否存活)。"""
        ...

    @abc.abstractmethod
    def connect(self) -> None:
        """启动收流并等待首帧。"""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """停止收流并清理资源。"""
        ...

    @abc.abstractmethod
    def async_read(self) -> NDArray:
        """
        取最新一帧 (非阻塞)。无新帧时返回上次缓存, 从未收到则返回空帧。

        Returns:
            NDArray: 形如 (H, W, C) 的触觉图。
        """
        ...
