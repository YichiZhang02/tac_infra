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

"""腕部相机 (Wrist Camera) 统一接口。"""

import abc

from numpy.typing import NDArray


class WristCameraBase(abc.ABC):
    """
    腕部相机抽象基类。

    腕部相机通常是"板子推流 + 本机收流"的远端相机 (如鱼眼 gRPC+UDP), 与本机直连的
    顶部 USB 相机接口不同, 故单独成类。异步读取最新一帧 RGB (H, W, 3)。
    实现示例: FisheyeGrpcCamera。
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
        取最新一帧 (非阻塞)。

        Returns:
            NDArray: 形如 (H, W, 3) 的 RGB 图。
        """
        ...
