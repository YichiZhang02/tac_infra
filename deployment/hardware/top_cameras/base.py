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

"""顶部 / 全景相机 (Top Camera) 统一接口。"""

import abc

from numpy.typing import NDArray


class TopCameraBase(abc.ABC):
    """
    顶部 / 全景相机抽象基类。

    顶部相机通常是本机直连相机 (USB / RealSense / Reachy2)。承接原 deployment/cameras/
    的各后端实现。提供同步 read() 与非阻塞 async_read()。
    实现示例: OpenCVTopCamera, RealSenseTopCamera, Reachy2TopCamera。
    """

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """是否已连接。"""
        ...

    @abc.abstractmethod
    def connect(self) -> None:
        """打开相机并预热。"""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """关闭相机并清理资源。"""
        ...

    @abc.abstractmethod
    def read(self) -> NDArray:
        """同步读取一帧 (阻塞直到拿到新帧)。"""
        ...

    @abc.abstractmethod
    def async_read(self) -> NDArray:
        """
        取最新一帧 (非阻塞)。

        Returns:
            NDArray: 形如 (H, W, 3) 的 RGB 图。
        """
        ...
