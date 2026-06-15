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
顶部 / 全景 USB 相机 (OpenCV VideoCapture)。

精简自包含实现: cv2.VideoCapture + 后台读线程, async_read() 取最新帧 (非阻塞)。
只保留实际在用的配置项 (index_or_path / fps / width / height / color_mode / rotation /
fourcc / warmup_s), 不含原 lerobot 版的视频文件回放、find_cameras 等分支。

配置驱动: robot 用 make_top_cameras_from_configs(config.cameras) 构建相机字典。
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .base import TopCameraBase

logger = logging.getLogger(__name__)


@dataclass
class OpenCVTopCameraConfig:
    """OpenCV USB 相机配置。

    Attributes:
        index_or_path: 相机设备索引 (int, 如 6 -> /dev/video6) 或视频设备路径。
        fps:           请求帧率。
        width/height:  请求分辨率 (像素)。
        color_mode:    "rgb" (默认) 或 "bgr"; cv2 原生 BGR, "rgb" 时转换。
        rotation:      顺时针旋转角度, 取 0 / 90 / 180 / 270。
        fourcc:        视频编码 FOURCC (如 "MJPG"), 高分辨率 USB 提帧率用; None=自动。
        warmup_s:      connect() 后丢弃读帧预热的时长 (秒)。
    """

    index_or_path: int | str | Path
    fps: int
    width: int
    height: int
    color_mode: str = "rgb"
    rotation: int = 0
    fourcc: str | None = None
    warmup_s: float = 1.0

    def __post_init__(self):
        if self.color_mode not in ("rgb", "bgr"):
            raise ValueError(f"color_mode 须为 'rgb'/'bgr', 收到 {self.color_mode}")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError(f"rotation 须为 0/90/180/270, 收到 {self.rotation}")


_ROTATION_FLAG = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class OpenCVTopCamera(TopCameraBase):
    """OpenCV USB 相机, 后台线程持续读、async_read 取最新帧。"""

    def __init__(self, config: OpenCVTopCameraConfig, name: str = "cam_top"):
        self.name = name
        self.config = config

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: NDArray | None = None

    def __str__(self) -> str:
        return f"OpenCVTopCamera({self.config.index_or_path})"

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def connect(self) -> None:
        if self.is_connected:
            logger.warning(f"[{self.name}] 已连接")
            return

        idx = self.config.index_or_path
        cap = cv2.VideoCapture(int(idx) if isinstance(idx, int) else str(idx))
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(
                f"无法打开 OpenCVTopCamera({idx})。请用 `ls /dev/video*` 确认设备/索引。"
            )

        if self.config.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._cap = cap

        # 预热: 丢弃前若干帧, 等曝光/自动白平衡稳定
        t0 = time.time()
        while time.time() - t0 < self.config.warmup_s:
            self._cap.read()

        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name=f"cam-{self.name}")
        self._thread.start()
        logger.info(f"[{self.name}] 已连接 ({self.config.width}x{self.config.height}@{self.config.fps})")

    def _postprocess(self, frame: NDArray) -> NDArray:
        if self.config.color_mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.config.rotation:
            frame = cv2.rotate(frame, _ROTATION_FLAG[self.config.rotation])
        return frame

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None:
                break
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            img = self._postprocess(frame)
            with self._lock:
                self._latest = img

    def read(self) -> NDArray:
        """同步读取一帧 (阻塞直到拿到新帧)。"""
        if not self.is_connected:
            raise ConnectionError(f"[{self.name}] 未连接")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"[{self.name}] 读取失败")
        return self._postprocess(frame)

    def async_read(self) -> NDArray:
        """取后台线程缓存的最新帧 (非阻塞); 从未拿到则返回黑帧。"""
        with self._lock:
            if self._latest is not None:
                return self._latest
        return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info(f"[{self.name}] 已断开")
