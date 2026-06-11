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
异步主臂位置读取器 (从 realman_tactile_shandd_high 中抽取，使 deployment 自包含)

在独立线程中持续读取主臂关节位置，与主循环解耦，避免串口读取 (~22ms) 阻塞主循环。
"""

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class AsyncLeaderArmReader(threading.Thread):
    """
    异步主臂位置读取器

    在独立线程中持续读取主臂关节位置，实现与主循环的解耦。
    主臂 read_position() 实测 ~22ms -> 最高 ~45Hz，
    在高频主循环中需要异步读取以避免阻塞。

    LeaderArm.read_position() 返回 np.ndarray(8,)，单位为弧度 (7关节 + 1夹爪)。
    本类直接缓存原始弧度值，不做单位转换，由调用方按需转换。
    """

    def __init__(self, leader_arm, history_size: int = 4):
        super().__init__(daemon=True, name="LeaderArmReader")
        self.leader_arm = leader_arm
        self.running = True
        self._lock = threading.Lock()

        # 缓存的位置 (8D: 7关节 + 1夹爪，弧度)
        self._position = np.zeros(8, dtype=np.float32)
        self._last_update = 0.0
        self._read_count = 0
        self._current_fps = 0.0
        self._fps_counter = 0
        self._fps_start = 0.0

        # 历史帧缓存
        self._history_size = history_size
        self._history_buffer: list[tuple[np.ndarray, float]] = []  # [(position, timestamp), ...]

    def run(self):
        """后台持续读取主臂位置 (全速读取，实测 ~45Hz)"""
        self._fps_start = time.time()

        while self.running:
            try:
                # 读取主臂位置 (实测 ~22ms)
                position = self.leader_arm.read_position()  # np.ndarray(8,), radians

                timestamp = time.time()

                with self._lock:
                    self._position = position.copy()
                    self._last_update = timestamp
                    self._read_count += 1

                    # 添加到历史缓存
                    self._history_buffer.append((position.copy(), timestamp))
                    while len(self._history_buffer) > self._history_size:
                        self._history_buffer.pop(0)

                self._fps_counter += 1

                # 每秒更新帧率统计
                now = time.time()
                if now - self._fps_start >= 1.0:
                    self._current_fps = self._fps_counter / (now - self._fps_start)
                    self._fps_counter = 0
                    self._fps_start = now
                    logger.debug(f"主臂位置读取频率: {self._current_fps:.1f} Hz, 历史缓存: {len(self._history_buffer)}")

            except Exception as e:
                logger.debug(f"异步读取主臂位置错误: {e}")

    def get_position(self) -> np.ndarray:
        """获取缓存的主臂位置 (最新一帧，弧度)"""
        with self._lock:
            return self._position.copy()

    def get_position_age(self) -> float:
        """获取位置数据的年龄 (秒)"""
        with self._lock:
            return time.time() - self._last_update if self._last_update > 0 else float('inf')

    def clear_history(self):
        """清空历史缓存"""
        with self._lock:
            self._history_buffer.clear()

    @property
    def current_fps(self) -> float:
        """当前实际帧率"""
        return self._current_fps

    @property
    def history_count(self) -> int:
        """当前历史缓存帧数"""
        with self._lock:
            return len(self._history_buffer)

    def stop(self):
        """停止读取线程"""
        self.running = False
