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
触觉传感器特征封装类 (Shear + Depth 模式)

封装 dmrobotics.Sensor SDK，使用 getFeat() 获取 shear 和 depth 数据，
合并为 RGB 图像格式以兼容 LeRobot 的视频编码流程。

输出格式:
- 通道 0 (B): depth, 归一化范围 [0, 4] -> [0, 255]
- 通道 1 (G): shear_x, 归一化范围 [-5, +5] -> [0, 255]
- 通道 2 (R): shear_y, 归一化范围 [-5, +5] -> [0, 255]
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TactileSensorFeatConfig:
    """触觉传感器特征配置类 (Shear + Depth 模式)
    
    Attributes:
        device_id: 设备索引或序列号
        fps: 帧率 (用于记录)
        width: 输出图像宽度 (getFeat 输出尺寸)
        height: 输出图像高度 (getFeat 输出尺寸)
        roi: 感兴趣区域，4个点的坐标，格式为 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        shear_min: shear 归一化最小值
        shear_max: shear 归一化最大值
        depth_min: depth 归一化最小值
        depth_max: depth 归一化最大值
    """
    device_id: int | str = 0
    fps: int = 30
    width: int = 320    # getFeat 输出尺寸 (可能与原始图像不同)
    height: int = 240   # getFeat 输出尺寸
    roi: list[list[int]] | None = None
    
    # 归一化参数 (基于实测数据)
    shear_min: float = -5.0
    shear_max: float = 5.0
    depth_min: float = 0.0
    depth_max: float = 4.0
    
    def __post_init__(self):
        # 默认 ROI (根据 tactile SDK 的默认值)
        if self.roi is None:
            self.roi = [[113, 111], [475, 116], [479, 376], [119, 399]]


class TactileSensorFeat:
    """
    触觉传感器特征类 (Shear + Depth 模式)
    
    使用 getFeat() 获取 shear 和 depth 数据，合并为 RGB 图像格式。
    
    RGB 通道含义:
    - B (通道0): depth (接触深度)
    - G (通道1): shear_x (X方向剪切力)
    - R (通道2): shear_y (Y方向剪切力)
    
    Usage:
        config = TactileSensorFeatConfig(device_id="M2505150275")
        sensor = TactileSensorFeat("tactile_left", config)
        sensor.connect()
        
        # 同步读取 (返回 RGB 图像)
        image = sensor.read()  # shape: (240, 320, 3), dtype: uint8
        
        # 异步读取 (从后台线程缓存)
        image = sensor.async_read()
        
        sensor.disconnect()
    """
    
    def __init__(self, name: str, config: TactileSensorFeatConfig):
        self.name = name
        self.config = config
        
        # SDK 实例
        self._sensor = None
        self._sdk_imported = False
        
        # 连接状态
        self._is_connected = False
        
        # 异步读取相关
        self._async_thread: threading.Thread | None = None
        self._async_running = False
        self._async_lock = threading.Lock()
        self._async_image: NDArray | None = None
        self._async_timestamp: float = 0.0
        
    def _import_sdk(self):
        """延迟导入 dmrobotics SDK"""
        if not self._sdk_imported:
            try:
                # 添加 dmrobotics 包路径 (deployment/sdk，使仓库自包含)
                import sys
                sdk_path = Path(__file__).resolve().parents[2] / "sdk"
                if str(sdk_path) not in sys.path:
                    sys.path.insert(0, str(sdk_path))

                from dmrobotics import Sensor as DMSensor
                self._DMSensor = DMSensor
                self._sdk_imported = True
            except ImportError as e:
                raise ImportError(
                    "无法导入 dmrobotics SDK。请确保 deployment/sdk/dmrobotics 存在且依赖已安装。"
                    f"原始错误: {e}"
                )
    
    @property
    def is_connected(self) -> bool:
        """检查传感器是否已连接"""
        return self._is_connected
    
    @property
    def height(self) -> int:
        return self.config.height
    
    @property
    def width(self) -> int:
        return self.config.width
    
    @property
    def channels(self) -> int:
        """输出 RGB 图像，3通道"""
        return 3
    
    def _normalize_shear(self, shear: NDArray) -> NDArray:
        """
        归一化 shear 数据
        
        Args:
            shear: 原始 shear 数据，shape (H, W, 2)
            
        Returns:
            归一化后的数据，范围 [0, 1]
        """
        shear_range = self.config.shear_max - self.config.shear_min
        return np.clip(
            (shear - self.config.shear_min) / shear_range, 
            0, 1
        )
    
    def _normalize_depth(self, depth: NDArray) -> NDArray:
        """
        归一化 depth 数据
        
        Args:
            depth: 原始 depth 数据，shape (H, W)
            
        Returns:
            归一化后的数据，范围 [0, 1]
        """
        depth_range = self.config.depth_max - self.config.depth_min
        return np.clip(
            (depth - self.config.depth_min) / depth_range, 
            0, 1
        )
    
    def connect(self, warmup: bool = True) -> None:
        """
        连接触觉传感器
        
        Args:
            warmup: 是否进行预热读取
        """
        if self._is_connected:
            logger.warning(f"触觉传感器 {self.name} 已连接")
            return
        
        self._import_sdk()
        
        # 构建 ROI 数组
        roi = np.array(self.config.roi, dtype="float32")
        
        logger.info(f"正在连接触觉传感器 {self.name} (device_id={self.config.device_id})...")
        
        try:
            self._sensor = self._DMSensor(
                dev_id=self.config.device_id,
                roi=roi,
            )
            
            # 等待传感器就绪
            max_wait = 5.0
            start_time = time.time()
            while self._sensor.getStatus() != 0:
                if time.time() - start_time > max_wait:
                    raise TimeoutError(f"触觉传感器 {self.name} 等待就绪超时")
                time.sleep(0.05)
            
            self._is_connected = True
            logger.info(f"触觉传感器 {self.name} 连接成功 (Shear+Depth 模式)")
            
            # 预热读取
            if warmup:
                logger.info(f"触觉传感器 {self.name} 预热中...")
                for _ in range(10):
                    self.read()
                    time.sleep(0.05)
                logger.info(f"触觉传感器 {self.name} 预热完成")
            
            # 启动异步读取线程
            self._start_async_read()
            
        except Exception as e:
            self._is_connected = False
            self._sensor = None
            raise ConnectionError(f"连接触觉传感器 {self.name} 失败: {e}")
    
    def disconnect(self) -> None:
        """断开触觉传感器连接"""
        if not self._is_connected:
            logger.warning(f"触觉传感器 {self.name} 未连接")
            return
        
        # 停止异步读取线程
        self._stop_async_read()
        
        if self._sensor is not None:
            try:
                self._sensor.disconnect()
            except Exception as e:
                logger.warning(f"断开触觉传感器 {self.name} 时出错: {e}")
            self._sensor = None
        
        self._is_connected = False
        logger.info(f"触觉传感器 {self.name} 已断开")
    
    def read(self) -> NDArray:
        """
        同步读取触觉传感器的 Shear+Depth 数据，返回 RGB 图像
        
        Returns:
            NDArray: RGB 图像，形状为 (height, width, 3), dtype: uint8
                    - 通道0 (B): depth
                    - 通道1 (G): shear_x
                    - 通道2 (R): shear_y
        """
        if not self._is_connected or self._sensor is None:
            raise RuntimeError(f"触觉传感器 {self.name} 未连接")
        
        # 检查传感器状态
        if self._sensor.getStatus() != 0:
            logger.warning(f"触觉传感器 {self.name} 状态异常")
        
        # 获取原始图像 (getFeat 需要)
        raw_img = self._sensor.getRawImage()
        
        # 使用 getFeat 获取 shear 和 depth
        # 返回值: (feat_img, deformation, depth, shear)
        _, _, depth, shear = self._sensor.getFeat(
            raw_img, 
            getdepth=True, 
            getshear=True
        )
        
        # 归一化
        norm_shear = self._normalize_shear(shear)   # (H, W, 2), range [0, 1]
        norm_depth = self._normalize_depth(depth)   # (H, W), range [0, 1]
        
        # 拼接为 RGB: depth, shear_x, shear_y (零位显示为蓝色)
        rgb = np.concatenate([
            norm_depth[..., np.newaxis],             # (H, W, 1) - depth
            norm_shear                               # (H, W, 2) - shear_x, shear_y
        ], axis=-1)                                  # (H, W, 3)
        
        # 转换为 uint8
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        return rgb_uint8
    
    def async_read(self, timeout_ms: float = 200.0) -> NDArray:
        """
        异步读取触觉传感器图像 (从后台线程缓存)
        
        Args:
            timeout_ms: 超时时间 (毫秒)
            
        Returns:
            NDArray: RGB 图像，形状为 (height, width, 3)
        """
        if not self._is_connected:
            raise RuntimeError(f"触觉传感器 {self.name} 未连接")
        
        with self._async_lock:
            if self._async_image is not None:
                # 检查数据是否过时
                age = (time.time() - self._async_timestamp) * 1000
                if age > timeout_ms:
                    logger.warning(f"触觉传感器 {self.name} 数据过时: {age:.0f}ms")
                return self._async_image.copy()
        
        # 如果没有异步数据，回退到同步读取
        logger.debug(f"触觉传感器 {self.name} 异步缓存为空，回退到同步读取")
        return self.read()
    
    def _start_async_read(self) -> None:
        """启动异步读取线程"""
        if self._async_running:
            return
        
        self._async_running = True
        self._async_thread = threading.Thread(
            target=self._async_read_loop,
            daemon=True,
            name=f"TactileFeatAsyncRead-{self.name}"
        )
        self._async_thread.start()
        logger.debug(f"触觉传感器 {self.name} 异步读取线程已启动")
    
    def _stop_async_read(self) -> None:
        """停止异步读取线程"""
        self._async_running = False
        if self._async_thread is not None:
            self._async_thread.join(timeout=1.0)
            self._async_thread = None
        logger.debug(f"触觉传感器 {self.name} 异步读取线程已停止")
    
    def _async_read_loop(self) -> None:
        """异步读取循环"""
        while self._async_running:
            try:
                if self._is_connected and self._sensor is not None:
                    if self._sensor.getStatus() == 0:
                        image = self.read()
                        with self._async_lock:
                            self._async_image = image
                            self._async_timestamp = time.time()
            except Exception as e:
                logger.debug(f"触觉传感器 {self.name} 异步读取错误: {e}")
            
            # 最小化等待，争取 60Hz
            # 实际频率取决于 getRawImage() + getFeat() 的处理时间
            time.sleep(0.001)
    
    def __repr__(self) -> str:
        return (f"TactileSensorFeat(name={self.name}, "
                f"device_id={self.config.device_id}, "
                f"size=({self.height}, {self.width}), "
                f"connected={self._is_connected})")


def make_tactile_sensors_feat_from_configs(
    configs: dict[str, TactileSensorFeatConfig]
) -> dict[str, TactileSensorFeat]:
    """
    从配置字典创建触觉传感器特征实例
    
    Args:
        configs: 传感器名称到配置的映射
        
    Returns:
        dict: 传感器名称到 TactileSensorFeat 实例的映射
    """
    sensors = {}
    for name, config in configs.items():
        sensors[name] = TactileSensorFeat(name, config)
    return sensors
