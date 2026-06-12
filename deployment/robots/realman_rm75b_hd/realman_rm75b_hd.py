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
睿尔曼 RM75b 机械臂 LeRobot 适配器 (HD 高清版，无触觉传感器)

基于 realman_tactile_shandd_hd，但去掉触觉传感器。

该适配器支持主从控制模式：
- 主臂（Leader）: 用于遥操作，读取位置作为目标
- 从臂（Follower）: 执行动作，跟随主臂运动
- 夹爪: 独立舵机控制
- 相机: 多相机图像采集

HD 版本特性:
- cam_top 全景相机使用 1920x1080 分辨率
- ROI 裁剪输出 896x896 正方形 (1:1 画幅)
- cam_right_wrist 输出 480x480 正方形

优化特性:
- 异步读取从臂状态 (解决 rm_get_joint_degree ~50ms 延迟问题)
- 支持 30Hz 数据采集频率
"""

import logging
import sys
import time
import threading
from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_realman_rm75b_hd import RealmanRM75bHdConfig

# 本地组件
from .gripper import Gripper
from .leader_arm import LeaderArm

# 添加 Robotic_Arm SDK 路径
sys.path.append(str(Path(__file__).resolve().parents[4]))

logger = logging.getLogger(__name__)

# ==================== 1:1 正方形 ROI 配置 ====================
# 目的: 充分利用 Pi0 的 224x224 输入，无黑边浪费
# 直接裁取，不做 resize，保留细节

# cam_top (D515): 1920x1080 -> 直接裁取底部 896x896
# 896 = 224 × 4，缩放到 Pi0 输入时无插值损失
# 位置: 水平居中，底边对齐（上半部分画面乱，只保留下半部分）
CAM_TOP_HD_ROI_WIDTH = 896
CAM_TOP_HD_ROI_HEIGHT = 896
CAM_TOP_HD_ROI_X_START = 512   # (1920 - 896) / 2 = 512
CAM_TOP_HD_ROI_Y_START = 184   # 1080 - 896 = 184 (底边对齐)
CAM_TOP_ROI_OUTPUT_SIZE = 896  # 输出 896x896 (无需resize)

# cam_right_wrist (D405): 640x480 -> 480x480 正方形 ROI
# 位置: 水平居中
CAM_WRIST_ROI_WIDTH = 480
CAM_WRIST_ROI_HEIGHT = 480
CAM_WRIST_ROI_X_START = 80     # (640 - 480) / 2 = 80
CAM_WRIST_ROI_Y_START = 0      # 顶部对齐
CAM_WRIST_ROI_OUTPUT_SIZE = 480  # 输出 480x480 (无需resize)


class AsyncFollowerStateReader(threading.Thread):
    """
    异步从臂状态读取器
    
    在独立线程中持续读取从臂关节状态，解决 rm_get_joint_degree() 
    约 50ms 延迟导致无法达到 30Hz 采集频率的问题。
    """
    
    def __init__(self, follower_arm, gripper, use_degrees: bool = False):
        super().__init__(daemon=True, name="FollowerStateReader")
        self.follower_arm = follower_arm
        self.gripper = gripper
        self.use_degrees = use_degrees
        self.running = True
        self._lock = threading.Lock()
        
        # 缓存的状态
        self._joints = [0.0] * 7  # 7 个关节
        self._gripper_pos = 0.0
        self._last_update = 0.0
        self._read_count = 0
        
    def run(self):
        """后台持续读取从臂状态"""
        while self.running:
            try:
                # 读取从臂关节 (这是耗时操作 ~50ms)
                ret, joints_deg = self.follower_arm.rm_get_joint_degree()
                
                if ret == 0:
                    # 转换单位
                    if self.use_degrees:
                        joints = list(joints_deg)
                    else:
                        joints = [np.radians(j) for j in joints_deg]
                    
                    # 读取夹爪
                    gripper_pos = 0.0
                    if self.gripper is not None:
                        try:
                            gripper_pos = self.gripper.read_position()
                        except Exception:
                            pass
                    
                    # 更新缓存
                    with self._lock:
                        self._joints = joints
                        self._gripper_pos = gripper_pos
                        self._last_update = time.time()
                        self._read_count += 1
                        
            except Exception as e:
                logger.debug(f"异步读取从臂状态错误: {e}")
            
            # 控制读取频率 (~20Hz)
            time.sleep(0.01)
    
    def get_state(self) -> tuple[list[float], float]:
        """获取缓存的从臂状态"""
        with self._lock:
            return self._joints.copy(), self._gripper_pos
    
    def get_state_age(self) -> float:
        """获取状态数据的年龄 (秒)"""
        with self._lock:
            return time.time() - self._last_update if self._last_update > 0 else float('inf')
    
    def stop(self):
        """停止读取线程"""
        self.running = False


class RealmanRM75bHd(Robot):
    """
    睿尔曼 RM75b 机械臂 (HD 高清版，无触觉传感器) - LeRobot 兼容实现
    
    支持主从控制模式，适用于遥操作数据采集和策略部署。
    
    HD 版本特性:
    - cam_top 使用 1920x1080 分辨率，ROI 裁剪为 1:1 正方形 (896x896)
    - cam_right_wrist 使用 640x480 分辨率，ROI 裁剪为 1:1 正方形 (480x480)
    - 所有相机输出正方形画幅，充分利用 Pi0 的 224x224 输入
    
    数据格式:
    - observation.state: [8] (7关节 + 1夹爪，弧度)
    - observation.images.cam_right_wrist: [480, 480, 3] (RGB, 正方形)
    - observation.images.cam_top: [896, 896, 3] (RGB, 正方形, HD)
    - action: [8] (7关节 + 1夹爪，弧度)
    """

    config_class = RealmanRM75bHdConfig
    name = "realman_rm75b_hd"
    
    # RM75b 是 7 自由度机械臂
    DOF = 7
    
    # 关节名称
    JOINT_NAMES = [
        "main_joint1",
        "main_joint2", 
        "main_joint3",
        "main_joint4",
        "main_joint5",
        "main_joint6",
        "main_joint7",
    ]
    
    # 夹爪名称
    GRIPPER_NAME = "main_gripper"

    def __init__(self, config: RealmanRM75bHdConfig):
        super().__init__(config)
        self.config = config
        
        # 延迟导入睿尔曼 SDK
        self._arm_sdk = None
        
        # 主臂 (Leader)
        self._leader_arm: LeaderArm | None = None
        
        # 从臂 (Follower)
        self._follower_arm = None
        self._follower_handle = None
        
        # 异步从臂状态读取器
        self._follower_state_reader: AsyncFollowerStateReader | None = None
        
        # 夹爪 (Gripper)
        self._gripper: Gripper | None = None
        self._gripper_connected = False
        
        # 相机
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # 连接状态
        self._leader_connected = False
        self._follower_connected = False

    def _import_sdk(self):
        """延迟导入睿尔曼 SDK"""
        if self._arm_sdk is None:
            try:
                from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
                self._arm_sdk = {
                    "RoboticArm": RoboticArm,
                    "rm_thread_mode_e": rm_thread_mode_e,
                }
            except ImportError as e:
                raise ImportError(
                    "无法导入睿尔曼 SDK。请确保 Robotic_Arm 文件夹在正确的位置。"
                    f"原始错误: {e}"
                )
        return self._arm_sdk

    # ==================== 特征定义 ====================
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征：7个关节 + 1个夹爪"""
        features = {f"{joint}": float for joint in self.JOINT_NAMES}
        features[self.GRIPPER_NAME] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征 (RGB 相机)
        
        HD 版本: 所有相机输出 1:1 正方形画幅
        - cam_top: 896x896
        - cam_right_wrist: 480x480
        """
        features = {}
        for cam in self.cameras:
            if cam == "cam_top":
                # 1920x1080 -> ROI 896x896 (无需 resize)
                features[cam] = (CAM_TOP_ROI_OUTPUT_SIZE, CAM_TOP_ROI_OUTPUT_SIZE, 3)
            elif cam == "cam_right_wrist":
                # 640x480 -> ROI 480x480 (无需 resize)
                features[cam] = (CAM_WRIST_ROI_OUTPUT_SIZE, CAM_WRIST_ROI_OUTPUT_SIZE, 3)
            else:
                h = self.config.cameras[cam].height
                w = self.config.cameras[cam].width
                features[cam] = (h, w, 3)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征：包括关节位置和相机图像"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征：关节目标位置"""
        return self._motors_ft

    # ==================== 连接状态 ====================
    
    @property
    def is_connected(self) -> bool:
        """检查所有设备是否已连接"""
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return self._follower_connected and cameras_connected

    # ==================== 连接/断开 ====================
    
    def connect(self, calibrate: bool = True) -> None:
        """连接所有设备"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已连接")

        sdk = self._import_sdk()
        RoboticArm = sdk["RoboticArm"]
        rm_thread_mode_e = sdk["rm_thread_mode_e"]

        # 1. 连接从臂
        logger.info(f"正在连接从臂 {self.config.follower_ip}:{self.config.follower_tcp_port}...")
        self._follower_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self._follower_handle = self._follower_arm.rm_create_robot_arm(
            self.config.follower_ip, 
            self.config.follower_tcp_port
        )
        
        if self._follower_handle.id < 0:
            raise ConnectionError(
                f"从臂连接失败! IP: {self.config.follower_ip}, 错误代码: {self._follower_handle.id}"
            )
        self._follower_connected = True
        logger.info(f"从臂连接成功! Handle ID: {self._follower_handle.id}")

        # 2. 连接主臂
        self._connect_leader()

        # 3. 连接夹爪
        self._connect_gripper()

        # 4. 连接相机
        for cam_name, cam in self.cameras.items():
            logger.info(f"正在连接相机 {cam_name}...")
            cam.connect()
            logger.info(f"相机 {cam_name} 连接成功")

        # 5. 启动异步从臂状态读取器
        self._start_async_state_reader()

        # 6. 校准
        if not self.is_calibrated and calibrate:
            logger.info("未找到校准数据，开始校准...")
            self.calibrate()

        # 7. 配置
        self.configure()
        
        time.sleep(0.5)
        logger.info(f"{self} 连接完成 (HD 模式，无触觉传感器)")

    def _connect_leader(self) -> None:
        """连接主臂"""
        import os
        
        if not self.config.connect_leader:
            logger.info("主臂连接已禁用 (connect_leader=False)")
            return
        
        if not os.path.exists(self.config.leader_port):
            logger.warning(f"主臂串口 {self.config.leader_port} 不存在，跳过")
            return
        
        try:
            logger.info(f"正在连接主臂 {self.config.leader_port}...")
            self._leader_arm = LeaderArm(
                port=self.config.leader_port,
                baudrate=self.config.leader_baudrate,
                hex_data=self.config.leader_hex_data,
            )
            self._leader_arm.connect()
            self._leader_connected = True
            logger.info(f"主臂连接成功!")
        except Exception as e:
            logger.warning(f"主臂连接失败: {e}")
            self._leader_arm = None
            self._leader_connected = False

    def _connect_gripper(self) -> None:
        """连接夹爪"""
        import os
        
        if not os.path.exists(self.config.gripper_port):
            logger.warning(f"夹爪串口 {self.config.gripper_port} 不存在，跳过")
            return
        
        try:
            logger.info(f"正在连接夹爪 {self.config.gripper_port}...")
            self._gripper = Gripper(
                port=self.config.gripper_port,
                motor_id=self.config.gripper_motor_id,
                motor_model=self.config.gripper_motor_model,
                baudrate=self.config.gripper_baudrate,
            )
            self._gripper.connect()
            self._gripper_connected = True
            logger.info(f"夹爪连接成功!")
        except Exception as e:
            logger.warning(f"夹爪连接失败: {e}")
            self._gripper = None
            self._gripper_connected = False

    def _start_async_state_reader(self) -> None:
        """启动异步从臂状态读取器"""
        if self._follower_arm is None or not self._follower_connected:
            logger.warning("从臂未连接，跳过异步状态读取器")
            return
        
        if self._follower_state_reader is not None:
            self._follower_state_reader.stop()
            self._follower_state_reader.join(timeout=1.0)
        
        self._follower_state_reader = AsyncFollowerStateReader(
            follower_arm=self._follower_arm,
            gripper=None,
            use_degrees=self.config.use_degrees,
        )
        self._follower_state_reader.start()
        logger.info("异步从臂状态读取器已启动")
        time.sleep(0.1)

    def get_leader_position(self) -> np.ndarray | None:
        """获取主臂当前位置"""
        if self._leader_arm is None or not self._leader_connected:
            return None
        return self._leader_arm.read_position()

    def disconnect(self) -> None:
        """断开所有设备连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        # 停止异步状态读取器
        if self._follower_state_reader is not None:
            try:
                self._follower_state_reader.stop()
                self._follower_state_reader.join(timeout=1.0)
                logger.info("异步状态读取器已停止")
            except Exception as e:
                logger.warning(f"停止异步状态读取器时出错: {e}")
            self._follower_state_reader = None

        # 断开主臂
        if self._leader_arm is not None:
            try:
                self._leader_arm.disconnect()
                logger.info("主臂已断开")
            except Exception as e:
                logger.warning(f"断开主臂时出错: {e}")
            self._leader_arm = None
            self._leader_connected = False

        # 断开从臂
        if self._follower_connected and self._follower_arm is not None:
            try:
                self._follower_arm.rm_delete_robot_arm()
            except Exception as e:
                logger.warning(f"断开从臂时出错: {e}")
            self._follower_connected = False
            logger.info("从臂已断开")

        # 断开夹爪
        if self._gripper is not None:
            try:
                self._gripper.disconnect()
                logger.info("夹爪已断开")
            except Exception as e:
                logger.warning(f"断开夹爪时出错: {e}")
            self._gripper = None
            self._gripper_connected = False

        # 断开相机
        for cam_name, cam in self.cameras.items():
            try:
                cam.disconnect()
                logger.info(f"相机 {cam_name} 已断开")
            except Exception as e:
                logger.warning(f"断开相机 {cam_name} 时出错: {e}")

        logger.info(f"{self} 已断开所有连接")

    # ==================== 校准 ====================
    
    @property
    def is_calibrated(self) -> bool:
        """睿尔曼机械臂出厂已校准"""
        return True

    def calibrate(self) -> None:
        """执行校准"""
        logger.info(f"开始校准 {self}...")
        
        self.calibration = {}
        for i, joint in enumerate(self.JOINT_NAMES):
            self.calibration[joint] = MotorCalibration(
                id=i + 1,
                drive_mode=0,
                homing_offset=0,
                range_min=-180,
                range_max=180,
            )
        
        self.calibration["gripper"] = MotorCalibration(
            id=8,
            drive_mode=0,
            homing_offset=0,
            range_min=0,
            range_max=100,
        )
        
        self._save_calibration()
        logger.info(f"校准数据已保存到 {self.calibration_fpath}")

    def configure(self) -> None:
        """配置机械臂参数"""
        logger.info(f"配置 {self}...")
        pass

    # ==================== 读取观测 ====================
    
    def _process_cam_top_image(self, image: np.ndarray) -> np.ndarray:
        """
        处理 cam_top 图像: 从底部裁取 1:1 正方形 ROI (无 resize)
        
        HD 版本 (1920x1080):
        - 直接从底部居中裁取 896x896
        - 896 = 224 × 4，缩放到 Pi0 输入时无插值损失
        - 位置: 水平居中 (x=512)，底边对齐 (y=184)
        - 无 resize，保留原始细节
        
        Args:
            image: 原始图像 (1080, 1920, 3)
        
        Returns:
            处理后的图像 (896, 896, 3)
        """
        if image is None:
            return None
        
        # ROI 裁剪 (1:1 正方形，底边对齐)
        x_start = CAM_TOP_HD_ROI_X_START
        y_start = CAM_TOP_HD_ROI_Y_START
        roi_w = CAM_TOP_HD_ROI_WIDTH
        roi_h = CAM_TOP_HD_ROI_HEIGHT
        
        # 裁剪 (无需 resize)
        roi_image = image[y_start:y_start+roi_h, x_start:x_start+roi_w]
        
        return roi_image.copy()

    def _process_cam_wrist_image(self, image: np.ndarray) -> np.ndarray:
        """
        处理 cam_right_wrist 图像: 1:1 正方形 ROI 裁剪
        
        D405 (640x480):
        - ROI: 480x480 (正方形)
        - 位置: 水平居中，顶部对齐
        - 无需 resize
        
        Args:
            image: 原始图像 (480, 640, 3)
        
        Returns:
            处理后的图像 (480, 480, 3)
        """
        if image is None:
            return None
        
        # ROI 裁剪 (1:1 正方形)
        x_start = CAM_WRIST_ROI_X_START
        y_start = CAM_WRIST_ROI_Y_START
        roi_w = CAM_WRIST_ROI_WIDTH
        roi_h = CAM_WRIST_ROI_HEIGHT
        
        # 裁剪 (无需 resize，已经是 480x480)
        roi_image = image[y_start:y_start+roi_h, x_start:x_start+roi_w]
        
        return roi_image.copy()

    def get_observation(self) -> dict[str, Any]:
        """
        读取当前观测：关节位置 + 相机图像
        
        Returns:
            dict: 包含所有传感器数据的字典
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        obs_dict = {}

        # 1. 读取从臂关节位置
        start = time.perf_counter()
        
        if self._follower_state_reader is not None:
            joints, _ = self._follower_state_reader.get_state()
            for i, joint in enumerate(self.JOINT_NAMES):
                obs_dict[joint] = joints[i]
            
            state_age = self._follower_state_reader.get_state_age()
            if state_age > 0.1:
                logger.warning(f"从臂状态数据过时: {state_age*1000:.0f}ms")
        else:
            result = self._follower_arm.rm_get_joint_degree()
            if result[0] == 0:
                joints_deg = result[1]
                for i, joint in enumerate(self.JOINT_NAMES):
                    if self.config.use_degrees:
                        obs_dict[joint] = joints_deg[i]
                    else:
                        obs_dict[joint] = np.radians(joints_deg[i])
            else:
                logger.warning(f"读取从臂状态失败，错误码: {result[0]}")
                for joint in self.JOINT_NAMES:
                    obs_dict[joint] = 0.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"读取关节状态: {dt_ms:.1f}ms")

        # 2. 读取夹爪位置
        start = time.perf_counter()
        if self._gripper is not None and self._gripper_connected:
            obs_dict[self.GRIPPER_NAME] = self._gripper.read_position()
        else:
            obs_dict[self.GRIPPER_NAME] = 0.0
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"读取夹爪状态: {dt_ms:.1f}ms")

        # 3. 读取相机图像 (1:1 正方形 ROI)
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            image = cam.async_read(timeout_ms=500)
            
            # 对相机做 1:1 正方形 ROI 裁剪
            if cam_key == "cam_top" and image is not None:
                # 1920x1080 -> 896x896
                image = self._process_cam_top_image(image)
            elif cam_key == "cam_right_wrist" and image is not None:
                # 640x480 -> 480x480
                image = self._process_cam_wrist_image(image)
            
            obs_dict[cam_key] = image
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"读取相机 {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    # ==================== 发送动作 ====================
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作到从臂"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        # 1. 提取关节目标位置
        goal_pos = {}
        for key, val in action.items():
            if key in self.JOINT_NAMES:
                goal_pos[key] = val

        # 2. 安全限制
        if self.config.max_relative_target is not None:
            result = self._follower_arm.rm_get_joint_degree()
            if result[0] == 0:
                present_joints = result[1]
                present_pos = {
                    joint: (np.radians(present_joints[i]) if not self.config.use_degrees else present_joints[i])
                    for i, joint in enumerate(self.JOINT_NAMES)
                }
                goal_present_pos = {key: (goal_pos[key], present_pos[key]) for key in goal_pos}
                goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # 3. 转换为角度列表
        target_degrees = []
        for joint in self.JOINT_NAMES:
            if joint in goal_pos:
                val = goal_pos[joint]
                if not self.config.use_degrees:
                    val = np.degrees(val)
                target_degrees.append(float(val))
            else:
                target_degrees.append(0.0)

        # 4. 发送到从臂
        start = time.perf_counter()
        result = self._follower_arm.rm_movej_canfd(target_degrees, False, 0)
        
        if result != 0:
            logger.warning(f"发送动作失败，错误码: {result}")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"发送动作: {dt_ms:.1f}ms")

        # 5. 发送夹爪动作
        gripper_pos = action.get(self.GRIPPER_NAME, None)
        if gripper_pos is not None and self._gripper is not None and self._gripper_connected:
            obj = getattr(self.config, "object", "") or ""
            obj_limits = getattr(self.config, "gripper_object_min_open", {}) or {}
            if obj and obj in obj_limits:
                limit = float(obj_limits[obj])
                if -1.309 <= limit <= 1.309:
                    if float(gripper_pos) < limit:
                        logger.debug(
                            f"夹爪目标 {float(gripper_pos):.4f} 低于物体 '{obj}' "
                            f"的最小开合下限 {limit:.4f}，已裁剪"
                        )
                        gripper_pos = limit
                else:
                    logger.warning(
                        f"忽略 gripper_object_min_open['{obj}']={limit:.4f}：超出"
                        f"物理可写范围 [-1.309, 1.309]"
                    )
            start = time.perf_counter()
            self._gripper.write_position(gripper_pos)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"发送夹爪动作: {dt_ms:.1f}ms")

        # 返回实际发送的动作
        sent_action = {joint: goal_pos.get(joint, 0.0) for joint in self.JOINT_NAMES}
        sent_action[self.GRIPPER_NAME] = gripper_pos if gripper_pos is not None else 0.0
        return sent_action
