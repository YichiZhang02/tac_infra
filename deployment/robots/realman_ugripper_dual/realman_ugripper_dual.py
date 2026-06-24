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
睿尔曼 RM75b 双臂 (ugripper 集成版) LeRobot 适配器

每条手臂: 从臂(TCP) + 领控电爪(gRPC/CAN) + 手腕鱼眼相机(gRPC/UDP) + 2 路触觉(gRPC)。
主臂由 bi_realman_ugripper_leader 遥操作器负责, 本机器人只管从臂侧设备。

数据格式 (每条启用的手臂, side ∈ {left, right}):
    observation.state:
        {side}_main_joint1..7  (float, 弧度)
        {side}_main_gripper    (float, 归一化 [0,1], 1=张开)
    observation.images:
        {side}_cam_wrist       (1080, 1920, 3) uint8  RGB 鱼眼
        {side}_cam_finger0     (288, 384, 3)   uint8 [depth, deform_x, deform_y] (各通道归一化到[0,255])
        {side}_cam_finger1     (288, 384, 3)   uint8
    action:
        {side}_main_joint1..7  (float, 弧度)
        {side}_main_gripper    (float, 归一化 [0,1])

并发: 各路视觉/触觉数据流各跑独立进程 (见 deployment/hardware 下各硬件类), 从臂关节状态
各跑一个后台线程 (RealmanTcpFollower 内部状态读取), get_observation 全程非阻塞。
"""

import logging
import sys
import threading
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from deployment.hardware.top_cameras import make_top_cameras_from_configs
from deployment.hardware.calibration import MotorCalibration
from vtla.engine.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_realman_ugripper_dual import RealmanUGripperDualConfig
from deployment.hardware.grippers import LingkongGripper
from deployment.hardware.wrist_cameras import FisheyeGrpcCamera, WristUndistorter, default_calib_path
from deployment.hardware.tactile_sensors import DmroboticsFlux
from deployment.hardware.follower_arms import RealmanTcpFollower

# 添加 Robotic_Arm SDK 路径 (deployment/sdk，使仓库自包含)
sys.path.append(str(Path(__file__).resolve().parents[2] / "sdk"))

logger = logging.getLogger(__name__)


class _ArmDevices:
    """单条手臂的从臂侧设备集合。"""

    def __init__(self, side: str):
        self.side = side
        self.follower: RealmanTcpFollower | None = None
        self.gripper: LingkongGripper | None = None
        # 数据流接收器
        self.fisheye: FisheyeGrpcCamera | None = None
        self.tactile0: DmroboticsFlux | None = None
        self.tactile1: DmroboticsFlux | None = None
        # 腕部去畸变器 (None = 不去畸变, 输出原生鱼眼)
        self.wrist_undistorter: WristUndistorter | None = None

    @property
    def receivers(self) -> list:
        """所有数据流硬件 (鱼眼 + 两路触觉), 均带 connect/async_read/disconnect/is_connected。"""
        return [r for r in (self.fisheye, self.tactile0, self.tactile1) if r is not None]


class RealmanUGripperDual(Robot):
    """睿尔曼 RM75b 双臂 (ugripper 集成版)。"""

    config_class = RealmanUGripperDualConfig
    name = "realman_ugripper_dual"

    DOF = 7
    JOINT_NAMES = [f"main_joint{i}" for i in range(1, 8)]
    GRIPPER_NAME = "main_gripper"

    def __init__(self, config: RealmanUGripperDualConfig):
        super().__init__(config)
        self.config = config

        for side in config.arms:
            if side not in ("left", "right"):
                raise ValueError(f"无效的手臂名 '{side}', 只支持 'left' / 'right'")

        self._arms: dict[str, _ArmDevices] = {side: _ArmDevices(side) for side in config.arms}

        # 腕部去畸变开关: "true" 开, "false"/"auto" 关 (auto 仅在 inference.py 改写为 true/false 时生效;
        # 采集/遥操作无 policy, auto 即关闭, 存原生鱼眼供离线 tools 去畸变)。
        self._undistort_on = str(config.undistort_wrist).lower() == "true"

        # 额外本地相机 (默认无)
        self.cameras = make_top_cameras_from_configs(config.cameras)

        # 各臂 home 目标关节位置 (单位同 read_joints / send_joints, 即 use_degrees 决定)
        # connect() 后由 _capture_home_joints() 填充
        self._home_joints: dict[str, list[float]] = {}

    # ==================== 配置辅助 ====================

    def _board_ip(self, side: str) -> str:
        return self.config.left_board_ip if side == "left" else self.config.right_board_ip

    def _follower_ip(self, side: str) -> str:
        return self.config.left_follower_ip if side == "left" else self.config.right_follower_ip

    def _fisheye_udp_port(self, side: str) -> int:
        return self.config.left_fisheye_udp_port if side == "left" else self.config.right_fisheye_udp_port

    def _wrist_calib_path(self, side: str) -> str:
        """该臂腕部去畸变标定文件: config.wrist_calib[side] 优先, 否则用 deployment 内置。"""
        wc = self.config.wrist_calib or {}
        return str(wc.get(side) or default_calib_path(side))

    def _tactile_pc_ports(self, side: str) -> tuple[int, int]:
        if side == "left":
            return self.config.left_tactile0_pc_port, self.config.left_tactile1_pc_port
        return self.config.right_tactile0_pc_port, self.config.right_tactile1_pc_port

    # ==================== 特征定义 ====================

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft: dict[str, type] = {}
        for side in self.config.arms:
            for joint in self.JOINT_NAMES:
                ft[f"{side}_{joint}"] = float
            ft[f"{side}_{self.GRIPPER_NAME}"] = float
        return ft

    @property
    def _stream_ft(self) -> dict[str, tuple]:
        ft: dict[str, tuple] = {}
        for side in self.config.arms:
            if self._undistort_on:
                crop = self.config.undistort_crop
                ft[f"{side}_cam_wrist"] = (crop, crop, 3)
            else:
                ft[f"{side}_cam_wrist"] = (self.config.fisheye_height, self.config.fisheye_width, 3)
            if self.config.use_tactile:
                ft[f"{side}_cam_finger0"] = (self.config.tactile_height, self.config.tactile_width, 3)
                ft[f"{side}_cam_finger1"] = (self.config.tactile_height, self.config.tactile_width, 3)
        return ft

    @staticmethod
    def _crop_4_3_size(h: int, w: int) -> tuple[int, int]:
        """给定原始 (H, W), 返回居中裁剪到 4:3 (宽:高) 后的 (H, W)。"""
        if w * 3 > h * 4:        # 太宽, 裁宽度
            return h, (h * 4) // 3
        else:                    # 太高(或正好), 裁高度
            return (w * 3) // 4, w

    @classmethod
    def _center_crop_4_3(cls, image: np.ndarray) -> np.ndarray:
        """中心裁剪到 4:3 (宽:高 = 4:3), 居中。"""
        h, w = image.shape[:2]
        ch, cw = cls._crop_4_3_size(h, w)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        return image[y0:y0 + ch, x0:x0 + cw]

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        ft: dict[str, tuple] = {}
        for cam in self.cameras:
            h = self.config.cameras[cam].height
            w = self.config.cameras[cam].width
            if cam in self.config.crop_4_3_cameras:
                ch, cw = self._crop_4_3_size(h, w)
                ft[cam] = (ch, cw, 3)
            else:
                ft[cam] = (h, w, 3)
        return ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._stream_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ==================== 连接状态 ====================

    @property
    def is_connected(self) -> bool:
        arms_ok = all(arm.follower is not None and arm.follower.is_connected for arm in self._arms.values())
        streams_ok = all(r.is_connected for arm in self._arms.values() for r in arm.receivers)
        cameras_ok = all(cam.is_connected for cam in self.cameras.values())
        return arms_ok and streams_ok and cameras_ok

    # ==================== 连接 ====================

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已连接")

        # 两条臂硬件完全独立 (不同板子/夹爪/从臂), 并行连接以缩短启动时间。
        # 各自的耗时大头: 3 路数据流首帧等待 + 夹爪 grip_init 自标定 (会开合两次)。
        errors: dict[str, Exception] = {}

        def _do(side: str):
            try:
                self._connect_one_arm(side)
            except Exception as e:  # noqa: BLE001
                errors[side] = e

        threads = [
            threading.Thread(target=_do, args=(side,), name=f"connect-{side}")
            for side in self.config.arms
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            # 任一臂失败则回滚已连接的设备, 再抛出
            self._safe_teardown()
            side, err = next(iter(errors.items()))
            raise ConnectionError(f"[{side}] 臂连接失败: {err}") from err

        # 额外本地相机
        for cam_name, cam in self.cameras.items():
            logger.info(f"正在连接本地相机 {cam_name}...")
            cam.connect()

        # 6. 校准 / 配置
        if not self.is_calibrated and calibrate:
            self.calibrate()
        self.configure()

        time.sleep(0.5)
        self._capture_home_joints()
        logger.info(f"{self} 连接完成 (ugripper 双臂, 启用: {self.config.arms})")

    def _connect_one_arm(self, side: str) -> None:
        """连接单条臂的全部从臂侧设备 (供并行线程调用)。"""
        arm = self._arms[side]
        logger.info(f"==== 正在连接 {side} 臂 ====")

        # 1. 数据流 (先连, 确保触觉传感器在夹爪夹紧前完成零点校准)
        self._connect_streams(side, arm)

        # 2. 领控电爪 (会夹紧)
        self._connect_gripper(side, arm)

        # 3. 从臂 TCP (RealmanTcpFollower 内部含句柄创建锁 + 异步状态读取线程)
        follower_ip = self._follower_ip(side)
        logger.info(f"[{side}] 正在连接从臂 {follower_ip}:{self.config.follower_tcp_port}...")
        arm.follower = RealmanTcpFollower(
            ip=follower_ip,
            port=self.config.follower_tcp_port,
            dof=self.DOF,
            use_degrees=self.config.use_degrees,
            name=f"{side}_follower",
        )
        arm.follower.connect()

    def _safe_teardown(self) -> None:
        """尽力断开已连接设备 (连接失败回滚用, 不抛异常)。"""
        for arm in self._arms.values():
            if arm.follower is not None:
                try:
                    arm.follower.disconnect()
                except Exception:
                    pass
                arm.follower = None
            if arm.gripper is not None:
                try:
                    arm.gripper.disconnect()
                except Exception:
                    pass
                arm.gripper = None
            for r in arm.receivers:
                try:
                    r.disconnect()
                except Exception:
                    pass
            arm.fisheye = arm.tactile0 = arm.tactile1 = None

    def _connect_streams(self, side: str, arm: _ArmDevices) -> None:
        cfg = self.config
        board_ip = self._board_ip(side)
        udp_port = self._fisheye_udp_port(side)
        pc_port0, pc_port1 = self._tactile_pc_ports(side)

        arm.fisheye = FisheyeGrpcCamera(
            ip=board_ip,
            grpc_port=cfg.fisheye_grpc_port,
            udp_port=udp_port,
            width=cfg.fisheye_width,
            height=cfg.fisheye_height,
            max_datagram=cfg.fisheye_max_datagram,
            max_fps=cfg.stream_max_fps,
            debug=cfg.stream_debug_fps,
            first_frame_timeout=cfg.stream_first_frame_timeout,
            name=f"{side}_cam_wrist",
        )
        # 腕部去畸变器 (开启时按训练流程: 鱼眼去畸变 + 居中裁 crop)。maps 只算一次。
        if self._undistort_on:
            arm.wrist_undistorter = WristUndistorter(
                self._wrist_calib_path(side), crop=cfg.undistort_crop
            )
        if cfg.use_tactile:
            arm.tactile0 = DmroboticsFlux(
                dev_id=cfg.tactile0_dev_id,
                remote_addr=f"{board_ip}:{cfg.tactile0_grpc_port}",
                pc_host=cfg.pc_host,
                pc_port=pc_port0,
                width=cfg.tactile_width,
                height=cfg.tactile_height,
                max_fps=cfg.stream_max_fps,
                debug=cfg.stream_debug_fps,
                depth_min=cfg.tactile_depth_min,
                depth_max=cfg.tactile_depth_max,
                deform_min=cfg.tactile_deform_min,
                deform_max=cfg.tactile_deform_max,
                first_frame_timeout=cfg.stream_first_frame_timeout,
                name=f"{side}_cam_finger0",
            )
            arm.tactile1 = DmroboticsFlux(
                dev_id=cfg.tactile1_dev_id,
                remote_addr=f"{board_ip}:{cfg.tactile1_grpc_port}",
                pc_host=cfg.pc_host,
                pc_port=pc_port1,
                width=cfg.tactile_width,
                height=cfg.tactile_height,
                max_fps=cfg.stream_max_fps,
                debug=cfg.stream_debug_fps,
                depth_min=cfg.tactile_depth_min,
                depth_max=cfg.tactile_depth_max,
                deform_min=cfg.tactile_deform_min,
                deform_max=cfg.tactile_deform_max,
                first_frame_timeout=cfg.stream_first_frame_timeout,
                name=f"{side}_cam_finger1",
            )
        # 3 路数据流相互独立, 并行启动 (各自要等首帧, 串行会叠加)
        stream_errors: dict[str, Exception] = {}

        def _connect_stream(r):
            try:
                logger.info(f"[{side}] 正在启动数据流 {r.name}...")
                r.connect()
            except Exception as e:  # noqa: BLE001
                stream_errors[r.name] = e

        ts = [threading.Thread(target=_connect_stream, args=(r,), name=f"stream-{r.name}")
              for r in arm.receivers]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if stream_errors:
            name, err = next(iter(stream_errors.items()))
            raise ConnectionError(f"数据流 {name} 启动失败: {err}") from err

    def _connect_gripper(self, side: str, arm: _ArmDevices) -> None:
        board_ip = self._board_ip(side)
        server = f"{board_ip}:{self.config.gripper_grpc_port}"
        try:
            gripper = LingkongGripper(
                server_address=server,
                can_interface=self.config.gripper_can_interface,
                can_bitrate=self.config.gripper_can_bitrate,
                speed=self.config.gripper_speed,
                torque=self.config.gripper_torque,
            )
            if not gripper.connect():
                logger.warning(f"[{side}] 夹爪 gRPC 连接失败, 该臂夹爪将不可用")
                return
            itinerary = (
                self.config.left_gripper_itinerary if side == "left"
                else self.config.right_gripper_itinerary
            )
            if not gripper.init_gripper(itinerary_override=itinerary):
                logger.warning(f"[{side}] 夹爪初始化失败, 该臂夹爪将不可用")
                return
            arm.gripper = gripper
        except Exception as e:
            logger.warning(f"[{side}] 夹爪连接异常: {e}")
            arm.gripper = None

    # ==================== 校准 ====================

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info(f"开始校准 {self}...")
        self.calibration = {}
        idx = 1
        for side in self.config.arms:
            for joint in self.JOINT_NAMES:
                self.calibration[f"{side}_{joint}"] = MotorCalibration(
                    id=idx, drive_mode=0, homing_offset=0, range_min=-180, range_max=180
                )
                idx += 1
            self.calibration[f"{side}_{self.GRIPPER_NAME}"] = MotorCalibration(
                id=idx, drive_mode=0, homing_offset=0, range_min=0, range_max=1000
            )
            idx += 1
        self._save_calibration()
        logger.info(f"校准数据已保存到 {self.calibration_fpath}")

    def configure(self) -> None:
        logger.info(f"配置 {self}...")

    # ==================== 初始位置复位 ====================

    def _capture_home_joints(self) -> None:
        """连接完成后读取各臂当前关节位置作为 home 目标 (优先 config.home_joints)。"""
        for side, arm in self._arms.items():
            if arm.follower is None:
                continue
            if self.config.home_joints is not None:
                target = [
                    self.config.home_joints.get(f"{side}_{j}", 0.0)
                    for j in self.JOINT_NAMES
                ]
                logger.info(f"[{side}] home 位置来自 config: {target}")
            else:
                pos = arm.follower.read_joints_now()
                if pos is None:
                    pos = arm.follower.read_joints()
                target = list(pos)
                logger.info(f"[{side}] home 位置自动捕获 (当前姿态): {target}")
            self._home_joints[side] = target

    def move_to_home(self) -> None:
        """机械臂和夹爪复位到初始位置 (推理时按→保存后调用)。

        流程: 先立即张开夹爪 (非阻塞), 再并行将双臂平滑插值到 home 位置。
        """
        if not self.is_connected:
            logger.warning("move_to_home: 机器人未连接, 跳过")
            return

        # 1. 立即张开夹爪 (与臂运动并行)
        for side, arm in self._arms.items():
            if arm.gripper is not None:
                try:
                    arm.gripper.move_norm(self.config.home_gripper)
                    logger.info(f"[{side}] 夹爪复位: {self.config.home_gripper:.2f}")
                except Exception as e:
                    logger.warning(f"[{side}] 夹爪复位出错: {e}")

        # 2. 双臂并行平滑运动到 home 位置
        def _home_arm(side: str, arm: _ArmDevices) -> None:
            target = self._home_joints.get(side)
            if target is None:
                logger.warning(f"[{side}] home 位置未记录, 跳过臂复位")
                return
            logger.info(f"[{side}] 机械臂复位目标: {[f'{v:.3f}' for v in target]}")
            try:
                arm.follower.move_to(target, duration_s=self.config.home_duration_s)
                logger.info(f"[{side}] 机械臂复位完成")
            except Exception as e:
                logger.warning(f"[{side}] 机械臂复位出错: {e}")

        threads = [
            threading.Thread(target=_home_arm, args=(side, arm), name=f"home-{side}")
            for side, arm in self._arms.items()
            if arm.follower is not None
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # ==================== 读取观测 ====================

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        obs: dict[str, Any] = {}

        # 1. 关节 + 夹爪 (state)
        for side in self.config.arms:
            arm = self._arms[side]
            joints = arm.follower.read_joints() if arm.follower is not None else [0.0] * self.DOF
            for i, joint in enumerate(self.JOINT_NAMES):
                obs[f"{side}_{joint}"] = joints[i]
            if arm.follower is not None and arm.follower.get_state_age() > 0.1:
                logger.warning(f"[{side}] 从臂状态过时: {arm.follower.get_state_age()*1000:.0f}ms")

            obs[f"{side}_{self.GRIPPER_NAME}"] = (
                arm.gripper.read_norm() if arm.gripper is not None else 0.0
            )

        # 2. 数据流图像
        for side in self.config.arms:
            arm = self._arms[side]
            wrist = arm.fisheye.async_read()
            if arm.wrist_undistorter is not None:
                wrist = arm.wrist_undistorter(wrist)  # 鱼眼去畸变 + 居中裁 (与训练一致)
            obs[f"{side}_cam_wrist"] = wrist
            if self.config.use_tactile:
                obs[f"{side}_cam_finger0"] = arm.tactile0.async_read()
                obs[f"{side}_cam_finger1"] = arm.tactile1.async_read()

        # 3. 额外本地相机
        for cam_key, cam in self.cameras.items():
            img = cam.async_read()
            if cam_key in self.config.crop_4_3_cameras:
                img = self._center_crop_4_3(img)
            obs[cam_key] = img

        return obs

    # ==================== 发送动作 ====================

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        sent_action: dict[str, Any] = {}

        for side in self.config.arms:
            arm = self._arms[side]

            # 1. 关节目标
            goal_pos = {
                joint: action[f"{side}_{joint}"]
                for joint in self.JOINT_NAMES
                if f"{side}_{joint}" in action
            }

            # 2. 安全限幅 (相对当前)
            if self.config.max_relative_target is not None and goal_pos and arm.follower is not None:
                present_arr = arm.follower.read_joints_now()  # 配置单位, 失败返回 None
                if present_arr is not None:
                    present = {joint: present_arr[i] for i, joint in enumerate(self.JOINT_NAMES)}
                    goal_present = {k: (goal_pos[k], present[k]) for k in goal_pos}
                    goal_pos = ensure_safe_goal_position(goal_present, self.config.max_relative_target)

            # 3. 下发 (follower 内部按配置单位换算成角度)
            target = [goal_pos.get(joint, 0.0) for joint in self.JOINT_NAMES]
            if arm.follower is not None:
                arm.follower.send_joints(target)

            for joint in self.JOINT_NAMES:
                sent_action[f"{side}_{joint}"] = goal_pos.get(joint, 0.0)

            # 4. 夹爪 (归一化 [0,1])
            gripper_val = action.get(f"{side}_{self.GRIPPER_NAME}", None)
            if gripper_val is not None and arm.gripper is not None:
                arm.gripper.move_norm(float(gripper_val))
            sent_action[f"{side}_{self.GRIPPER_NAME}"] = (
                float(gripper_val) if gripper_val is not None else 0.0
            )

        return sent_action

    # ==================== 断开 ====================

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        for side, arm in self._arms.items():
            # 从臂 (内部含状态线程停止 + 句柄删除)
            if arm.follower is not None:
                try:
                    arm.follower.disconnect()
                except Exception as e:
                    logger.warning(f"[{side}] 断开从臂出错: {e}")
                arm.follower = None

            # 夹爪
            if arm.gripper is not None:
                try:
                    arm.gripper.disconnect()
                except Exception as e:
                    logger.warning(f"[{side}] 断开夹爪出错: {e}")
                arm.gripper = None

            # 数据流
            for r in arm.receivers:
                try:
                    r.disconnect()
                except Exception as e:
                    logger.warning(f"[{side}] 断开 {r.name} 出错: {e}")
            arm.fisheye = arm.tactile0 = arm.tactile1 = None

        for cam_name, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"断开相机 {cam_name} 出错: {e}")

        logger.info(f"{self} 已断开所有连接")
