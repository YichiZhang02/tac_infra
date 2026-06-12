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
多进程视觉/触觉数据流接收器 (ugripper 双臂版)

架构参考 ugripper/unified_client.py:
    dmrobotics 多传感器/多 gRPC 流在单进程的多线程里会互相饿死 GIL, 因此每路数据源
    (鱼眼 / 触觉) 各跑一个独立进程, 通过 mp.Queue(maxsize=1) 把最新帧回传主进程。
    主进程 async_read() 只做"取队列最新", 不阻塞采集主循环。

两类数据源:
    - 鱼眼(手腕)相机: fish_camera gRPC(50088) -> OpenStream(MJPG) -> FCP1 分片 UDP
      -> 重组 + imdecode, 输出原生分辨率 (默认 1920x1080) RGB uint8。
    - 触觉传感器: dmrobotics Flux gRPC(50051/50052) -> getDepth + getDeformation2D,
      归一化拼成 (H, W, 3) uint16 RGB: [depth, deform_x, deform_y]
      (与参考实现 ugripper/unified_client.py 一致)。

所有进程用 spawn 上下文创建, 避免 fork 后 grpc / dmrobotics 句柄复制带来的问题。
"""

import logging
import multiprocessing as mp
import socket
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# tac_infra 自包含: 硬件客户端 / SDK 统一放在 deployment/sdk 下 (与 lerobot_tactile_ws 解耦)。
# 本文件路径: .../tac_infra/deployment/robots/realman_ugripper_dual/stream_receivers.py
#   parents[2] = .../tac_infra/deployment
# 部署到真机时需将鱼眼相机客户端放在 deployment/sdk/fish_camera_client/
# (含 camera_proxy_pb2[_grpc].py, udp_frame.py); dmrobotics 已随仓库自带在 deployment/sdk/。
_SDK_ROOT = Path(__file__).resolve().parents[2] / "sdk"
_FISH_CLIENT_DIR = _SDK_ROOT / "fish_camera_client"
_SDK_DIR = _SDK_ROOT


def decode_fisheye_jpeg(buf) -> Optional[NDArray]:
    """消费端(主进程)解码鱼眼 JPEG 字节 -> RGB uint8 数组。解码失败返回 None。"""
    import cv2
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _put_latest(q, item) -> None:
    """maxsize=1 队列: 丢旧留新, 始终只保留最新一帧。"""
    try:
        q.get_nowait()
    except Empty:
        pass
    try:
        q.put_nowait(item)
    except Exception:
        pass


# ==================== 鱼眼(手腕)相机进程 ====================
def _fisheye_worker(ip, grpc_port, udp_port, want_w, want_h, max_datagram, max_fps, debug, q, stop):
    """鱼眼相机工作进程: gRPC OpenStream(MJPG) + UDP 重组, 输出 RGB uint8。

    debug=True 时每 ~2s 打印: 产出fps / 平均imdecode耗时 / UDP数据报到达率,
    用于区分瓶颈在板子+网线(到达率低) 还是 PC解码(decode_ms高)。
    """
    if str(_FISH_CLIENT_DIR) not in sys.path:
        sys.path.insert(0, str(_FISH_CLIENT_DIR))

    import cv2
    import grpc
    import camera_proxy_pb2 as pb
    import camera_proxy_pb2_grpc as pbg
    from udp_frame import FrameReassembler

    while not stop.is_set():
        sock = None
        channel = None
        call = None
        try:
            channel = grpc.insecure_channel(f"{ip}:{grpc_port}")
            grpc.channel_ready_future(channel).result(timeout=8)
            stub = pbg.CameraProxyStub(channel)

            caps = stub.ListCapabilities(pb.CapabilityRequest(device=""), timeout=5)
            device = w = h = fps = None
            for cam in caps.cameras:
                for codec in cam.codecs:
                    if codec.codec == "MJPG" and len(codec.modes):
                        modes = list(codec.modes)
                        m = next(
                            (x for x in modes if x.width == want_w and x.height == want_h),
                            max(modes, key=lambda x: x.width * x.height),
                        )
                        device, w, h = cam.device, m.width, m.height
                        fps = max(m.fps) if m.fps else 30
                        break
                if device:
                    break
            if device is None:
                raise RuntimeError("服务端无 MJPG 模式")
            # 打印实际选定的分辨率: 若请求的 want_w x want_h 板子不支持, 会回退到最大模式,
            # 这一行能确认是否真的拿到了你想要的分辨率 (如 640x480)。
            print(f"[fisheye {ip}] 选定 MJPG {device} {w}x{h}@{fps} "
                  f"(请求 {want_w}x{want_h})", flush=True)

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
            sock.bind(("0.0.0.0", udp_port))
            sock.settimeout(1.0)

            req = pb.StreamRequest(
                codec="MJPG", width=w, height=h, fps=int(fps),
                udp_port=udp_port, client_ip="", device=device, max_datagram=max_datagram,
            )
            call = stub.OpenStream(req)

            import threading

            def _drain():
                try:
                    for _ in call:
                        pass
                except Exception:
                    pass  # 关闭时 call.cancel() 会抛 CANCELLED, 忽略

            threading.Thread(target=_drain, daemon=True).start()

            reasm = FrameReassembler(timeout_sec=0.5)
            min_interval = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0
            next_emit = 0.0
            n_dg = n_fr = 0
            dec_ms = 0.0
            t_stat = time.time()
            while not stop.is_set():
                try:
                    datagram, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                n_dg += 1
                done = reasm.push(datagram)
                if done is None:
                    continue
                # 限速(可选): 到点才出帧, 其余整帧丢弃。
                now = time.perf_counter()
                if min_interval and now < next_emit:
                    continue
                next_emit = now + min_interval
                # ⚠️ 关键: 入队的是原始 JPEG 字节(~150KB), 不是解码后的 6MB 数组。
                # 6MB 过 mp.Queue 要在 feeder 线程 pickle ~8ms, 高帧率下"槽位搬运中"窗口
                # 频繁, 消费端 get_nowait 频繁扑空 -> 取到旧缓存 -> 重复帧/卡顿(实测满速
                # 唯一帧仅 18fps)。改传 JPEG 字节, 传输量缩 ~40 倍, 解码移到消费端(主进程,
                # 有空闲), 消费端每帧都能拿到最新帧。
                _put_latest(q, bytes(done.payload))
                n_fr += 1

                if debug and time.time() - t_stat >= 2.0:
                    dt = time.time() - t_stat
                    print(f"[diag fisheye {ip}] 产出fps={n_fr/dt:.1f} "
                          f"(送JPEG字节, 解码在消费端) UDP数据报/s={n_dg/dt:.0f}",
                          flush=True)
                    n_dg = n_fr = 0
                    t_stat = time.time()
        except Exception as e:
            if not stop.is_set():
                print(f"[fisheye {ip}:{grpc_port}] 出错, 2s 后重连: {e}", flush=True)
                time.sleep(2)
        finally:
            try:
                if call is not None:
                    call.cancel()
            except Exception:
                pass
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
            try:
                if channel is not None:
                    channel.close()
            except Exception:
                pass


# ==================== 触觉传感器进程 ====================
def _tactile_worker(
    dev_id, remote_addr, pc_host, pc_port,
    max_fps, debug,
    depth_min, depth_max, deform_min, deform_max,
    q, stop,
):
    """触觉工作进程: dmrobotics Flux getDepth+getDeformation2D -> (H,W,3) uint8 RGB。

    通道: B=depth, G=deform_x, R=deform_y。
    SDK 输出的 depth/deformation 均为 float32, 各通道按 [min,max] 线性归一化到 [0,255] uint8:
        depth  -> clip((depth  - depth_min)  / (depth_max  - depth_min),  0, 1) * 255
        deform -> clip((deform - deform_min) / (deform_max - deform_min), 0, 1) * 255
    (与 tac_infra uint8 触觉约定一致; 用 deformation 而非 shear。)
    debug=True 时每 ~2s 打印产出fps / 取数+处理耗时。
    """
    if str(_SDK_DIR) not in sys.path:
        sys.path.insert(0, str(_SDK_DIR))

    from dmrobotics import Mode, Sensor, SensorOptions

    while not stop.is_set():
        sensor = None
        try:
            opt = SensorOptions(
                dev_id=dev_id, backend="Flux", mode=Mode.STANDARD, show_fps=False,
                enable_raw=False, enable_deformation=True, enable_depth=True,
                enable_shear=False, enable_force=False,
                remote_addr=remote_addr, pc_host=pc_host, pc_port=pc_port,
            )
            sensor = Sensor(opt)
            last = -1
            min_interval = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0
            next_emit = 0.0
            n_fr = 0
            proc_ms = 0.0
            t_stat = time.time()
            while not stop.is_set():
                if sensor.getDevStatus() != 0:
                    time.sleep(0.01)
                    continue
                if not sensor.wait_for_new(last, timeout_ms=500):
                    continue
                # 限速到 max_fps: 传感器原生 ~110fps, 无需全要; sleep 让出 CPU 给主循环。
                now = time.perf_counter()
                if min_interval and now < next_emit:
                    time.sleep(next_emit - now)
                next_emit = time.perf_counter() + min_interval
                _s = time.perf_counter()
                fid, depth = sensor.getDepth()
                _, deform = sensor.getDeformation2D()
                last = fid if fid is not None else last
                if depth is None or deform is None:
                    continue

                depth = np.asarray(depth, dtype=np.float32)
                deform = np.asarray(deform, dtype=np.float32)
                nd = np.clip((depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)     # (H, W)
                ng = np.clip((deform - deform_min) / (deform_max - deform_min), 0.0, 1.0)  # (H, W, 2)
                rgb = np.concatenate([nd[..., np.newaxis], ng], axis=-1)       # (H, W, 3)
                rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

                proc_ms += (time.perf_counter() - _s) * 1000
                n_fr += 1
                _put_latest(q, rgb8)

                if debug and time.time() - t_stat >= 2.0:
                    dt = time.time() - t_stat
                    print(f"[diag tactile {remote_addr}] 产出fps={n_fr/dt:.1f} "
                          f"取数+处理={proc_ms/max(n_fr,1):.1f}ms", flush=True)
                    n_fr = 0
                    proc_ms = 0.0
                    t_stat = time.time()
        except Exception as e:
            if not stop.is_set():
                print(f"[tactile {remote_addr}] 出错, 2s 后重连: {e}", flush=True)
                time.sleep(2)
        finally:
            try:
                if sensor is not None:
                    sensor.disconnect()
            except Exception:
                pass


# ==================== 进程管理封装 ====================
class StreamReceiver:
    """单路数据流的多进程封装, 暴露与本地相机一致的 async_read() 接口。"""

    def __init__(
        self,
        name: str,
        target: Callable,
        args: tuple,
        empty_frame: NDArray,
        first_frame_timeout: float = 15.0,
        decode_fn: Optional[Callable] = None,
    ):
        self.name = name
        self._target = target
        self._args = args
        self._empty_frame = empty_frame
        self._first_frame_timeout = first_frame_timeout
        # 可选: 队列里是"原始字节"时, 在消费端(主进程)解码成数组。鱼眼传 JPEG 字节用此
        # 解码, 避免 6MB 数组过队列导致消费端饥饿。tactile 传数组, decode_fn=None。
        self._decode_fn = decode_fn

        self._ctx = mp.get_context("spawn")
        self._q = None
        self._stop = None
        self._proc = None
        self._last = None            # 队列里的原始项(数组 或 JPEG字节)
        self._last_decoded = None    # decode_fn 解出的数组(缓存, 同一帧不重复解码)

    @property
    def is_connected(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def connect(self) -> None:
        self._q = self._ctx.Queue(maxsize=1)
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=self._target,
            args=(*self._args, self._q, self._stop),
            daemon=True,
            name=f"stream-{self.name}",
        )
        self._proc.start()

        start = time.time()
        while time.time() - start < self._first_frame_timeout:
            try:
                self._last = self._q.get(timeout=0.2)
                logger.info(
                    f"[{self.name}] 首帧已接收 shape={getattr(self._last, 'shape', None)} "
                    f"dtype={getattr(self._last, 'dtype', None)}"
                )
                return
            except Empty:
                if not self._proc.is_alive():
                    raise ConnectionError(f"[{self.name}] 工作进程在收到首帧前退出")
        logger.warning(f"[{self.name}] 等待首帧超时 ({self._first_frame_timeout}s), 进程已启动, 继续")

    def async_read(self) -> NDArray:
        """取队列最新帧 (非阻塞); 无新帧则返回上次缓存, 从未收到则返回空帧。"""
        new = False
        try:
            while True:
                self._last = self._q.get_nowait()
                new = True
        except Empty:
            pass

        if self._last is None:
            return self._empty_frame
        if self._decode_fn is None:
            return self._last
        # 有 decode_fn: 仅在拿到新一帧字节时解码, 同一帧重复读不重复解码
        if new or self._last_decoded is None:
            decoded = self._decode_fn(self._last)
            self._last_decoded = decoded if decoded is not None else self._empty_frame
        return self._last_decoded

    def disconnect(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None
        logger.info(f"[{self.name}] 已断开")
