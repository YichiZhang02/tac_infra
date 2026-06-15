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
腕部鱼眼相机 (fish_camera gRPC + UDP)。

板子上 fish_camera gRPC 服务 OpenStream(MJPG) -> FCP1 分片 UDP -> 本机重组 + imdecode,
输出原生分辨率 RGB uint8。

收流跑在独立进程 (spawn): gRPC/解码在单进程多线程里会和触觉互相饿死 GIL, 故每路一个进程,
通过 mp.Queue(maxsize=1) 只回传最新帧。本模块自包含 (不与触觉共享收流代码)。

⚠️ 队列里传的是原始 JPEG 字节 (~150KB) 而非解码后的 ~6MB 数组: 大数组过 mp.Queue 要在
feeder 线程 pickle ~8ms, 高帧率下消费端频繁取到旧缓存 -> 重复帧/卡顿。解码移到消费端
(主进程 async_read) 进行。
"""

import logging
import multiprocessing as mp
import socket
import time
from queue import Empty
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .._sdk_paths import ensure_fisheye_sdk
from .base import WristCameraBase

logger = logging.getLogger(__name__)


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


def _fisheye_worker(ip, grpc_port, udp_port, want_w, want_h, max_datagram, max_fps, debug, q, stop):
    """鱼眼相机工作进程: gRPC OpenStream(MJPG) + UDP 重组, 队列传原始 JPEG 字节。

    debug=True 时每 ~2s 打印: 产出fps / UDP数据报到达率, 用于区分瓶颈在板子+网线还是 PC 解码。
    """
    ensure_fisheye_sdk()

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
            # 打印实际选定的分辨率: 若请求的 want_w x want_h 板子不支持, 会回退到最大模式。
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


class FisheyeGrpcCamera(WristCameraBase):
    """腕部鱼眼相机 (fish_camera gRPC + UDP), 自带独立收流进程。"""

    def __init__(
        self,
        ip: str,
        grpc_port: int,
        udp_port: int,
        width: int,
        height: int,
        max_datagram: int = 1200,
        max_fps: float = 0.0,
        debug: bool = False,
        first_frame_timeout: float = 15.0,
        name: str = "cam_wrist",
    ):
        self.name = name
        self._ip = ip
        self._grpc_port = grpc_port
        self._udp_port = udp_port
        self._width = width
        self._height = height
        self._max_datagram = max_datagram
        self._max_fps = max_fps
        self._debug = debug
        self._first_frame_timeout = first_frame_timeout

        self._empty_frame = np.zeros((height, width, 3), dtype=np.uint8)

        self._ctx = mp.get_context("spawn")
        self._q = None
        self._stop = None
        self._proc = None
        self._last = None            # 队列里的原始 JPEG 字节
        self._last_decoded = None    # 解码后的数组缓存 (同一帧不重复解码)

    @property
    def is_connected(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def connect(self) -> None:
        self._q = self._ctx.Queue(maxsize=1)
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_fisheye_worker,
            args=(self._ip, self._grpc_port, self._udp_port,
                  self._width, self._height, self._max_datagram,
                  self._max_fps, self._debug, self._q, self._stop),
            daemon=True,
            name=f"stream-{self.name}",
        )
        self._proc.start()

        start = time.time()
        while time.time() - start < self._first_frame_timeout:
            try:
                self._last = self._q.get(timeout=0.2)
                logger.info(f"[{self.name}] 首帧已接收 (JPEG {len(self._last)} 字节)")
                return
            except Empty:
                if not self._proc.is_alive():
                    raise ConnectionError(f"[{self.name}] 工作进程在收到首帧前退出")
        logger.warning(f"[{self.name}] 等待首帧超时 ({self._first_frame_timeout}s), 进程已启动, 继续")

    def async_read(self) -> NDArray:
        """取最新帧 (非阻塞); 无新帧返回上次解码缓存, 从未收到则返回空帧。"""
        new = False
        try:
            while True:
                self._last = self._q.get_nowait()
                new = True
        except Empty:
            pass

        if self._last is None:
            return self._empty_frame
        if new or self._last_decoded is None:
            decoded = decode_fisheye_jpeg(self._last)
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
