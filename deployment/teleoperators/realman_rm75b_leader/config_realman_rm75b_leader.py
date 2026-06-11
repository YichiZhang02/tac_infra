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
睿尔曼 RM75b 主臂遥操作器配置
"""

from dataclasses import dataclass, field

from deployment.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("realman_rm75b_leader")
@dataclass
class RealmanRM75bLeaderConfig(TeleoperatorConfig):
    """
    睿尔曼 RM75b 主臂遥操作器配置
    
    主臂通过 USB 串口连接，用于遥操作控制从臂
    """
    
    # 设备类型标识
    type: str = "realman_rm75b_leader"
    
    # ============ 主臂串口配置 ============
    # 主臂串口路径
    port: str = "/dev/ttyLeaderR"
    
    # 串口波特率
    baudrate: int = 460800
    
    # 串口通信协议数据 (55 AA 02 00 00 67 = 请求关节角度)
    hex_data: str = "55 AA 02 00 00 67"
    
    # ============ 数据格式配置 ============
    # 是否使用角度制 (False = 弧度制)
    use_degrees: bool = False

    # ============ 异步读取配置 ============
    # 启用异步读取模式 (用于 60Hz 全局采集)
    # 当 async_read=True 时，connect() 会启动后台线程持续读取主臂位置，
    # get_action() 从缓存读取而非同步串口通信，避免 ~22ms 阻塞。
    async_read: bool = False

    # ============ 标识配置 ============
    # 设备 ID
    id: str | None = "realman_leader"
