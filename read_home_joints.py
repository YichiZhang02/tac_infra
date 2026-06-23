#!/usr/bin/env python3
"""读取当前机械臂关节角度，输出可直接粘贴到 inference.sh 的 --robot.home_joints 参数。"""

import sys
import time
import json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from deployment.hardware.follower_arms.realman_tcp import RealmanTcpFollower

ARMS = {
    "left":  "192.168.1.200",
    "right": "192.168.1.201",
}
PORT = 8080
JOINT_NAMES = [f"main_joint{i}" for i in range(1, 8)]

home_joints = {}

for side, ip in ARMS.items():
    print(f"连接 {side} 臂 ({ip})...")
    arm = RealmanTcpFollower(ip=ip, port=PORT, dof=7, use_degrees=False, name=f"{side}_follower")
    arm.connect()
    time.sleep(0.2)  # 等后台线程首帧
    joints = arm.read_joints_now()
    arm.disconnect()
    if joints is None:
        print(f"  [错误] {side} 臂读取失败，跳过")
        continue
    print(f"  {side}: {[f'{v:.4f}' for v in joints]}")
    for i, jname in enumerate(JOINT_NAMES):
        home_joints[f"{side}_{jname}"] = round(float(joints[i]), 6)

print("\n========== 复制到 inference.sh ==========")
print(f"--robot.home_joints='{json.dumps(home_joints)}'")
print("=========================================")
