#!/bin/bash
# ============================================================================
# 遥操作采数据 (睿尔曼 RM75b + 16路触觉，触觉以 uint8 保存)
#
# 人手拖动主臂 (Leader)，从臂 (Follower) 跟随运动，同时记录:
#   observation.state            7关节 + 1夹爪 (弧度)
#   observation.images.cam_top         全景相机 (D515) 896x896 RGB uint8
#   observation.images.cam_right_wrist 腕部相机 (D405) 480x480 RGB uint8
#   observation.images.cam_finger0/1   触觉 (Shear+Depth) 240x320x3 uint8
#   action                       主臂 7关节 + 1夹爪 (弧度)
#
# 键盘控制 (终端需有焦点): 左方向键=复位重录当前集, 右方向键=保存当前集
#
# 用法:  bash record_teleop.sh <repo_id> <task> <num_episodes> <leader_port>
# 例:    bash record_teleop.sh local/rm_pick_cube "Grab the cube" 30 /dev/ttyLeaderR
# ============================================================================
set -e
REPO_ROOT=/mnt/data/xidong_data/tac_infra        # 需调整为实际路径
cd "${REPO_ROOT}"

# =================== 可调参数 ===================
repo_id=${1:-local/rm_tactile_demo}              # 数据集名 (本地保存到 playground/data 下)
single_task=${2:-"Grab the object"}             # 任务文字描述 (会写入每一帧)
num_episodes=${3:-30}                             # 录制集数
leader_port=${4:-/dev/ttyLeaderR}               # 主臂 USB 串口

robot_type=realman_ugripper_dual              # 双臂 (触觉随 use_tactile, 默认开)
robot_id=realman_dual
fps=30
episode_time_s=60                                # 每集最长录制秒数 (可中途按右键提前保存)
reset_time_s=15                                  # 集间复位场景秒数
dataset_root=${REPO_ROOT}/playground/data/${repo_id##*/}

# =================== 启动 ===================
python -m deployment.collect \
  --robot.type=${robot_type} \
  --robot.id=${robot_id} \
  --teleop.type=bi_realman_ugripper_leader \
  --dataset.repo_id=${repo_id} \
  --dataset.root=${dataset_root} \
  --dataset.single_task="${single_task}" \
  --dataset.num_episodes=${num_episodes} \
  --dataset.fps=${fps} \
  --dataset.episode_time_s=${episode_time_s} \
  --dataset.reset_time_s=${reset_time_s} \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --display_data=false \
  --play_sounds=true
