#!/bin/sh
set -e
cd "$(dirname "$0")"   # 切到仓库根, 使 playground/... 相对路径生效, 服务器/本地通用

# =================== 可调参数 ===================
pretrained_id=${1:-20260624_230105_rm_umi_dual_260617_pen_place_cap_notop_undist_256_diffusion_wristonly_true_tactile_none_state_episode_ee_action_relative_ee}
step=${2:-15000}

# step 自动补零到 6 位: 5000 -> 005000 (expr 强制十进制, 兼容已带前导零的输入, POSIX sh 可用)
step=$(printf "%06d" "$(expr "$step" + 0)")

policy_path=playground/results/models/${pretrained_id}/checkpoints/${step}/pretrained_model
echo "测试policy: ${policy_path}"

# 按时间命名: local/<时间戳>_<基础名>
name=${pretrained_id}_step_${step}    
repo_id="local/eval_$(date +%Y%m%d_%H%M%S)_${name}"
echo "录制数据集: ${repo_id}  ->  playground/eval/${repo_id##*/}"

# =================== 启动 (match_policy 自动对齐硬件 + 任务) ===================
python -m deployment.inference \
  --robot.type=realman_ugripper_dual \
  --policy.path=${policy_path} \
  --dataset.repo_id=${repo_id} \
  --match_policy=true \
  --robot.home_joints='{"left_main_joint1": -0.018531, "left_main_joint2": 0.24981, "left_main_joint3": 0.155477, "left_main_joint4": 1.486658, "left_main_joint5": 0.042135, "left_main_joint6": 1.292984, "left_main_joint7": 0.120061, "right_main_joint1": 0.053178, "right_main_joint2": 0.188806, "right_main_joint3": -0.154118, "right_main_joint4": 1.481587, "right_main_joint5": -0.002833, "right_main_joint6": 1.343616, "right_main_joint7": -0.149679}' \
  --robot.home_gripper=1.0 \
  --robot.max_ee_pos_step_m=0.1  # ee用这个值来防止直接撞 初次0.01 后面0.1

# --- 手动模式示例 (关掉自动对齐时, 硬件/任务需自己给) ---
# python -m deployment.inference \
#   --robot.type=realman_ugripper_dual \
#   --policy.path=${policy_path} \
#   --dataset.repo_id=${repo_id} \
#   --match_policy=false \
#   --robot.use_tactile=false \
#   --dataset.single_task="Grasp the cap and pull it off the pen."
