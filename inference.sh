#!/bin/sh
set -e
cd "$(dirname "$0")"   # 切到仓库根, 使 playground/... 相对路径生效, 服务器/本地通用

# =================== 可调参数 ===================
pretrained_id=${1:-20260623_162109_rm_umi_dual_260618_pick_majiang_256_diffusion_wristonly_false_tactile_none_state_joint_action_joint}
step=${2:-5000}

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
  --robot.home_joints='{"left_main_joint1": -0.091996, "left_main_joint2": 0.200154, "left_main_joint3": 0.062090, "left_main_joint4": 0.916534, "left_main_joint5": -0.040657, "left_main_joint6": 1.984177, "left_main_joint7": -0.111946, "right_main_joint1": 0.091996, "right_main_joint2": 0.200154, "right_main_joint3": -0.062090, "right_main_joint4": 0.916534, "right_main_joint5": 0.040657, "right_main_joint6": 1.984177, "right_main_joint7": 0.111946}' \
  --robot.home_gripper=1.0

# --- 手动模式示例 (关掉自动对齐时, 硬件/任务需自己给) ---
# python -m deployment.inference \
#   --robot.type=realman_ugripper_dual \
#   --policy.path=${policy_path} \
#   --dataset.repo_id=${repo_id} \
#   --match_policy=false \
#   --robot.use_tactile=false \
#   --dataset.single_task="Grasp the cap and pull it off the pen."
