#!/bin/sh
set -e
cd "$(dirname "$0")"   # 切到仓库根, 使 playground/... 相对路径生效, 服务器/本地通用

# =================== 可调参数 ===================
pretrained_id=${1:-20260624_230105_rm_umi_dual_260617_pen_place_cap_notop_undist_256_diffusion_wristonly_true_tactile_none_state_episode_ee_action_relative_ee}
step=${2:-15000}

#   n_action_steps      执行 chunk 里的前几个动作后再重规划 (chunk[:n])。chunk_size 与训练同步, 不用动。
#   action_start_offset 执行前先丢掉 chunk 的前 m 个动作 (chunk[m:]); 实际执行 chunk[m : m+n]。
action_start_offset=${3:-0}
n_action_steps=${4:-20}

# ===============================================
# step 自动补零到 6 位: 5000 -> 005000 (expr 强制十进制, 兼容已带前导零的输入, POSIX sh 可用)
step=$(printf "%06d" "$(expr "$step" + 0)")

# 仅在显式给了值时才拼 override (空则完全不出现, 保持用 checkpoint 默认)
policy_overrides=""
[ -n "$n_action_steps" ] && policy_overrides="$policy_overrides --policy.n_action_steps=${n_action_steps}"
[ -n "$action_start_offset" ] && policy_overrides="$policy_overrides --policy.action_start_offset=${action_start_offset}"

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
  ${policy_overrides} \
  --match_policy=true \
  --robot.home_joints='{"left_main_joint1": -0.109262, "left_main_joint2": 0.235679, "left_main_joint3": 0.118975, "left_main_joint4": 1.265910, "left_main_joint5": 0.034194, "left_main_joint6": 1.589552, "left_main_joint7": -0.278270, "right_main_joint1": 0.041508, "right_main_joint2": 0.100594, "right_main_joint3": 0.046601, "right_main_joint4": 1.527823, "right_main_joint5": 0.011595, "right_main_joint6": 1.477732, "right_main_joint7": 0.472311}' \
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
