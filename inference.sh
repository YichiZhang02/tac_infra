#!/bin/bash
# ============================================================================
# 用训练好的策略控制机械臂 (模型推理)
#
# 从臂读取观测 (含触觉 uint8) -> 策略前处理/推理/后处理 -> 下发给从臂执行。
# 与遥操作采数据共用同一条 observation->action->execute 管线，区别仅在动作来源是模型。
# 同时把每一帧观测+动作录制下来 (便于回放/评估)。
#
# 用法:  bash deploy_policy.sh <policy_path> <repo_id> <task> <num_episodes>
# 例:    bash deploy_policy.sh \
#            playground/results/models/xxx/checkpoints/last/pretrained_model \
#            local/eval_pick "Grab the cube" 10
# ============================================================================
set -e
REPO_ROOT=~/yichi/tac_infra        # 需调整为实际路径
cd "${REPO_ROOT}"

# =================== 可调参数 ===================
policy_path=${1:-"/mnt/data/xidong_data/tac_infra/playground/results/models/rm_umi_dual_pen_open_pi05_wristonly_true_tactile_encode_state_joint/checkpoints/last/pretrained_model"}

# --- 从 policy_path 自动推导一个可读的录制标签 ---
# 例: playground/results/models/<model_tag>/checkpoints/<ckpt_tag>/pretrained_model
#     -> model_tag=<model_tag>, ckpt_tag=<ckpt_tag>
model_tag=$(echo "${policy_path}" | sed -nE 's#.*/models/([^/]+)/.*#\1#p')
ckpt_tag=$(echo "${policy_path}" | sed -nE 's#.*/checkpoints/([^/]+)/.*#\1#p')
# 兜底: 推导失败时用路径中靠上的目录名
[ -z "${model_tag}" ] && model_tag=$(basename "$(dirname "$(dirname "$(dirname "${policy_path%/}")")")")
[ -z "${ckpt_tag}" ] && ckpt_tag=$(basename "$(dirname "${policy_path%/}")")
run_tag="$(date +%Y%m%d_%H%M%S)_${model_tag}_${ckpt_tag}"

repo_id=${2:-local/${run_tag}}                   # 评估录制数据集名 (默认按 policy_path 自动命名)
single_task=${3:-"Grab the object"}             # 任务文字描述 (需与训练时一致)
num_episodes=${4:-10}                             # 评估集数

robot_type=realman_tactile_shandd_hd             # uint8 触觉版从臂适配器
robot_id=realman_right
fps=30
episode_time_s=60                                # 每集最长推理秒数 (按右键提前结束)
reset_time_s=15
# 录制自动保存到 playground/record/<时间戳>_<model>_<ckpt>
dataset_root=${REPO_ROOT}/playground/record/${run_tag}
echo "录制将保存到: ${dataset_root}"

# =================== 启动 (无 teleop, 由 policy 产生动作) ===================
python -m deployment.record \
  --robot.type=${robot_type} \
  --robot.id=${robot_id} \
  --policy.path=${policy_path} \
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
