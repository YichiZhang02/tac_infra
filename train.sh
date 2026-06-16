#!/bin/bash
cd "$(dirname "$0")" || exit 1   # 切到脚本所在目录(仓库根), 服务器/本地通用, 无需手改
REPO_ROOT="$(pwd)"               # 自动探测 (仅用于 PYTHONPATH 等运行期, 不写进保存的 config)

# ===================模型和数据集配置===================
dataset_id=${1:-rm_umi_dual_pen_open}
policy_type=${2:-diffusion}          # act | diffusion | pi05 | starvla_groot

# 两个不同概念，按 policy_type 绑定：
#   pretrained_path = 完整 policy 检查点（含 model.safetensors + 预/后处理器），
#     传给 --policy.pretrained_path，会触发 from_pretrained 加载整个 policy。pi05_base 属于这种。
#   base_vlm = 仅 starvla_groot 用，是底座 VLM（HF 模型目录），传给 --policy.base_vlm，
#     policy 本身从零训练，所以这种情况 pretrained_path 必须留空。
pretrained_path=
base_vlm=
case "${policy_type}" in
  pi05)          pretrained_path=playground/pretrained_models/pi05_base ;;
  starvla_groot) base_vlm=playground/pretrained_models/Qwen3.5-0.8B ;;
  act|diffusion) : ;;  # 这两个从零训练
  *)             echo "Unknown policy_type: ${policy_type} (expected act|diffusion|pi05|starvla_groot)"; exit 1 ;;
esac
# ===================训练配置===================
gpu_id=0
num_processes=${3:-1}
batch_size=${4:-32}
steps=${5:-10_000}
save_freq=5_000
log_freq=100

# ===================视觉/触觉路由（四个 framework 通用）===================
# wrist_only: true 只用 wrist 相机; false 用 top + wrist
# tactile_mode: none(触觉不进模型) / as_image(触觉作为图像输入) / encode(触觉 encoder, 预留未实现)
# state_mode: none(完全不用 state) / joint(关节角) / ee(末端位姿，当前预留未实现)
wrist_only=${6:-false}
tactile_mode=${7:-none}
state_mode=${8:-joint}

# tactile encoder（仅 tactile_mode=encode 时生效）
#   tactile_encoder_path: tactile-MAE 权重（.pth）或 HF 目录，作为 encoder 初始化；
#     arch / sensor_id / image_size 会从 checkpoint 自动读取，无需手动指定。
#     注意：encode 模式下 tactile-MAE encoder + query token 会随 policy 一起训练（非冻结）。
#   tactile_insert_location: encoder | decoder（Diffusion 忽略该项）。
#   tactile_num_tokens: 每张触觉图的可学习 query token 数（默认 8）；总 token = 指数 × 该值。
tactile_encoder_path=${TACTILE_ENCODER_PATH:-${9:-playground/pretrained_models/AnyTouch-ViT-L-16}}
tactile_insert_location=${TACTILE_INSERT_LOCATION:-${10:-encoder}}
tactile_num_tokens=${TACTILE_NUM_TOKENS:-8}

# ===================相机 / 触觉 key（按数据集命名调整，均支持多路）===================
# 三者都是 draccus 列表，语法为 [key1,key2,...]（无需给每个元素加引号）。
# 可多路：如双臂数据 rm_umi_dual_pen_open 有 left/right 两路 wrist + 四路 finger。
#   top_cam:      顶部/全景相机，可多路；wrist_only=true 时被忽略
#   wrist_cam:    腕部相机，可多路（双臂 left+right）
#   tactile_keys: 触觉图 key，可多路（tactile_mode=as_image/encode 时生效）
top_cam=${TOP_CAM:-'[observation.images.cam_top]'}
wrist_cam=${WRIST_CAM:-'[observation.images.left_cam_wrist,observation.images.right_cam_wrist]'}
tactile_keys=${TACTILE_KEYS:-'[observation.images.left_cam_finger0,observation.images.left_cam_finger1,observation.images.right_cam_finger0,observation.images.right_cam_finger1]'}

policy_suffix="wristonly_${wrist_only}_tactile_${tactile_mode}_state_${state_mode}"
# 运行名: <时间>_<数据集>_<framework>_<路由后缀>, 用于输出目录/job_name/日志名 (保持一致)
run_name="$(date +%Y%m%d_%H%M%S)_${dataset_id}_${policy_type}_${policy_suffix}"

# ===================路径配置 (相对路径, 会被写进 train_config.json -> 跨机器可移植)===================
dataset_root=playground/data
output_root=playground/results/models
output_dir=${output_root}/${run_name}

# ===================按 framework 拼装专属参数===================
# 不同 framework 的配置字段不同（act/diffusion 没有 dtype/compile_model/
# gradient_checkpointing 等），统一在这里按类型追加，避免 draccus 报未知参数。
extra_args=""
case "${policy_type}" in
  pi05)
    extra_args="${extra_args} --policy.dtype=bfloat16 --policy.compile_model=false --policy.gradient_checkpointing=false"
    ;;
  starvla_groot)
    extra_args="${extra_args} --policy.dtype=bfloat16 --policy.gradient_checkpointing=false --policy.base_vlm=${base_vlm}"
    ;;
  act|diffusion)
    : # 这两个没有 VLM/dtype 相关字段
    ;;
  *)
    echo "Unknown policy_type: ${policy_type} (expected act|diffusion|pi05|starvla_groot)"; exit 1
    ;;
esac

# pretrained_path 是所有 framework 的公共字段；非空才传
if [ -n "${pretrained_path}" ]; then
  extra_args="${extra_args} --policy.pretrained_path=${pretrained_path}"
fi

# tactile_mode=encode 时追加 tactile encoder 相关参数（四个 framework 通用）
if [ "${tactile_mode}" = "encode" ]; then
  if [ -z "${tactile_encoder_path}" ]; then
    echo "tactile_mode=encode 需要提供 TACTILE_ENCODER_PATH（或第 9 个位置参数）指向 tactile-MAE 权重"; exit 1
  fi
  extra_args="${extra_args} --policy.tactile_encoder_path=${tactile_encoder_path}"
  extra_args="${extra_args} --policy.tactile_insert_location=${tactile_insert_location}"
  extra_args="${extra_args} --policy.tactile_num_tokens=${tactile_num_tokens}"
fi

# ===================日志文件===================
# 训练日志（含下面的参数打印 + accelerate/训练全部 stdout+stderr）保存到 output_dir 下，
# 固定文件名 train.log，每次运行覆盖（只留最新）。终端仍实时显示（靠 tee）。
log_dir=playground/logs/models
mkdir -p "${log_dir}"
log_file="${log_dir}/${run_name}.log"

# 用花括号组把「参数打印 + 训练」整体管道给 tee，这样日志里既有配置也有训练过程。
# 注：dash 不支持 set -o pipefail，脚本退出码会是 tee 的(0)；这是训练启动器，可接受。
{
echo "Log file: $log_file"
echo "Training with dataset: $dataset_id"
echo "Policy type: $policy_type"
echo "Pretrained path: ${pretrained_path:-<scratch>} | Base VLM: ${base_vlm:-<none>}"
echo "Steps: $steps | Batch size: $batch_size | Num processes: $num_processes"
echo "Wrist only: $wrist_only | Tactile mode: $tactile_mode | State mode: $state_mode"
echo "Top cam keys:   ${top_cam}"
echo "Wrist cam keys: ${wrist_cam}"
echo "Tactile keys:   ${tactile_keys}"
if [ "${tactile_mode}" = "encode" ]; then
  echo "Tactile encoder path: ${tactile_encoder_path} | Insert: ${tactile_insert_location} | Num tokens: ${tactile_num_tokens} (encoder trained jointly)"
fi
echo "Output dir: $output_dir"
echo "Extra args: ${extra_args}"

PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch \
    --num_processes=$num_processes \
    -m vtla.train \
    --dataset.repo_id=$dataset_id \
    --dataset.root=${dataset_root}/${dataset_id} \
    --dataset.video_backend=pyav \
    --policy.type=${policy_type} \
    --policy.wrist_only=${wrist_only} \
    --policy.tactile_mode=${tactile_mode} \
    --policy.state_mode=${state_mode} \
    --policy.top_camera_keys="${top_cam}" \
    --policy.wrist_camera_keys="${wrist_cam}" \
    --policy.tactile_keys="${tactile_keys}" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    ${extra_args} \
    --output_dir=${output_dir} \
    --job_name=${run_name} \
    --steps=${steps} \
    --save_freq=${save_freq} \
    --batch_size=${batch_size} \
    --log_freq=${log_freq} \
    --tolerance_s=0.04 \
    --wandb.enable=false
} 2>&1 | tee "$log_file"
