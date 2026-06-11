#!/bin/bash
# Tactile MAE pretraining launcher.
#
# Usage: scripts/train.sh <init_mode> [arch] [num_gpus] [dataset_ids...]
#   init_mode : scratch | clip | anytouch
#   arch      : vit_l (default) | vit_b
#
# Examples:
#   scripts/train.sh anytouch vit_l 4 rm_nist_260320_strawberry rm_nist_260520_usb
#   scripts/train.sh clip vit_l 4 rm_nist_260320_strawberry
#   scripts/train.sh scratch vit_b 1 rm_nist_260320_strawberry
set -e

init_mode=${1:-anytouch}
arch=${2:-vit_l}
num_gpus=${3:-4}
shift $(( $# < 3 ? $# : 3 ))
dataset_ids=${@:-rm_nist_260320_strawberry}

REPO_ROOT=/mnt/data/tac_infra_train
cd "${REPO_ROOT}"
PM=playground/pretrained_models

case "${init_mode}" in
  scratch)  pretrained_path="" ;;
  clip)     pretrained_path="${PM}/CLIP-ViT-L-14-DataComp.XL-s13B-b90K" ;;   # vit_l only out-of-the-box
  anytouch) pretrained_path="${PM}/checkpoint.pth" ;;                        # or converted: ${PM}/anytouch_mae_vitl
  *) echo "Unknown init_mode: ${init_mode} (scratch|clip|anytouch)"; exit 1 ;;
esac

output_dir=playground/results/tac_mae_${arch}_${init_mode}
mkdir -p "${output_dir}"

launcher="python -m"
if [ "${num_gpus}" -gt 1 ]; then
  launcher="torchrun --nproc_per_node=${num_gpus} -m"
fi

echo "init_mode=${init_mode} arch=${arch} gpus=${num_gpus} datasets=[${dataset_ids}]"
echo "pretrained_path=${pretrained_path:-<scratch>}  output_dir=${output_dir}"

PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} ${launcher} vtla.tac_encoder.tactile_mae.train \
  --arch ${arch} \
  --pretrained_path "${pretrained_path}" \
  --dataset_root playground/data \
  --dataset_ids ${dataset_ids} \
  --camera_keys observation.images.cam_finger0 observation.images.cam_finger1 \
  --sensor_id -1 \
  --mask_ratio 0.75 \
  --use_sensor_token \
  --use_same_patchemb \
  --sensor_token_for_all --beta_start 0.0 --beta_end 0.75 \
  --batch_size 64 --epochs 20 --warmup_epochs 1 \
  --weight_decay 0.1 --blr 1e-3 \
  --val_ratio 0.05 --tolerance_s 0.1 \
  --num_workers 12 \
  --output_dir "${output_dir}" \
  2>&1 | tee "${output_dir}/train.log"
