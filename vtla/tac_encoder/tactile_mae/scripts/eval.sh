#!/bin/bash
# Tactile MAE evaluation + visualization launcher.
#
# Usage: scripts/eval.sh <checkpoint> [arch] [dataset_ids...]
# Example:
#   scripts/eval.sh playground/results/tac_mae_vit_l_anytouch/checkpoint-19.pth vit_l rm_nist_260320_strawberry
set -e

checkpoint=${1:-playground/pretrained_models/checkpoint.pth}
arch=${2:-vit_l}
shift $(( $# < 2 ? $# : 2 ))
dataset_ids=${@:-rm_nist_260320_strawberry}

REPO_ROOT=/mnt/data/tac_infra_train
cd "${REPO_ROOT}"
output_dir=playground/results/tac_mae_eval

PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} python -m vtla.tac_encoder.tactile_mae.eval \
  --arch ${arch} \
  --checkpoint "${checkpoint}" \
  --dataset_root playground/data \
  --dataset_ids ${dataset_ids} \
  --camera_keys observation.images.cam_finger0 observation.images.cam_finger1 \
  --sensor_id -1 \
  --val_ratio 0.1 --split val \
  --mask_ratio 0.75 \
  --use_sensor_token --use_same_patchemb \
  --recon_metric --recon_vis --tsne \
  --tsne_samples 1000 \
  --output_dir "${output_dir}"
