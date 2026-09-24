#!/usr/bin/env bash
# Train EfficientGEBD (video-domain CSN backbone + FPN + DiffFormer/DiffMixer heads)
# on the industrial sewing Step Segmentation dataset.
#
# Prerequisites:
#   1. Build the dataset (frames + pickle annotations):
#        python tools/prepare_efficient_gebd_dataset.py
#   2. Download CSN-pretrained/ (mmaction2 configs + ig65m checkpoints), see README.md.
#   3. pip install -r src/step_segment/EfficientGEBD/requirements.txt
#
# Usage (run from src/step_segment/EfficientGEBD/):
#   bash script/train/train_sewing_csn.sh [NUM_GPUS]
set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_GPUS="${1:-1}"

torchrun --nproc_per_node "${NUM_GPUS}" --master_port 1112 train.py \
  --config-file config-files/sewing_csn.yaml \
  --expname sewing_csn_x2 \
  "${@:2}"
