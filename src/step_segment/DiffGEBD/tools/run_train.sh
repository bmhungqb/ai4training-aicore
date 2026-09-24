#!/usr/bin/env bash
# Train DiffGEBD (ResNet-50 backbone + DiffFormer/Transformer diffusion head with
# Classifier-Free Guidance) on the industrial sewing Step Segmentation dataset.
#
# Prerequisites:
#   1. Build the dataset (frames + pickle annotations), run from repo root:
#        python tools/prepare_diff_gebd_dataset.py
#   2. pip install -r src/step_segment/DiffGEBD/requirements.txt
#
# Usage (run from src/step_segment/DiffGEBD/):
#   bash tools/run_train.sh [NUM_GPUS] [EXTRA_OPTS...]
#
# Examples:
#   bash tools/run_train.sh 1
#   bash tools/run_train.sh 2 SOLVER.BATCH_SIZE 4
#   nohup bash tools/run_train.sh 1 > train.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure a symlink to data/ exists if running from src/step_segment/DiffGEBD/
if [ ! -e data ] && [ -d ../../../data ]; then
  ln -s ../../../data data
fi

NUM_GPUS="${1:-1}"
shift || true

torchrun --nproc_per_node "${NUM_GPUS}" --master_port 10211 train.py \
  --config-file config/sewing_diffgebd_resnet50.yaml \
  --seed 42 \
  "$@"
