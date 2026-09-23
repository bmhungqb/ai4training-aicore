#!/bin/bash
# Launch DDM-Net training on the step-segmentation dataset.
#
# Run from the repository root (ai4training-aicore-poc/):
#   bash src/step_segment/DDM-Net/tools/run_train.sh [NUM_GPUS]
#
# Prerequisites:
#   python tools/process_step_segments.py
#   python tools/prepare_ddm_dataset.py --split-mode random --val-ratio 0.2 --seed 42
set -euo pipefail

NUM_GPUS="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${SCRIPT_DIR}/train_sop_lightning.py" \
  --config "${SCRIPT_DIR}/config/ddm_train_config.yaml" \
  --exp-name step_segment_ddm_net \
  --backbone resnet50 \
  --output lightning_output \
  --pretrained True \
  --learning-rate 0.0001 \
  --min-lr 1e-10 \
  --warmup-epochs 0 \
  --epochs 30 \
  --decay-epochs 2 \
  --decay-rate 0.5 \
  --model-ema \
  --model-ema-decay 0.999 \
  --model-ema-start-epoch 10 \
  --eval-metric f1_score \
  --num-workers 4 \
  --num-gpus "${NUM_GPUS}" \
  --save-visualizations
