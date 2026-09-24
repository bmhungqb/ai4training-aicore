#!/usr/bin/env bash
# Evaluate a trained EfficientGEBD checkpoint on the sewing Step Segmentation val split.
#
# Usage (run from src/step_segment/EfficientGEBD/):
#   bash script/test/test_sewing_csn.sh /path/to/model_best.pth
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT="${1:?Usage: test_sewing_csn.sh <checkpoint.pth>}"

torchrun --nproc_per_node 1 --master_port 1113 train.py \
  --config-file config-files/sewing_csn.yaml \
  --expname sewing_csn_x2_eval \
  --resume "${CKPT}" \
  --test-only \
  "${@:2}"
