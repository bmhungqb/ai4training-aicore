#!/bin/bash
# Runner: exp_001_baseline for DDM-Net + EfficientGEBD, with early stopping enabled.
#
# Early stopping is configured in each model's config.yaml:
#   training_params.early_stopping: true
#   training_params.early_stopping_patience: 5
#   training_params.early_stopping_min_delta: 0.0
#
# - DDM-Net:       Lightning `EarlyStopping` callback, monitors val/<eval_metric>.
# - EfficientGEBD: patience counter around the model_best.pth save loop, monitors F1.
#
# Usage:
#   bash experiments/step_segment/run_exp01_ddm_efficient_early_stopping.sh [--dry-run]
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo " exp_001_baseline: DDM-Net + EfficientGEBD (Early Stopping ON)"
echo "=========================================================="

echo ""
echo "[1/2] Running DDM-Net exp_001_baseline..."
python3 "$DIR/ddm_net/iter_01/exp_001_baseline/run.py" --mode train "$@"

echo ""
echo "[2/2] Running EfficientGEBD exp_001_baseline..."
python3 "$DIR/efficient_gebd/iter_01/exp_001_baseline/run.py" --mode train "$@"

echo ""
echo "=========================================================="
echo " Both baseline runs completed! Check overview tracker:"
echo "   $DIR/step_segment_overview.md"
echo "=========================================================="
