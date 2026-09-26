#!/usr/bin/env bash
set -e

# AI Research Loop: Batch Runner for this Iteration
ITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$ITER_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRACK_DIR/../.." && pwd)"

echo "=========================================================="
echo " Starting Batch Experiments for: $(basename "$ITER_DIR")"
echo "=========================================================="

# List of experiment subdirectories in this iteration
EXPERIMENTS=(
    # "exp_001_wrist_filter"
    # "exp_002_adaptive_th"
)

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_PATH="$ITER_DIR/$EXP"
    if [ -d "$EXP_PATH" ]; then
        echo ""
        echo ">>> [RUNNING] $EXP ..."
        python "$EXP_PATH/run.py" --config "$EXP_PATH/config.yaml"
        echo ">>> [DONE] $EXP"
    else
        echo ">>> [WARNING] Subdirectory $EXP not found, skipping."
    fi
done

echo ""
echo "=========================================================="
echo " All experiments in batch finished!"
echo " Next step: Run Evaluator Agent to summarize and compare."
echo "=========================================================="
