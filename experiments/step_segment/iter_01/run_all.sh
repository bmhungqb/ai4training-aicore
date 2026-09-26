#!/usr/bin/env bash
set -e

# AI Research Loop: Batch Runner for Step Segmentation (iter_01)
ITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$ITER_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRACK_DIR/../.." && pwd)"

MODE="${1:-train}"  # "train" or "infer"

echo "=========================================================="
echo " Starting Batch Experiments for: $(basename "$ITER_DIR")"
echo " Execution Mode: $MODE"
echo "=========================================================="

EXPERIMENTS=(
    "exp_001_efficientgebd_baseline"
    "exp_002_diffgebd_chunked"
    "exp_003_ddm_net_baseline"
)

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_PATH="$ITER_DIR/$EXP"
    if [ -d "$EXP_PATH" ]; then
        echo ""
        echo ">>> [RUNNING] $EXP ($MODE) ..."
        python "$EXP_PATH/run.py" --mode "$MODE"
        echo ">>> [COMPLETED] $EXP"
    else
        echo ">>> [WARNING] Subdirectory $EXP not found, skipping."
    fi
done

echo ""
echo "=========================================================="
echo " Batch finished!"
echo " Next step: Run Evaluator Agent to summarize and compare:"
echo "   Outputs are in outputs/step_segment/iter_01/"
echo "=========================================================="
