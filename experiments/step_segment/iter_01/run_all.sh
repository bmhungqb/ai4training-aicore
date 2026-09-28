#!/usr/bin/env bash
set -e

# AI Research Loop: Batch Runner for step_segment / iter_01 (Revision 2, approved 04_debate_verdict.md)
ITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$ITER_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRACK_DIR/../.." && pwd)"

echo "=========================================================="
echo " Starting Batch Experiments for: $(basename "$ITER_DIR")"
echo "=========================================================="

# Ordered: exp_000/exp_000b are zero-cost BLOCKING GATES that must be
# inspected (status == PASS/WARN, hypothesis_*_generalizes == true) before
# trusting exp_002/exp_003. Script does not auto-abort on WARN/DE-PRIORITIZE
# -- Human must review each gate's report before proceeding to the next step.
EXPERIMENTS=(
    "exp_000_training_health_audit"
    "exp_000b_aggregate_error_histograms"
    "exp_002_val_overlap_context"
    "exp_003_min_peak_distance_suppression"
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
