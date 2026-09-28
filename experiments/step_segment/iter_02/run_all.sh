#!/usr/bin/env bash
set -e

# AI Research Loop: Batch Runner for step_segment / iter_02 (Revision 2,
# approved per Human sign-off on 03_research_plan.md Rev 2)
ITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$ITER_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRACK_DIR/../.." && pwd)"

echo "=========================================================="
echo " Starting Batch Experiments for: $(basename "$ITER_DIR")"
echo "=========================================================="

# Ordered per Debater's stated execution order (04_debate_verdict.md WARN):
# exp_002 (CPU-only, zero-cost, zero-risk, reads already-saved artifacts)
# runs FIRST since its result is a precondition for correctly interpreting
# exp_001's noise-band comparison. exp_003 is independent (CPU-only,
# descriptive profiling). exp_004 is the corrective intervention, run last
# since its expected effect (collapsing the noise floor) is best interpreted
# after exp_001/exp_002 establish the pre-fix noise/confound picture.
EXPERIMENTS=(
    "exp_002_premerge_boundary_audit"
    "exp_001_overlap_treatment_noise_floor"
    "exp_003_session_confound_profile"
    "exp_004_deterministic_noise_seeding"
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
