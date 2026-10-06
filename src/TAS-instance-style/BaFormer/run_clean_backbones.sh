#!/usr/bin/env bash
# ==============================================================================
# Script: run_clean_backbones.sh
# Purpose: Train Clean BaFormer Baseline aligned with problem_definition.md
#          All proposal 04-14 auxiliary losses and complex heuristics are TURNED OFF.
#          Uses Core Mask2Former Losses: CE + Mask + Dice + Standard Boundary.
#
# Usage:
#   ./run_clean_backbones.sh dinov3     # Run DINOv3 clean baseline
#   ./run_clean_backbones.sh dinov2     # Run DINOv2 clean baseline
#   ./run_clean_backbones.sh videomae   # Run VideoMAE clean baseline
#   ./run_clean_backbones.sh all        # Run all 3 sequentially
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate Python virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/hungbm/ai4training/venv/bin/activate" ]; then
        echo "[INFO] Activating virtual environment: /home/hungbm/ai4training/venv"
        source /home/hungbm/ai4training/venv/bin/activate
    fi
fi

PYTHON_BIN="$(which python)"
echo "[INFO] Using Python: $PYTHON_BIN"

TARGET="${1:-dinov3}"

run_backbone() {
    local bb="$1"
    local cfg="configs/tas_instance_clean_${bb}.yaml"

    if [ ! -f "$cfg" ]; then
        echo "[ERROR] Config file not found: $cfg"
        exit 1
    fi

    echo ""
    echo "=============================================================================="
    echo "  STARTING CLEAN BASELINE TRAINING: ${bb^^}"
    echo "  Config : $cfg"
    echo "  Output : experiments/tas_instance/bk_fde_tde/clean_${bb}/1/"
    echo "  Time   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    python main.py --config "$cfg"

    echo ""
    echo "[SUCCESS] Training for clean_${bb} completed!"
    echo "Artifacts saved at: experiments/tas_instance/bk_fde_tde/clean_${bb}/1/"
}

case "$TARGET" in
    dinov3)
        run_backbone "dinov3"
        ;;
    dinov2)
        run_backbone "dinov2"
        ;;
    videomae)
        run_backbone "videomae"
        ;;
    all)
        echo "[INFO] Running ALL 3 backbones sequentially: dinov3 -> dinov2 -> videomae"
        run_backbone "dinov3"
        run_backbone "dinov2"
        run_backbone "videomae"
        ;;
    *)
        echo "Usage: $0 [dinov3 | dinov2 | videomae | all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================================================="
echo "  ALL REQUESTED EXPERIMENTS FINISHED SUCCESSFULLY!"
echo "=============================================================================="
