#!/usr/bin/env bash
# ==============================================================================
# Script: run_dataset_backbones.sh
# Purpose: Train Clean BaFormer Baseline on the NEW `dataset/` (built from
#          original_data/processed_data/ + balanced_split, chunk-based
#          temporal-backbone features from extract_chunk_features.py).
#
#          Same clean-baseline config as run_clean_backbones.sh (CE + Mask +
#          Dice + Standard Boundary only, auxiliary losses off), just pointed
#          at `dataset/` instead of `dataset_tas_instance/`, with the
#          matching feature dim / feat_folders / 5-class weights per backbone.
#
# Usage:
#   ./run_dataset_backbones.sh videomae    # VideoMAE  (768-dim, chunk-based)
#   ./run_dataset_backbones.sh mvit_v1_b   # MViT-v1-B (768-dim, chunk-based)
#   ./run_dataset_backbones.sh s3d         # S3D       (1024-dim, chunk-based)
#   ./run_dataset_backbones.sh i3d_r50     # I3D-R50   (2048-dim, chunk-based)
#   ./run_dataset_backbones.sh dinov2      # DINOv2    (768-dim, frame-by-frame)
#   ./run_dataset_backbones.sh all         # run all 5 sequentially
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate Python virtual environment if not already activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "/home/hungbm/ai4training/venv/bin/activate" ]; then
        echo "[INFO] Activating virtual environment: /home/hungbm/ai4training/venv"
        source /home/hungbm/ai4training/venv/bin/activate
    fi
fi

PYTHON_BIN="$(which python)"
echo "[INFO] Using Python: $PYTHON_BIN"

TARGET="${1:-videomae}"

BACKBONES=(videomae mvit_v1_b s3d i3d_r50 dinov2)

run_backbone() {
    local bb="$1"
    local cfg="configs/dataset_clean_${bb}.yaml"

    if [ ! -f "$cfg" ]; then
        echo "[ERROR] Config file not found: $cfg"
        exit 1
    fi

    echo ""
    echo "=============================================================================="
    echo "  STARTING TRAINING ON dataset/ : ${bb^^}"
    echo "  Config : $cfg"
    echo "  Output : experiments/tas_instance/bk_fde_tde/dataset_clean_${bb}/1/"
    echo "  Time   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    python main.py --config "$cfg"

    echo ""
    echo "[SUCCESS] Training for dataset_clean_${bb} completed!"
    echo "Artifacts saved at: experiments/tas_instance/bk_fde_tde/dataset_clean_${bb}/1/"
}

case "$TARGET" in
    videomae|mvit_v1_b|s3d|i3d_r50|dinov2)
        run_backbone "$TARGET"
        ;;
    all)
        echo "[INFO] Running ALL 4 backbones sequentially: ${BACKBONES[*]}"
        for bb in "${BACKBONES[@]}"; do
            run_backbone "$bb"
        done
        ;;
    *)
        echo "Usage: $0 [videomae | mvit_v1_b | s3d | i3d_r50 | dinov2 | all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================================================="
echo "  ALL REQUESTED EXPERIMENTS FINISHED SUCCESSFULLY!"
echo "=============================================================================="
