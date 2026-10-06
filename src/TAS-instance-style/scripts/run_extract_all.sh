#!/usr/bin/env bash
# Extract chunk-based features for ALL videos in dataset/, for ALL backbones.
# Mask-crop (ROI cropping via original_data/processed_data/*.mask.png) is ON
# by default in extract_chunk_features.py.
#
# Usage:
#   bash src/TAS-instance-style/scripts/run_extract_all.sh
#   bash src/TAS-instance-style/scripts/run_extract_all.sh --overwrite      # force re-extraction
#   bash src/TAS-instance-style/scripts/run_extract_all.sh --no-mask-crop   # disable ROI cropping
#
# Any extra args are forwarded as-is to extract_chunk_features.py for every backbone.

set -euo pipefail

VENV_DIR="/home/hungbm/ai4training/venv"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="$REPO_ROOT/src/TAS-instance-style/scripts/extract_chunk_features.py"

BACKBONES=(videomae mvit_v1_b s3d i3d_r50)

cd "$REPO_ROOT"

for bb in "${BACKBONES[@]}"; do
  echo ""
  echo "=================================================================="
  echo "=== backbone: $bb"
  echo "=================================================================="
  python3 "$SCRIPT" --backbone "$bb" "$@"
done

echo ""
echo "All backbones done: ${BACKBONES[*]}"
