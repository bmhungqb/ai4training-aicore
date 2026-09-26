#!/bin/bash
# Master Runner: Baseline Experiments for all 3 Step Segmentation Models
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo " Starting Step Segmentation Multi-Model Baseline Benchmark"
echo " Models: EfficientGEBD -> DiffGEBD -> DDM-Net"
echo " Overview Tracker: experiments/step_segment/step_segment_overview.md"
echo "=========================================================="

echo ""
echo "[1/3] Running EfficientGEBD Baseline..."
bash "$DIR/efficient_gebd/iter_01/run_all.sh" "$@"

echo ""
echo "[2/3] Running DiffGEBD Baseline..."
bash "$DIR/diff_gebd/iter_01/run_all.sh" "$@"

echo ""
echo "[3/3] Running DDM-Net Baseline..."
bash "$DIR/ddm_net/iter_01/run_all.sh" "$@"

echo ""
echo "=========================================================="
echo " All 3 Step Segmentation baseline runs completed!"
echo " Check overview tracker:"
echo "   $DIR/step_segment_overview.md"
echo "=========================================================="
