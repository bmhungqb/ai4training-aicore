#!/bin/bash
# Batch Runner for DDM-Net iter_01 Experiments
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=========================================================="
echo " Starting DDM-Net iter_01 baseline experiment"
echo "=========================================================="

python3 "$DIR/exp_001_baseline/run.py" --mode train "$@"

echo "=========================================================="
echo " DDM-Net iter_01 baseline experiment completed!"
echo "=========================================================="
