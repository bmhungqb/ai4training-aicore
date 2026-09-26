#!/bin/bash
# Batch Runner for DiffGEBD iter_01 Experiments
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=========================================================="
echo " Starting DiffGEBD iter_01 baseline experiment"
echo "=========================================================="

python3 "$DIR/exp_001_baseline/run.py" --mode train "$@"

echo "=========================================================="
echo " DiffGEBD iter_01 baseline experiment completed!"
echo "=========================================================="
