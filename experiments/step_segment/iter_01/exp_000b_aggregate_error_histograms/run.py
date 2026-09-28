#!/usr/bin/env python3
"""exp_000b_aggregate_error_histograms: BLOCKING GATE, zero-cost, CPU-only.

Builds dataset-wide FN chunk-relative-position and FP nearest-prediction-
distance histograms from the existing baseline predictions.json, to confirm
the anecdotal (n=2/n=2) failure patterns generalize before Exp 1/2 are
trusted. See tools/analyze_diffgebd_error_distributions.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    inputs = config["inputs"]
    eval_params = config["eval_params"]

    out_dir = REPO_ROOT / "outputs" / "step_segment" / "iter_01" / config["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "analyze_diffgebd_error_distributions.py"),
        "--pred", str(REPO_ROOT / inputs["predictions"]),
        "--chunked-annotation", str(REPO_ROOT / inputs["chunked_annotation"]),
        "--data-dir", str(REPO_ROOT / inputs["data_dir"]),
        "--out-dir", str(out_dir),
        "--tolerance", str(eval_params["primary_window"]),
        "--near-chunk-start-s", str(eval_params["near_chunk_start_s"]),
        "--chunk-seconds", str(eval_params["chunk_seconds"]),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    print(f"\nDone. Report written to {out_dir}/error_distribution_report.json")
    print("Check 'verdict' field: if exp_002/exp_003 says DE-PRIORITIZE, do not proceed with that paid experiment.")


if __name__ == "__main__":
    main()
