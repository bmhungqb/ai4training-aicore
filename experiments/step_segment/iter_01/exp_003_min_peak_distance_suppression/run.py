#!/usr/bin/env python3
"""exp_003_min_peak_distance_suppression (Revision 2).

Step 0: run the existing tools/sweep_diffgebd_threshold.py as a zero-new-code
        control (Occam's Razor) -- no GPU, pure post-processing.
Main:   Leave-One-Video-Out (LOVO) cross-validation tuning of
        --min-peak-distance via tools/sweep_diffgebd_min_peak_distance_lovo.py
        -- no GPU, pure post-processing on the same raw score pickle.

No re-inference anywhere in this experiment (CPU-only, ~3-5 min total).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[3]


def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    pred_pkl = REPO_ROOT / config["pred_pkl"]
    dataset_dir = REPO_ROOT / config["dataset_dir"]
    base_out = REPO_ROOT / "outputs" / "step_segment" / "iter_01" / config["experiment_id"]
    base_out.mkdir(parents=True, exist_ok=True)

    # --- Step 0: simple-threshold-sweep control (existing tool) ---
    print("\n========== Step 0: Simple threshold-sweep control ==========")
    ts = config["threshold_sweep_control"]
    step0_out = base_out / "step0_threshold_sweep_control"
    step0_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "sweep_diffgebd_threshold.py"),
        "--pred-pkl", str(pred_pkl),
        "--dataset-dir", str(dataset_dir),
        "--split", "val",
        "--merge-eps", str(ts["merge_eps"]),
        "--tolerance", str(ts["tolerance"]),
        "--thresholds", *[str(t) for t in ts["thresholds"]],
    ]
    # Tee stdout to a log file since sweep_diffgebd_threshold.py only prints
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
    print(result.stdout)
    (step0_out / "threshold_sweep_log.txt").write_text(result.stdout)

    # --- Main: LOVO-tuned min-peak-distance suppression ---
    print("\n========== Main: LOVO min-peak-distance tuning ==========")
    lovo = config["lovo_tuning"]
    lovo_out = base_out / "lovo_min_peak_distance"
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "sweep_diffgebd_min_peak_distance_lovo.py"),
        "--pred-pkl", str(pred_pkl),
        "--dataset-dir", str(dataset_dir),
        "--split", "val",
        "--threshold", str(lovo["base_threshold"]),
        "--tolerance", str(lovo["tolerance"]),
        "--candidates", *[str(c) for c in lovo["candidates_seconds"]],
        "--out-dir", str(lovo_out),
    ]
    run(cmd)

    print(f"\nDone. Outputs under {base_out}/")
    print("Compare step0_threshold_sweep_control/threshold_sweep_log.txt (simple baseline) against "
          "lovo_min_peak_distance/lovo_report.json (new suppression code). Only adopt the suppression "
          "code if it Pareto-dominates the simple threshold sweep at matched recall.")
    print(f"Also check lovo_min_peak_distance/lovo_report.json's 'worker_station_leakage_check_cd12' field: "
          f"if cd12_chuyen1 precision improves but cd12_chuyen2 recall regresses, treat as overfitting, not a fix.")


if __name__ == "__main__":
    main()
