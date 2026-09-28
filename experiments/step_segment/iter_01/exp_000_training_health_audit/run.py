#!/usr/bin/env python3
"""exp_000_training_health_audit: BLOCKING GATE, zero-cost, CPU-only.

Parses outputs/.../train.log for the iter_01 baseline, reconstructs the true
chronological Rel@0.05 F1 curve across all restart segments, and verifies the
saved model_best.pth corresponds to the global-best epoch before Exp 1/2 are
trusted. See tools/audit_diffgebd_training_log.py for the implementation.
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

    out_dir = REPO_ROOT / "outputs" / "step_segment" / "iter_01" / config["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "audit_diffgebd_training_log.py"),
        "--train-log", str(REPO_ROOT / inputs["train_log"]),
        "--pred-dir", str(REPO_ROOT / inputs["pred_dir"]),
        "--out-dir", str(out_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    print(f"\nDone. Report written to {out_dir}/audit_report.json")
    print("If status == FAIL: STOP -- do not trust Exp 1/2 comparisons until baseline is retrained cleanly.")


if __name__ == "__main__":
    main()
