#!/usr/bin/env python3
"""exp_002_premerge_boundary_audit (iter_02).

Read-only, zero-inference diagnostic for Hypothesis 2 (Overlap+Merge
Interaction Nets Fewer Surviving Predictions). Invokes
`tools/audit_diffgebd_premerge_candidates.py` against the already-saved
`model_pred_dict_ep-1.pkl` / `predictions.json` from iter_01's
`exp_002_val_overlap_context/main_overlap_run/` -- no re-inference, no GPU.

Per the Debate verdict's stated execution order, run this BEFORE
`exp_001_overlap_treatment_noise_floor` -- its result is a precondition for
correctly interpreting Exp 1's noise-band comparison.
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
    base_out = REPO_ROOT / "outputs" / "step_segment" / "iter_02" / config["experiment_id"]
    base_out.mkdir(parents=True, exist_ok=True)

    ap = config["audit_params"]
    pred_pkl = REPO_ROOT / config["pred_pkl"]
    post_merge_preds = REPO_ROOT / config["post_merge_preds"]
    dataset_dir = REPO_ROOT / config["dataset_dir"]

    if not pred_pkl.exists() or not post_merge_preds.exists():
        raise FileNotFoundError(
            f"Expected pre-existing artifacts from iter_01/exp_002_val_overlap_context not found: "
            f"{pred_pkl} / {post_merge_preds}. Verify iter_01's outputs are still on disk before running this audit."
        )

    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "audit_diffgebd_premerge_candidates.py"),
        "--pred-pkl", str(pred_pkl),
        "--post-merge-preds", str(post_merge_preds),
        "--dataset-dir", str(dataset_dir),
        "--split", config["split"],
        "--threshold", str(ap["threshold"]),
        "--merge-eps", str(ap["merge_eps"]),
        "--gt-tolerance", str(ap["gt_tolerance"]),
        "--worker-pair", *ap["worker_pair"],
        "--out-dir", str(base_out),
    ]
    run(cmd)

    print(f"\nDone. Report written under {base_out}/premerge_audit_report.json")
    print("Next: inspect 'dataset_wide_summary' and 'worker_pair_breakdown' fields to determine "
          "whether merge_close_boundaries is discarding higher-confidence TP-adjacent candidates "
          "(Hypothesis 2 supported) or just collapsing legitimately-duplicated detections "
          "(Hypothesis 2 weakened). This result should be reviewed BEFORE interpreting "
          "exp_001_overlap_treatment_noise_floor's noise-band comparison.")


if __name__ == "__main__":
    main()
