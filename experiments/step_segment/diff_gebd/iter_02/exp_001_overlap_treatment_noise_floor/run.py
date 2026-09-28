#!/usr/bin/env python3
"""exp_001_overlap_treatment_noise_floor (iter_02, Revision 2).

Establishes a 3-seed noise floor for the val-overlap=2.5 TREATMENT condition
(Hypothesis 1: Stochastic DDIM Sampling Noise Dominates Small Effect Sizes),
on the EXISTING trained checkpoint (no retraining):

  - Reuses iter_01's exp_002 seed=42 main_overlap_run result (Macro F1=0.2975)
    as the 1st data point (copied by reference, NOT re-run).
  - Runs 2 additional seeds (123, 2024) against the SAME val-overlap=2.5
    chunking, using the exact `regenerate_val_chunks()` / `infer_and_export()`
    orchestration already implemented in iter_01's exp_002 run.py (imported,
    not re-implemented).

Per Revision 2 (post-debate correction), this experiment does NOT claim the
overlap-treatment 3-seed sample is seed-matched/paired against the baseline
(overlap=0.0) 3-seed sample -- the two are independent/unpaired distributions.

Execution order (per 04_debate_verdict.md WARN): run this AFTER
exp_002_premerge_boundary_audit (zero-cost, CPU-only, no dependency on this
experiment's result, but its output is a precondition for correctly
interpreting this experiment's noise-band comparison).

This script only prepares data and issues the inference/export commands; the
Human (`torchrun`, GPU-heavy) executes it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import yaml

def find_repo_root(p: Path) -> Path:
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return p.resolve().parents[4]

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(EXP_DIR)

# Import (not copy) regenerate_val_chunks() / infer_and_export() from iter_01's
# exp_002_val_overlap_context/run.py, per the plan's "no modification to their
# logic" constraint.
_ITER01_RUN_PY = REPO_ROOT / "experiments" / "step_segment" / "diff_gebd" / "iter_01" / "exp_002_val_overlap_context" / "run.py"
_spec = importlib.util.spec_from_file_location("iter01_exp002_run", _ITER01_RUN_PY)
_iter01_exp002 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_iter01_exp002)
regenerate_val_chunks = _iter01_exp002.regenerate_val_chunks
infer_and_export = _iter01_exp002.infer_and_export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    base_out = REPO_ROOT / "outputs" / "step_segment" / "diff_gebd" / "iter_02" / config["experiment_id"]
    base_out.mkdir(parents=True, exist_ok=True)

    ot = config["overlap_treatment"]

    # --- Copy-by-reference: existing seed=42 main_overlap_run result ---
    seed42_src = REPO_ROOT / "outputs" / "step_segment" / "diff_gebd" / "iter_01" / "exp_002_val_overlap_context" / "main_overlap_run"
    seed42_dst = base_out / "overlap_seed42_reference_copy"
    if seed42_src.exists() and not seed42_dst.exists():
        shutil.copytree(seed42_src, seed42_dst)
        print(f"Copied existing seed=42 reference run: {seed42_src} -> {seed42_dst}")
    elif not seed42_src.exists():
        print(f"WARNING: expected existing seed=42 reference run not found at {seed42_src}. "
              f"Re-verify iter_01/exp_002_val_overlap_context outputs before trusting this experiment's report.")

    # --- New seeds against the SAME val-overlap=2.5 chunking ---
    print(f"\n========== Regenerating val chunks: val-overlap-seconds={ot['val_overlap_seconds']} ==========")
    regenerate_val_chunks(config, val_overlap_seconds=ot["val_overlap_seconds"])

    for seed in ot["new_seeds"]:
        print(f"\n========== Overlap-treatment run: seed={seed} ==========")
        out_dir = base_out / f"overlap_seed{seed}"
        infer_and_export(config, out_dir, seed=seed)

    # Restore baseline (unmodified) val chunking so the dataset is left clean
    print("\n========== Restoring unmodified (val-overlap=0.0) val chunking ==========")
    regenerate_val_chunks(config, val_overlap_seconds=0.0)

    print(f"\nDone. All run outputs under {base_out}/")
    print("Next: compare Macro F1@0.5s across overlap_seed42_reference_copy, overlap_seed123, "
          "overlap_seed2024 (3-seed overlap-treatment noise band) against the baseline "
          "(overlap=0.0) 3-seed noise band [0.3435, 0.3790] from iter_01/exp_002. "
          "State explicitly whether the two bands overlap (treated as independent/unpaired samples).")


if __name__ == "__main__":
    main()
