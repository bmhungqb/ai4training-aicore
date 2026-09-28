#!/usr/bin/env python3
"""exp_002_val_overlap_context: Overlapping validation chunking (Revision 2).

Orchestrates, on the EXISTING trained checkpoint (no retraining):
  Step 0   -- Seed-variance noise-floor control: 3x re-inference on the
              UNMODIFIED (val-overlap=0.0) val chunks with different seeds,
              to establish how much Macro F1 swings from pure DDIM/CFG
              sampling stochasticity alone.
  Step 0.5 -- Shifted single-window control: re-chunk val with a single
              non-overlapping grid shifted +2.0s, re-run inference once.
  Main run -- Overlapping val chunks (--val-overlap-seconds 2.5), re-run
              inference once.

Each run regenerates data/diff_gebd_dataset/val_annotation_chunked.pkl (train
split / model weights are never touched), so runs are executed sequentially
and each snapshot is copied into its own output subdirectory before the next
overwrites the shared chunked-annotation file.

This script only prepares data and issues the inference/export commands; the
Human (`torchrun`, GPU-heavy) executes it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[3]


def run(cmd, cwd=REPO_ROOT, env=None):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def regenerate_val_chunks(config, val_overlap_seconds: float, val_offset_seconds: float = 0.0):
    dataset_dir = REPO_ROOT / config["dataset_dir"]
    chunk_params = config["chunk_params"]
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "chunk_diff_gebd_dataset.py"),
        "--dataset-dir", str(dataset_dir),
        "--chunk-seconds", str(chunk_params["chunk_seconds"]),
        "--overlap-seconds", str(chunk_params["overlap_seconds"]),
        "--val-overlap-seconds", str(val_overlap_seconds),
        "--val-offset-seconds", str(val_offset_seconds),
    ]
    run(cmd)


def infer_and_export(config, out_dir: Path, seed: int):
    """Runs DiffGEBD native --test-only inference with the given seed, then
    exports predictions + standardized metrics into out_dir."""
    diff_dir = REPO_ROOT / "src" / "step_segment" / "DiffGEBD"
    config_file = REPO_ROOT / config["diffgebd_config_file"]
    weights = REPO_ROOT / config["checkpoint"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Direct train.py --test-only invocation (not tools/infer_diffgebd.py) so
    # --seed can be controlled per-run for the noise-floor control.
    cmd = [
        "torchrun", "--nproc_per_node", "1", "--master_port", "10213",
        "train.py",
        "--config-file", str(config_file),
        "--resume", str(weights),
        "--test-only",
        "--seed", str(seed),
    ]
    run(cmd, cwd=diff_dir)

    # Locate the freshly-written model_pred_dict_ep-1.pkl (test-only dump)
    output_parent = diff_dir / "output"
    pkl_files = sorted(output_parent.rglob("model_pred_dict_ep-1.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkl_files:
        raise RuntimeError(f"Could not find model_pred_dict_ep-1.pkl under {output_parent}")
    latest_pkl = pkl_files[0]
    # Snapshot the raw pkl into this run's out_dir before the next run overwrites it
    shutil.copy2(latest_pkl, out_dir / "model_pred_dict_ep-1.pkl")

    export_params = config["export_params"]
    export_cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "export_diffgebd_predictions.py"),
        "--pred-pkl", str(out_dir / "model_pred_dict_ep-1.pkl"),
        "--split", "val",
        "--dataset-dir", str(REPO_ROOT / config["dataset_dir"]),
        "--out-dir", str(out_dir),
        "--threshold", str(export_params["threshold"]),
        "--merge-eps", str(export_params["merge_eps"]),
    ]
    run(export_cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--skip-noise-floor", action="store_true", help="Skip Step 0 (already established from a prior run)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    base_out = REPO_ROOT / "outputs" / "step_segment" / "iter_01" / config["experiment_id"]
    base_out.mkdir(parents=True, exist_ok=True)

    # --- Step 0: seed-variance noise-floor control (unmodified val chunks) ---
    if not args.skip_noise_floor:
        print("\n========== Step 0: Seed-variance noise-floor control ==========")
        nf = config["noise_floor_control"]
        regenerate_val_chunks(config, val_overlap_seconds=nf["val_overlap_seconds"])
        for seed in nf["seeds"]:
            out_dir = base_out / f"step0_noise_floor_seed{seed}"
            infer_and_export(config, out_dir, seed=seed)

    # --- Step 0.5: shifted single-window control ---
    print("\n========== Step 0.5: Shifted single-window control ==========")
    sw = config["shifted_window_control"]
    regenerate_val_chunks(config, val_overlap_seconds=sw["val_overlap_seconds"], val_offset_seconds=sw["grid_start_offset_s"])
    out_dir = base_out / "step0_5_shifted_window"
    infer_and_export(config, out_dir, seed=sw["seed"])

    # --- Main run: overlapping val chunks ---
    print("\n========== Main run: overlapping val chunks ==========")
    mr = config["main_run"]
    regenerate_val_chunks(config, val_overlap_seconds=mr["val_overlap_seconds"])
    out_dir = base_out / "main_overlap_run"
    infer_and_export(config, out_dir, seed=mr["seed"])

    # Restore baseline (unmodified) val chunking so the dataset is left clean
    print("\n========== Restoring unmodified (val-overlap=0.0) val chunking ==========")
    regenerate_val_chunks(config, val_overlap_seconds=0.0)

    print(f"\nDone. All run outputs under {base_out}/")
    print("Next: run `@research-evaluator` to compare Macro F1 across step0_noise_floor_seed*, step0_5_shifted_window, and main_overlap_run against the baseline, and check severe_fn_cases_to_track.")


if __name__ == "__main__":
    main()
