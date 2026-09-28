#!/usr/bin/env python3
"""exp_004_deterministic_noise_seeding (iter_02, NEW -- corrective intervention).

Unlike Experiments 1-3 (diagnostic-only), this is a **fix**: it removes the
noise source (ddim_sample()'s global-RNG-stream-order-dependent initial
noise draw) rather than characterizing its size.

Code change (already applied, single isolated code path):
  - src/step_segment/DiffGEBD/modeling/diffusion_model.py::ddim_sample() now
    accepts an optional `sample_seeds` list; when provided, each batch row's
    initial noise is drawn from its own seeded torch.Generator instead of the
    global RNG stream.
  - src/step_segment/DiffGEBD/train.py::validate_end_to_end() computes
    per-sample seeds (hash(video_id, chunk_index)) and passes them through
    IFF the new config flag `DIFFUSION.DETERMINISTIC_SAMPLE_SEEDING` is True
    (default False -- legacy behavior is completely unaffected unless this
    flag is explicitly set).

This script only prepares data and issues the inference/export commands with
`DIFFUSION.DETERMINISTIC_SAMPLE_SEEDING True` set; the Human (`torchrun`,
GPU-heavy) executes it.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[3]

# Reuse regenerate_val_chunks() from iter_01's exp_002 run.py (unmodified logic).
_ITER01_RUN_PY = REPO_ROOT / "experiments" / "step_segment" / "iter_01" / "exp_002_val_overlap_context" / "run.py"
_spec = importlib.util.spec_from_file_location("iter01_exp002_run", _ITER01_RUN_PY)
_iter01_exp002 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_iter01_exp002)
regenerate_val_chunks = _iter01_exp002.regenerate_val_chunks


def run(cmd, cwd=REPO_ROOT):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def infer_and_export_with_overrides(config, out_dir: Path, seed: int, overrides: list[str]):
    """Same as iter_01/exp_002's infer_and_export(), but additionally passes
    config-override key/value pairs (e.g. DIFFUSION.DETERMINISTIC_SAMPLE_SEEDING
    True) through to train.py --test-only via yacs CLI override syntax."""
    diff_dir = REPO_ROOT / "src" / "step_segment" / "DiffGEBD"
    config_file = REPO_ROOT / config["diffgebd_config_file"]
    weights = REPO_ROOT / config["checkpoint"]

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun", "--nproc_per_node", "1", "--master_port", "10214",
        "train.py",
        "--config-file", str(config_file),
        "--resume", str(weights),
        "--test-only",
        "--seed", str(seed),
    ] + overrides
    run(cmd, cwd=diff_dir)

    output_parent = diff_dir / "output"
    pkl_files = sorted(output_parent.rglob("model_pred_dict_ep-1.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkl_files:
        raise RuntimeError(f"Could not find model_pred_dict_ep-1.pkl under {output_parent}")
    latest_pkl = pkl_files[0]
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
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    base_out = REPO_ROOT / "outputs" / "step_segment" / "iter_02" / config["experiment_id"]
    base_out.mkdir(parents=True, exist_ok=True)

    overrides = config["config_overrides"]

    # --- Verification 1: baseline chunking (overlap=0.0), 2 different global seeds ---
    vb = config["verification_baseline_repeat"]
    print(f"\n========== Verification 1: baseline (overlap={vb['val_overlap_seconds']}) repeat, "
          f"DETERMINISTIC_SAMPLE_SEEDING=True ==========")
    regenerate_val_chunks(config, val_overlap_seconds=vb["val_overlap_seconds"])
    for seed in vb["seeds"]:
        out_dir = base_out / f"verif1_baseline_seed{seed}"
        infer_and_export_with_overrides(config, out_dir, seed=seed, overrides=overrides)

    # --- Verification 2 (spot-check): overlap=2.5 config ---
    vo = config["verification_overlap_spotcheck"]
    print(f"\n========== Verification 2: overlap-config spot-check (overlap={vo['val_overlap_seconds']}) "
          f"==========")
    regenerate_val_chunks(config, val_overlap_seconds=vo["val_overlap_seconds"])
    out_dir = base_out / "verif2_overlap_spotcheck"
    infer_and_export_with_overrides(config, out_dir, seed=vo["seed"], overrides=overrides)

    # Restore baseline (unmodified) val chunking so the dataset is left clean
    print("\n========== Restoring unmodified (val-overlap=0.0) val chunking ==========")
    regenerate_val_chunks(config, val_overlap_seconds=0.0)

    print(f"\nDone. All run outputs under {base_out}/")
    print("Next: compare Macro F1@0.5s between verif1_baseline_seed42 and verif1_baseline_seed123 -- "
          "with DETERMINISTIC_SAMPLE_SEEDING=True, these should now be identical or near-identical "
          "(down to floating-point determinism limits), collapsing the previously-measured 0.0355 "
          "noise-floor spread to ~0. Also inspect verif2_overlap_spotcheck to confirm the per-sample "
          "noise draw for a given video_id+chunk_index pair is stable regardless of chunk count.")


if __name__ == "__main__":
    main()
