#!/usr/bin/env python3
"""Unified Automated Checkpoint Evaluator for the AI Research Loop.

Takes a trained model checkpoint (e.g. `model_best.pth`), runs inference,
exports standardized predictions, extracts visual error frames, and generates
the complete Evaluator Report (`01_eval_report.md` & `eval_report.json`).

Usage:
    python tools/evaluate_checkpoint.py \\
        --weights src/step_segment/DiffGEBD/output/sewing_diffgebd_resnet50_chunk10s_ann1_dim512_len150/model_best.pth \\
        --track step_segment --iter-id iter_01 --exp-id exp_001_chunk10s

Or simply:
    python tools/evaluate_checkpoint.py --weights <path_to_model_best.pth>
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_config_for_weights(weights_path: Path, config_arg: Optional[Path]) -> Path:
    """Auto-detect config file from weights parent directory or default configs."""
    if config_arg and config_arg.exists():
        return config_arg

    # 1. Check if config.yaml was saved next to weights
    local_cfg = weights_path.parent / "config.yaml"
    if local_cfg.exists():
        return local_cfg

    # 2. Check by filename patterns in src/step_segment/DiffGEBD/config/
    cfg_dir = REPO_ROOT / "src" / "step_segment" / "DiffGEBD" / "config"
    w_str = str(weights_path).lower()
    if "chunk10s" in w_str:
        candidate = cfg_dir / "sewing_diffgebd_resnet50_chunk10s.yaml"
        if candidate.exists():
            return candidate
    elif "chunk5s" in w_str:
        candidate = cfg_dir / "sewing_diffgebd_resnet50_chunk5s.yaml"
        if candidate.exists():
            return candidate
    elif "chunked" in w_str:
        candidate = cfg_dir / "sewing_diffgebd_resnet50_chunked.yaml"
        if candidate.exists():
            return candidate

    default_cfg = cfg_dir / "sewing_diffgebd_resnet50_chunk10s.yaml"
    if default_cfg.exists():
        return default_cfg

    raise FileNotFoundError(f"Could not auto-detect config file for {weights_path}. Please pass --config explicitly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="Path to checkpoint (e.g. model_best.pth)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML (auto-detected if omitted)")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--track", type=str, default="step_segment", help="Research track name")
    parser.add_argument("--iter-id", type=str, default="iter_01", help="Iteration ID (e.g. iter_01)")
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment ID (auto-deduced if omitted)")
    parser.add_argument("--top-k-errors", type=int, default=4, help="Number of severe error cases to extract")
    parser.add_argument("--threshold", type=float, default=None, help="Score threshold (default: uses config TEST.THRESHOLD)")
    parser.add_argument("--viz", action="store_true", default=True, help="Extract video error frames and visualizations")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    config_path = resolve_config_for_weights(weights_path, args.config)
    print(f"✓ Target weights: {weights_path}")
    print(f"✓ Resolved config: {config_path}")

    # Determine experiment ID
    exp_id = args.exp_id
    if not exp_id:
        exp_id = weights_path.parent.name
        if exp_id.startswith("output"):
            exp_id = exp_id.replace("output/", "").replace("output\\", "")
        if not exp_id:
            exp_id = "exp_eval"

    out_dir = REPO_ROOT / "outputs" / args.track / args.iter_id / exp_id
    report_dir = REPO_ROOT / "experiments" / args.track / args.iter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f" [1/3] Running Native Inference via tools/infer_diffgebd.py...")
    print("=" * 65)
    infer_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "infer_diffgebd.py"),
        "--config-file", str(config_path),
        "--weights", str(weights_path),
        "--split", args.split,
        "--out-dir", str(out_dir),
    ]
    subprocess.run(infer_cmd, cwd=str(REPO_ROOT), check=True)

    # Copy metrics.txt / train log if available
    train_log_src = weights_path.parent / "metrics.txt"
    if train_log_src.exists():
        shutil.copyfile(train_log_src, out_dir / "train.log")
        print(f"✓ Copied training log from {train_log_src} -> {out_dir / 'train.log'}")
    else:
        # Check infer.log
        if (out_dir / "infer.log").exists() and not (out_dir / "train.log").exists():
            shutil.copyfile(out_dir / "infer.log", out_dir / "train.log")

    # If predictions are saved as diffgebd_preds.json, ensure predictions.json exists
    if (out_dir / "diffgebd_preds.json").exists() and not (out_dir / "predictions.json").exists():
        shutil.copyfile(out_dir / "diffgebd_preds.json", out_dir / "predictions.json")

    print("\n" + "=" * 65)
    print(f" [2/3] Running Comprehensive Evaluation & Visual Error Inspector...")
    print("=" * 65)
    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "eval_and_inspect_errors.py"),
        "--output-dir", str(out_dir),
        "--report-dir", str(report_dir),
        "--track", args.track,
        "--iter-id", args.iter_id,
        "--exp-id", exp_id,
        "--top-k-errors", str(args.top_k_errors),
    ]
    if args.viz:
        eval_cmd.append("--viz")
    subprocess.run(eval_cmd, cwd=str(REPO_ROOT), check=True)

    print("\n" + "=" * 65)
    print(" [3/3] EVALUATION COMPLETE!")
    print(f" - Report Markdown:     {report_dir / '01_eval_report.md'}")
    print(f" - Report JSON:         {report_dir / 'eval_report.json'}")
    print(f" - Visual Error Frames: {out_dir / 'error_cases'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
