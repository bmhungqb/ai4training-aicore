#!/usr/bin/env python3
"""Sweep DiffGEBD's score threshold (cfg.TEST.THRESHOLD) to balance recall vs precision.

DiffGEBD currently prioritizes recall because the default score threshold
(0.5) keeps too many low-confidence frames as predicted boundaries, inflating
false positives (low precision). The score threshold is the single knob that
trades recall for precision: raising it keeps only higher-confidence
boundaries (higher precision, lower recall); lowering it does the opposite.

This tool re-thresholds the *same* raw model_pred_dict_ep*.pkl (produced once
by `train.py --test-only` / `tools/infer_diffgebd.py`) at many candidate
thresholds -- no re-inference needed -- and reports recall/precision/F1 at
each, so you can pick a balanced operating point instead of guessing.

Usage:
    python tools/sweep_diffgebd_threshold.py \\
        --pred-pkl src/step_segment/DiffGEBD/output/.../model_pred_dict_ep42.pkl \\
        --split val

After picking a threshold, set it as the new default:
  - config YAML: TEST.THRESHOLD: <value>
  - tools/export_diffgebd_predictions.py --threshold <value>
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.export_diffgebd_predictions import (
    load_annotation,
    load_chunked_annotation,
    scores_to_preds_json,
)
from tools.eval_step_segment_predictions import load_ground_truth, evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-pkl", type=Path, required=True, help="model_pred_dict_ep*.pkl from DiffGEBD train.py")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/diff_gebd_dataset"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--merge-eps", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=0.5, help="Matching tolerance window (s) used to rank thresholds")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Candidate score thresholds to evaluate",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    with open(args.pred_pkl, "rb") as f:
        model_pred_dict = pickle.load(f)

    annotation = load_annotation(args.dataset_dir, args.split)
    chunked_annotation = load_chunked_annotation(args.dataset_dir, args.split)
    gt = load_ground_truth(repo_root / "data")

    rows = []
    best = None  # (f1, threshold, recall, precision)
    for th in sorted(args.thresholds):
        preds_json = scores_to_preds_json(model_pred_dict, annotation, chunked_annotation, th, args.merge_eps)
        eval_gt = {k: v for k, v in gt.items() if k in preds_json} or gt
        report = evaluate_predictions(eval_gt, preds_json, thresholds=[args.tolerance], primary_window=args.tolerance)
        pm = report["primary_metrics"]
        recall, precision, f1 = pm["macro_recall"], pm["macro_precision"], pm["macro_f1"]
        rows.append([th, f"{recall*100:.2f}%", f"{precision*100:.2f}%", f"{f1*100:.2f}%"])
        if best is None or f1 > best[0]:
            best = (f1, th, recall, precision)

    print(f"Recall/Precision/F1 @ {args.tolerance}s tolerance, by score threshold:")
    header = ["threshold", "recall", "precision", "f1"]
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(4)]
    for row in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))

    f1, th, recall, precision = best
    print(
        f"\nBest F1-balanced threshold: {th} "
        f"(recall={recall*100:.2f}%, precision={precision*100:.2f}%, f1={f1*100:.2f}%)"
    )
    print(f"Apply it via: TEST.THRESHOLD: {th} in the config YAML, "
          f"and `--threshold {th}` in tools/export_diffgebd_predictions.py / tools/infer_diffgebd.py.")


if __name__ == "__main__":
    main()
