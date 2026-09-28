#!/usr/bin/env python3
"""exp_003_min_peak_distance_suppression: Leave-One-Video-Out (LOVO) tuning.

Re-thresholds the SAME raw model_pred_dict_ep*.pkl scores (no re-inference)
with an added greedy min-peak-distance suppression step
(tools/export_diffgebd_predictions.py::suppress_min_peak_distance), and tunes
`--min-peak-distance` via Leave-One-Video-Out cross-validation instead of
directly sweeping against the full reported val set (avoids tuning-on-test
leakage, per Debater critique / 03_research_plan.md Exp 2).

For each of the 9 val videos, selects the best `min-peak-distance` (by F1)
using the OTHER 8 videos, applies it to the held-out video, and aggregates
the resulting held-out-only F1 across all 9 folds.

Usage:
    python tools/sweep_diffgebd_min_peak_distance_lovo.py \\
        --pred-pkl src/step_segment/DiffGEBD/output/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/model_pred_dict_ep-1.pkl \\
        --split val \\
        --out-dir experiments/step_segment/iter_01/exp_003_min_peak_distance_suppression
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_diffgebd_predictions import (
    load_annotation,
    load_chunked_annotation,
    scores_to_preds_json,
)
from tools.eval_step_segment_predictions import load_ground_truth, evaluate_predictions


def f1_for_video_subset(gt: dict, preds: dict, video_ids: list[str], tolerance: float) -> float:
    subset_gt = {v: gt[v] for v in video_ids if v in gt}
    if not subset_gt:
        return 0.0
    report = evaluate_predictions(subset_gt, preds, thresholds=[tolerance], primary_window=tolerance)
    return report["primary_metrics"]["macro_f1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-pkl", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/diff_gebd_dataset"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5, help="Base score threshold, matches cfg.TEST.THRESHOLD")
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--candidates", type=float, nargs="+", default=[0.5, 1.0, 1.5],
                         help="Candidate --min-peak-distance values (seconds) to tune over per LOVO fold")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = REPO_ROOT

    with open(args.pred_pkl, "rb") as f:
        model_pred_dict = pickle.load(f)

    annotation = load_annotation(args.dataset_dir, args.split)
    chunked_annotation = load_chunked_annotation(args.dataset_dir, args.split)
    gt = load_ground_truth(repo_root / "data")
    gt = {k: v for k, v in gt.items() if k in {c.get("source_vid", vid) for vid, c in chunked_annotation.items()} or k in annotation}

    all_video_ids = sorted(gt.keys())
    print(f"LOVO over {len(all_video_ids)} videos: {all_video_ids}")

    # Precompute preds_json for every candidate min-peak-distance ONCE (cheap, CPU-only).
    preds_by_candidate = {
        c: scores_to_preds_json(model_pred_dict, annotation, chunked_annotation, args.threshold, min_peak_distance=c)
        for c in args.candidates
    }

    fold_results = []
    held_out_preds_final = {}
    for held_out in all_video_ids:
        train_videos = [v for v in all_video_ids if v != held_out]
        # Select best candidate using the OTHER 8 videos only
        best_candidate, best_train_f1 = None, -1.0
        for c in args.candidates:
            f1 = f1_for_video_subset(gt, preds_by_candidate[c], train_videos, args.tolerance)
            if f1 > best_train_f1:
                best_train_f1 = f1
                best_candidate = c
        # Apply selected candidate to the held-out video only
        held_out_f1 = f1_for_video_subset(gt, preds_by_candidate[best_candidate], [held_out], args.tolerance)
        held_out_preds_final[held_out] = preds_by_candidate[best_candidate].get(held_out, [])
        fold_results.append({
            "held_out_video": held_out,
            "selected_min_peak_distance": best_candidate,
            "train_fold_f1": round(best_train_f1, 4),
            "held_out_f1": round(held_out_f1, 4),
        })

    # Aggregate held-out-only F1 across all 9 folds (each video evaluated only
    # under the min-peak-distance value selected without seeing it).
    lovo_aggregate_f1 = f1_for_video_subset(gt, held_out_preds_final, all_video_ids, args.tolerance)

    # Baseline comparison: threshold-only (Step 0 control, min_peak_distance=0)
    baseline_preds = scores_to_preds_json(model_pred_dict, annotation, chunked_annotation, args.threshold, min_peak_distance=0.0)
    baseline_f1 = f1_for_video_subset(gt, baseline_preds, all_video_ids, args.tolerance)

    # Worker/station-leakage joint check: cd12_chuyen1 + cd12_chuyen2 must BOTH
    # be reported together per Debater's overfitting-risk requirement.
    worker_pair = ["cd12_chuyen1", "cd12_chuyen2"]
    worker_pair_detail = {}
    for vid in worker_pair:
        if vid in gt:
            report = evaluate_predictions({vid: gt[vid]}, {vid: held_out_preds_final.get(vid, [])}, thresholds=[args.tolerance], primary_window=args.tolerance)
            worker_pair_detail[vid] = report["primary_metrics"]

    report_out = {
        "caveat": "LOVO cross-validation used in place of a true disjoint held-out test set (9 val videos too few for a 3-way split). This must be stated in the eventual results report per Debater's requirement.",
        "candidates_tested": args.candidates,
        "baseline_f1_no_suppression": round(baseline_f1, 4),
        "lovo_aggregate_f1": round(lovo_aggregate_f1, 4),
        "improvement": round(lovo_aggregate_f1 - baseline_f1, 4),
        "fold_results": fold_results,
        "worker_station_leakage_check_cd12": worker_pair_detail,
        "note_on_worker_leakage": "If cd12_chuyen1 precision improves but cd12_chuyen2 recall regresses vs baseline, this indicates overfitting to one session's noise profile and the change should NOT be adopted despite aggregate F1 looking better.",
    }
    (args.out_dir / "lovo_report.json").write_text(json.dumps(report_out, indent=2))
    (args.out_dir / "predictions.json").write_text(json.dumps(held_out_preds_final, indent=2))

    print(json.dumps(report_out, indent=2))
    print(f"\nFull report -> {args.out_dir / 'lovo_report.json'}")


if __name__ == "__main__":
    main()
