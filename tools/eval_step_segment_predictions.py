#!/usr/bin/env python3
"""Standardized evaluation tool for Step Segmentation models in the AI Research Loop.

Computes precision, recall, and F1 at standardized tolerance windows
(default: 0.25s, 0.5s, 1.0s) against ground truth step_segments.json.
Exports metrics formatted for the Evaluator Agent (eval_report_schema.json).
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple


def is_unknown(name: str | None) -> bool:
    return not name or not name.strip() or name.strip().upper() in ["UNKNOWN", "NONE"]


def gt_boundaries_for_video(step_segments: dict) -> list[float]:
    """Internal boundaries between consecutive (unknown-trimmed) segments,
    matching tools/prepare_efficient_gebd_dataset.py and tools/benchmark_step_segment_models.py."""
    segs = step_segments.get("segments", [])
    if not segs:
        return []
    start_idx, end_idx = 0, len(segs)
    while start_idx < len(segs) and is_unknown(segs[start_idx].get("operation_name", "")):
        start_idx += 1
    while end_idx > start_idx and is_unknown(segs[end_idx - 1].get("operation_name", "")):
        end_idx -= 1
    segs = segs[start_idx:end_idx]
    bounds = []
    for i in range(1, len(segs)):
        prev_end = float(segs[i - 1]["end_time_s"])
        cur_start = float(segs[i]["start_time_s"])
        bounds.append(round((prev_end + cur_start) / 2.0, 3))
    return bounds


def load_ground_truth(data_dir: Path) -> dict[str, list[float]]:
    gt = {}
    for f in sorted(data_dir.rglob("step_segments.json")):
        video_id = f"{f.parent.parent.name}_{f.parent.name}"
        gt[video_id] = gt_boundaries_for_video(json.loads(f.read_text()))
    return gt


def match_f1(gt: list[float], pred: list[float], threshold: float) -> tuple[int, int, int]:
    """Returns (tp, num_gt, num_pred) using greedy nearest-neighbour matching."""
    remaining_pred = list(pred)
    tp = 0
    for g in gt:
        if not remaining_pred:
            break
        closest = min(remaining_pred, key=lambda p: abs(p - g))
        if abs(closest - g) <= threshold:
            tp += 1
            remaining_pred.remove(closest)
    return tp, len(gt), len(pred)


def evaluate_predictions(
    gt: dict[str, list[float]],
    pred: dict[str, list[float]],
    thresholds: list[float] = (0.25, 0.5, 1.0),
    primary_window: float = 0.5,
) -> dict:
    """Evaluates boundary predictions across thresholds and returns a structured dictionary."""
    results_by_th = {}
    for th in thresholds:
        tp_all = num_gt_all = num_pred_all = 0
        per_video_th = {}
        for vid, gt_bounds in gt.items():
            pred_bounds = pred.get(vid, [])
            tp, num_gt, num_pred = match_f1(gt_bounds, pred_bounds, th)
            rec = tp / num_gt if num_gt else 0.0
            prec = tp / num_pred if num_pred else 0.0
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
            per_video_th[vid] = {
                "tp": tp,
                "num_gt": num_gt,
                "num_pred": num_pred,
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
            }
            tp_all += tp
            num_gt_all += num_gt
            num_pred_all += num_pred

        overall_rec = tp_all / num_gt_all if num_gt_all else 0.0
        overall_prec = tp_all / num_pred_all if num_pred_all else 0.0
        overall_f1 = (
            2 * overall_rec * overall_prec / (overall_rec + overall_prec)
            if (overall_rec + overall_prec)
            else 0.0
        )
        results_by_th[str(th)] = {
            "recall": round(overall_rec, 4),
            "precision": round(overall_prec, 4),
            "f1": round(overall_f1, 4),
            "tp": tp_all,
            "num_gt": num_gt_all,
            "num_pred": num_pred_all,
            "per_video": per_video_th,
        }

    # Format primary metrics for Evaluator schema
    primary_metrics = {}
    for th in thresholds:
        th_key = f"{th}".replace(".", "_")
        primary_metrics[f"f1_at_{th_key}s"] = results_by_th[str(th)]["f1"]
        primary_metrics[f"recall_at_{th_key}s"] = results_by_th[str(th)]["recall"]
        primary_metrics[f"precision_at_{th_key}s"] = results_by_th[str(th)]["precision"]

    # Main reference metrics (e.g. at primary_window)
    pw_str = str(primary_window)
    primary_metrics["macro_f1"] = results_by_th.get(pw_str, {}).get("f1", 0.0)
    primary_metrics["macro_recall"] = results_by_th.get(pw_str, {}).get("recall", 0.0)
    primary_metrics["macro_precision"] = results_by_th.get(pw_str, {}).get("precision", 0.0)

    # Identify notable error cases
    error_cases = []
    pw_video_stats = results_by_th.get(pw_str, {}).get("per_video", {})
    for vid, stats in pw_video_stats.items():
        if stats["num_pred"] > stats["num_gt"] * 1.8:
            error_cases.append({
                "case_id": f"{vid}_oversegmentation",
                "category": "false_positive",
                "description": f"Video {vid} over-segmented: {stats['num_pred']} preds vs {stats['num_gt']} GT boundaries.",
                "evidence_reference": f"per_video.{vid}",
                "metrics_impact": f"Precision dropped to {stats['precision'] * 100:.1f}%",
            })
        elif stats["recall"] < 0.5 and stats["num_gt"] > 0:
            error_cases.append({
                "case_id": f"{vid}_missed_boundaries",
                "category": "false_negative",
                "description": f"Video {vid} missed over 50% boundaries: recall={stats['recall'] * 100:.1f}%.",
                "evidence_reference": f"per_video.{vid}",
                "metrics_impact": f"Low recall {stats['recall'] * 100:.1f}%",
            })

    report = {
        "task": "step_segmentation",
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "primary_window": primary_window,
        "tolerance_windows": thresholds,
        "primary_metrics": primary_metrics,
        "by_threshold": {th: {k: v for k, v in data.items() if k != "per_video"} for th, data in results_by_th.items()},
        "per_video_performance": pw_video_stats,
        "error_cases": error_cases,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate step segmentation predictions against GT.")
    parser.add_argument("--pred", type=Path, required=True, help="Path to predictions JSON {video_id: [timestamps]}")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data/ containing step_segments.json")
    parser.add_argument("--out", type=Path, default=None, help="Path to save metrics.json")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--primary-window", type=float, default=0.5)
    args = parser.parse_args()

    gt = load_ground_truth(args.data_dir)
    pred = json.loads(args.pred.read_text(encoding="utf-8"))

    # If predictions are nested or keyed differently, check matching
    eval_gt = {k: v for k, v in gt.items() if k in pred}
    if not eval_gt:
        # Try evaluating on all gt
        eval_gt = gt

    report = evaluate_predictions(eval_gt, pred, thresholds=args.thresholds, primary_window=args.primary_window)

    print("\n" + "=" * 65)
    print(" STEP SEGMENTATION BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Tolerance (s)':>15} | {'Recall':>10} | {'Precision':>11} | {'F1 Score':>10} | {'TP/GT/Pred'}")
    print("-" * 65)
    for th in args.thresholds:
        r = report["by_threshold"][str(th)]
        print(f"{th:>15.2f}s | {r['recall']*100:9.2f}% | {r['precision']*100:10.2f}% | {r['f1']*100:9.2f}% | {r['tp']}/{r['num_gt']}/{r['num_pred']}")
    print("=" * 65)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved metrics report -> {args.out}")


if __name__ == "__main__":
    main()
