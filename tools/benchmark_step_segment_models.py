#!/usr/bin/env python3
"""Benchmark EfficientGEBD vs DDM-Net on the Step Segmentation task.

Both models ultimately produce a set of predicted boundary timestamps (in
seconds) per video. This script compares those predictions against the same
ground truth (`data/**/step_segments.json`) and reports Boundary
Recall/Precision/F1 at multiple tolerance thresholds, plus any manually
supplied resource-usage numbers (VRAM, train/infer speed) for a side-by-side
table.

Prediction file format (one JSON per model), keyed by the same video id used
by `tools/prepare_ddm_dataset.py` / `tools/prepare_efficient_gebd_dataset.py`
(`"<cd_folder>_<chuyen_folder>"`, e.g. `"cd1_chuyen1"`):

    {
      "cd1_chuyen1": [1.23, 5.08, 8.6, ...],
      "cd1_chuyen2": [...],
      ...
    }

Usage:
    python tools/benchmark_step_segment_models.py \\
        --data-dir data \\
        --pred-a efficient_gebd_preds.json --name-a EfficientGEBD \\
        --pred-b ddm_net_preds.json --name-b DDM-Net \\
        --thresholds 0.05 0.1 0.2 0.3 0.4 0.5 \\
        --vram-a-mb 4200 --vram-b-mb 6100 \\
        --train-speed-a "12 min/epoch" --train-speed-b "18 min/epoch" \\
        --infer-fps-a 1215 --infer-fps-b 340
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_unknown(name: str | None) -> bool:
    return not name or not name.strip() or name.strip().upper() in ["UNKNOWN", "NONE"]


def gt_boundaries_for_video(step_segments: dict) -> list[float]:
    """Internal boundaries between consecutive (unknown-trimmed) segments,
    matching tools/prepare_efficient_gebd_dataset.py::boundary_timestamps."""
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


def evaluate(gt: dict[str, list[float]], pred: dict[str, list[float]], thresholds: list[float]) -> dict[float, dict]:
    results = {}
    for th in thresholds:
        tp_all = num_gt_all = num_pred_all = 0
        for vid, gt_bounds in gt.items():
            pred_bounds = pred.get(vid, [])
            tp, num_gt, num_pred = match_f1(gt_bounds, pred_bounds, th)
            tp_all += tp
            num_gt_all += num_gt
            num_pred_all += num_pred
        rec = tp_all / num_gt_all if num_gt_all else 0.0
        prec = tp_all / num_pred_all if num_pred_all else 0.0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
        results[th] = {"recall": rec, "precision": prec, "f1": f1, "tp": tp_all, "num_gt": num_gt_all, "num_pred": num_pred_all}
    return results


def print_table(name: str, results: dict[float, dict]) -> None:
    print(f"\n=== {name} ===")
    print(f"{'Threshold':>10} | {'Recall':>8} | {'Precision':>9} | {'F1':>8} | {'TP/GT/Pred'}")
    print("-" * 65)
    f1s = []
    for th, r in results.items():
        f1s.append(r["f1"])
        print(f"{th:>10.2f} | {r['recall']*100:7.2f}% | {r['precision']*100:8.2f}% | {r['f1']*100:7.2f}% | {r['tp']}/{r['num_gt']}/{r['num_pred']}")
    print("-" * 65)
    print(f"{'Avg F1':>10} | {'':>8} | {'':>9} | {sum(f1s)/len(f1s)*100:7.2f}% |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pred-a", type=Path, required=True)
    parser.add_argument("--pred-b", type=Path, required=True)
    parser.add_argument("--name-a", default="EfficientGEBD")
    parser.add_argument("--name-b", default="DDM-Net")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--vram-a-mb", type=float, default=None)
    parser.add_argument("--vram-b-mb", type=float, default=None)
    parser.add_argument("--train-speed-a", default=None)
    parser.add_argument("--train-speed-b", default=None)
    parser.add_argument("--infer-fps-a", type=float, default=None)
    parser.add_argument("--infer-fps-b", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Save the full comparison as JSON")
    args = parser.parse_args()

    gt = load_ground_truth(args.data_dir)
    pred_a = json.loads(args.pred_a.read_text())
    pred_b = json.loads(args.pred_b.read_text())

    results_a = evaluate(gt, pred_a, args.thresholds)
    results_b = evaluate(gt, pred_b, args.thresholds)

    print(f"Ground truth videos: {len(gt)}")
    print_table(args.name_a, results_a)
    print_table(args.name_b, results_b)

    print(f"\n=== Resource usage ({args.name_a} vs {args.name_b}) ===")
    print(f"{'Metric':<20} | {args.name_a:<20} | {args.name_b:<20}")
    print("-" * 66)
    print(f"{'VRAM (MB)':<20} | {str(args.vram_a_mb):<20} | {str(args.vram_b_mb):<20}")
    print(f"{'Train speed':<20} | {str(args.train_speed_a):<20} | {str(args.train_speed_b):<20}")
    print(f"{'Infer speed (fps)':<20} | {str(args.infer_fps_a):<20} | {str(args.infer_fps_b):<20}")

    if args.out:
        payload = {
            "num_videos": len(gt),
            args.name_a: {"metrics": results_a, "vram_mb": args.vram_a_mb, "train_speed": args.train_speed_a, "infer_fps": args.infer_fps_a},
            args.name_b: {"metrics": results_b, "vram_mb": args.vram_b_mb, "train_speed": args.train_speed_b, "infer_fps": args.infer_fps_b},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved full comparison -> {args.out}")


if __name__ == "__main__":
    main()
