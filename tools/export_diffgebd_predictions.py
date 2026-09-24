#!/usr/bin/env python3
"""Convert DiffGEBD prediction pickles into project-standard outputs.

DiffGEBD's `train.py::validate_end_to_end` dumps, once per epoch,
`<output_dir>/model_pred_dict_ep<N>.pkl`:

    {
      "<video_id>": {"frame_idx": [1, 5, 9, ...], "scores": [0.01, 0.92, ...]},
      ...
    }

This script:
  1. Thresholds the per-frame boundary scores into discrete boundary frame
     indices (same peak-grouping logic as
     `src/step_segment/DiffGEBD/utils/eval.py::get_idx_from_score_by_threshold`).
  2. Converts frame indices -> timestamps (seconds) using the `fps` recorded in
     `data/diff_gebd_dataset/{train,val}_annotation.pkl`.
  3. Writes:
       - `<out-dir>/diffgebd_preds.json`: {video_id: [boundary_timestamps]},
         directly usable by `tools/benchmark_step_segment_models.py --pred-a/-b`.
       - `<out-dir>/<video_id>/step_segments_pred.json`: placeholder-named
         segments between consecutive predicted boundaries, matching the
         `step_segments.json` schema used across this project.

Usage:
    python tools/export_diffgebd_predictions.py \\
        --pred-pkl src/step_segment/DiffGEBD/output/sewing_diffgebd_resnet50_.../model_pred_dict_ep42.pkl \\
        --split val \\
        --out-dir experiments/diffgebd_preds

    # then compare against another model's predictions:
    python tools/benchmark_step_segment_models.py \\
        --pred-a experiments/diffgebd_preds/diffgebd_preds.json --name-a DiffGEBD \\
        --pred-b efficient_gebd_preds.json --name-b EfficientGEBD
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def get_boundary_frame_indices(threshold: float, frame_idx: list[int], scores: list[float]) -> list[int]:
    """Group consecutive above-threshold frames and return the center frame
    index of each group. Mirrors DiffGEBD/utils/eval.py::get_idx_from_score_by_threshold
    without any torch dependency."""
    groups: list[list[int]] = []
    current: list[int] = []
    n = len(scores)
    for i in range(n):
        if scores[i] >= threshold:
            current.append(i)
        elif current:
            groups.append(current)
            current = []
        if i == n - 1 and current:
            groups.append(current)

    boundaries = []
    for group in groups:
        center = round(sum(group) / len(group))
        boundaries.append(frame_idx[center])
    return boundaries


def load_annotation(dataset_dir: Path, split: str) -> dict:
    ann_path = dataset_dir / f"{split}_annotation.pkl"
    with open(ann_path, "rb") as f:
        return pickle.load(f)


def build_step_segments_pred(vid: str, boundaries_s: list[float], duration: float, fps: float) -> dict:
    boundaries_s = sorted(t for t in boundaries_s if 0.0 < t < duration)
    edges = [0.0] + boundaries_s + [duration]
    segments = []
    for i in range(len(edges) - 1):
        segments.append({
            "operation_name": f"SEGMENT_{i}",
            "start_time_s": round(edges[i], 3),
            "end_time_s": round(edges[i + 1], 3),
        })
    return {
        "video_id": vid,
        "fps": fps,
        "n_segments": len(segments),
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-pkl", type=Path, required=True, help="model_pred_dict_ep*.pkl from DiffGEBD train.py")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/diff_gebd_dataset"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold, matches cfg.TEST.THRESHOLD")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/diffgebd_preds"))
    args = parser.parse_args()

    with open(args.pred_pkl, "rb") as f:
        model_pred_dict = pickle.load(f)

    annotation = load_annotation(args.dataset_dir, args.split)

    preds_json: dict[str, list[float]] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for vid, pred in model_pred_dict.items():
        meta = annotation.get(vid)
        if meta is None:
            print(f"WARNING: {vid} not found in {args.split}_annotation.pkl, skipping")
            continue
        fps = float(meta["fps"])
        duration = float(meta["video_duration"])

        boundary_frames = get_boundary_frame_indices(args.threshold, pred["frame_idx"], pred["scores"])
        # frame_idx is 1-indexed (ffmpeg frame%d.jpg convention), same as
        # tools/prepare_diff_gebd_dataset.py::build_annotation.
        boundary_ts = [round((f - 1) / fps, 3) for f in boundary_frames]
        preds_json[vid] = boundary_ts

        video_out_dir = args.out_dir / vid
        video_out_dir.mkdir(parents=True, exist_ok=True)
        step_segments_pred = build_step_segments_pred(vid, boundary_ts, duration, fps)
        (video_out_dir / "step_segments_pred.json").write_text(json.dumps(step_segments_pred, indent=2, ensure_ascii=False))

    preds_path = args.out_dir / "diffgebd_preds.json"
    preds_path.write_text(json.dumps(preds_json, indent=2))
    print(f"Wrote {len(preds_json)} video predictions -> {preds_path}")
    print(f"Per-video step_segments_pred.json written under {args.out_dir}/<video_id>/")


if __name__ == "__main__":
    main()
