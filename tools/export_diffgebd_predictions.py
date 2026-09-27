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
  2b. If predictions come from the *chunked* dataset (tools/chunk_diff_gebd_dataset.py),
      i.e. `vid` is found in `{split}_annotation_chunked.pkl`, timestamps are
      remapped back into the source video's frame/time reference (using the
      chunk's `source_vid`/`chunk_start_frame`) and boundaries from overlapping
      chunks that fall within `--merge-eps` seconds of each other are merged.
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


def load_chunked_annotation(dataset_dir: Path, split: str) -> dict:
    ann_path = dataset_dir / f"{split}_annotation_chunked.pkl"
    if not ann_path.exists():
        return {}
    with open(ann_path, "rb") as f:
        return pickle.load(f)


def merge_close_boundaries(timestamps: list[float], eps: float = 0.3) -> list[float]:
    """Cluster boundary timestamps produced by overlapping chunks: any two
    boundaries within `eps` seconds of each other (e.g. the same real boundary
    detected in two overlapping chunks) collapse into their average."""
    if not timestamps:
        return []
    ordered = sorted(timestamps)
    clusters = [[ordered[0]]]
    for t in ordered[1:]:
        if t - clusters[-1][-1] <= eps:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return [round(sum(c) / len(c), 3) for c in clusters]


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



def plot_scores(vid, frame_idx, scores, fps, threshold, gt_frame_idx, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = [(i - 1) / fps for i in frame_idx]
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(times, scores, color='C0', linewidth=1, label='score')
    ax.axhline(threshold, color='gray', linestyle=':', label=f'threshold={threshold}')
    
    # Predict boundaries
    boundaries = get_boundary_frame_indices(threshold, frame_idx, scores)
    for b in boundaries:
        ax.axvline((b - 1) / fps, color='red', linestyle='--', alpha=0.8)
        
    if gt_frame_idx:
        for g in gt_frame_idx:
            ax.axvline((g - 1) / fps, color='green', linestyle='-', alpha=0.5)
            
    ax.set_ylabel('score')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('time (s)')
    fig.suptitle(f'{vid} (red dashed = predicted, green = GT)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def render_annotated_video(frame_dir, vlen, fps, pred_frame_idx, gt_frame_idx, out_path, freeze_s=5.0, gt_match_tol_s=0.5):
    import cv2
    from tqdm import tqdm
    # We will try both naming formats
    fmt = "frame{:06d}.jpg"
    if not (frame_dir / fmt.format(1)).exists():
        fmt = "img_{:05d}.jpg"
    if not (frame_dir / fmt.format(1)).exists():
        fmt = "frame{}.jpg"

    first = cv2.imread(str(frame_dir / fmt.format(1)))
    if first is None:
        print(f"Could not read frames in {frame_dir}, skipping video render.")
        return
        
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    freeze_frames = max(1, int(round(freeze_s * fps)))
    gt_match_tol_frames = max(1, int(round(gt_match_tol_s * fps)))
    pred_sorted = sorted(int(p) for p in pred_frame_idx)
    gt_sorted = sorted(int(g) for g in gt_frame_idx) if gt_frame_idx else []
    next_boundary = 0

    for i in tqdm(range(1, int(vlen) + 1), desc=f"rendering {out_path.name}"):
        frame = cv2.imread(str(frame_dir / fmt.format(i)))
        if frame is None:
            continue
        cv2.putText(frame, f"t={(i - 1) / fps:.2f}s", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

        if next_boundary < len(pred_sorted) and i >= pred_sorted[next_boundary]:
            boundary_idx = pred_sorted[next_boundary]
            next_boundary += 1
            freeze_frame = frame.copy()
            cv2.rectangle(freeze_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.putText(freeze_frame, "STEP BOUNDARY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            if gt_sorted:
                matched = any(abs(boundary_idx - g) <= gt_match_tol_frames for g in gt_sorted)
                label = "matches GT" if matched else "no GT match nearby"
                color = (0, 255, 0) if matched else (0, 165, 255)
                cv2.putText(freeze_frame, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            for _ in range(freeze_frames):
                writer.write(freeze_frame)
    writer.release()

def scores_to_preds_json(
    model_pred_dict: dict,
    annotation: dict,
    chunked_annotation: dict,
    threshold: float,
    merge_eps: float = 0.3,
) -> dict[str, list[float]]:
    """Threshold raw per-frame scores into boundary timestamps per source video.

    Pure function (no I/O) so it can be swept over many threshold values, e.g.
    by tools/sweep_diffgebd_threshold.py, without re-reading the pickle/annotation
    for every candidate threshold.
    """
    # video_id -> accumulated global-timestamp boundaries (chunked predictions
    # are mapped back to the source video's frame/time reference and merged;
    # plain (non-chunked) predictions pass straight through).
    boundaries_by_source: dict[str, list[float]] = {}

    for vid, pred in model_pred_dict.items():
        chunk_meta = chunked_annotation.get(vid)
        boundary_frames = get_boundary_frame_indices(threshold, pred["frame_idx"], pred["scores"])

        if chunk_meta is not None:
            # Chunked video: remap local (chunk) frame indices -> global frame
            # indices in the source video, using the offset recorded by
            # tools/chunk_diff_gebd_dataset.py.
            source_vid = chunk_meta["source_vid"]
            chunk_start = int(chunk_meta["chunk_start_frame"])
            fps = float(chunk_meta["fps"])
            global_frames = [chunk_start + f - 1 for f in boundary_frames]
            boundary_ts = [round((f - 1) / fps, 3) for f in global_frames]
            boundaries_by_source.setdefault(source_vid, []).extend(boundary_ts)
        else:
            meta = annotation.get(vid)
            if meta is None:
                continue
            fps = float(meta["fps"])
            # frame_idx is 1-indexed (ffmpeg frame%d.jpg convention), same as
            # tools/prepare_diff_gebd_dataset.py::build_annotation.
            boundary_ts = [round((f - 1) / fps, 3) for f in boundary_frames]
            boundaries_by_source.setdefault(vid, []).extend(boundary_ts)

    preds_json: dict[str, list[float]] = {}
    for source_vid, boundary_ts in boundaries_by_source.items():
        meta = annotation.get(source_vid)
        if meta is None:
            continue
        merged_ts = merge_close_boundaries(boundary_ts, eps=merge_eps)
        preds_json[source_vid] = merged_ts
    return preds_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-pkl", type=Path, required=True, help="model_pred_dict_ep*.pkl from DiffGEBD train.py")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/diff_gebd_dataset"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold, matches cfg.TEST.THRESHOLD")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/diffgebd_preds"))
    parser.add_argument("--viz", action="store_true", help="Render annotated.mp4 and score_curve.png like EfficientGEBD")
    parser.add_argument("--merge-eps", type=float, default=0.35,
                         help="Merge boundaries within this many seconds (handles duplicate detections from overlapping/multi-grid chunks); 0.3-0.4s recommended for 5s chunks with 1.0s overlap")
    args = parser.parse_args()

    with open(args.pred_pkl, "rb") as f:
        model_pred_dict = pickle.load(f)

    annotation = load_annotation(args.dataset_dir, args.split)
    chunked_annotation = load_chunked_annotation(args.dataset_dir, args.split)

    preds_json = scores_to_preds_json(model_pred_dict, annotation, chunked_annotation, args.threshold, args.merge_eps)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for source_vid, merged_ts in preds_json.items():
        meta = annotation.get(source_vid)
        if meta is None:
            print(f"WARNING: source video {source_vid} not found in {args.split}_annotation.pkl, skipping")
            continue
        fps = float(meta["fps"])
        duration = float(meta["video_duration"])

        video_out_dir = args.out_dir / source_vid
        video_out_dir.mkdir(parents=True, exist_ok=True)

        step_segments_pred = build_step_segments_pred(source_vid, merged_ts, duration, fps)
        (video_out_dir / "step_segments_pred.json").write_text(json.dumps(step_segments_pred, indent=2, ensure_ascii=False))

        boundaries_json = {
            "pred_boundaries": {
                "frame_idx": [int(round(t * fps)) + 1 for t in merged_ts],
                "time_s": merged_ts
            }
        }
        
        gt_frame_idx = None
        if "substages_timestamps" in meta and meta["substages_timestamps"]:
            # annotations have shape [num_annotators][num_boundaries]
            # we just use the first annotator's boundaries
            gt_times = meta["substages_timestamps"][0]
            gt_frame_idx = [int(round(t * fps)) + 1 for t in gt_times]
            boundaries_json["gt_boundaries"] = {
                "frame_idx": gt_frame_idx,
                "time_s": gt_times
            }
            
        (video_out_dir / "boundaries.json").write_text(json.dumps(boundaries_json, indent=2))

        if args.viz:
            try:
                # Plot scores (if this is not a chunked prediction where scores are scattered)
                # For simplicity, we only plot if source_vid is in model_pred_dict (meaning not chunked, or we plot the first chunk)
                if source_vid in model_pred_dict:
                    plot_scores(source_vid, model_pred_dict[source_vid]["frame_idx"], model_pred_dict[source_vid]["scores"],
                                fps, args.threshold, gt_frame_idx, video_out_dir / "score_curve.png")
            except Exception as e:
                print(f"Failed to plot score_curve.png for {source_vid}: {e}")
                
            try:
                frame_dir = args.dataset_dir / "images" / args.split / source_vid
                if frame_dir.exists():
                    render_annotated_video(frame_dir, int(meta["video_duration"] * fps), fps,
                                           [int(round(t * fps)) + 1 for t in merged_ts],
                                           gt_frame_idx, video_out_dir / "annotated.mp4")
            except Exception as e:
                print(f"Failed to render annotated.mp4 for {source_vid}: {e}")


    preds_path = args.out_dir / "diffgebd_preds.json"
    preds_path.write_text(json.dumps(preds_json, indent=2))
    
    # Standardized output for AI Research Loop
    std_preds_path = args.out_dir / "predictions.json"
    std_preds_path.write_text(json.dumps(preds_json, indent=2))
    print(f"Wrote {len(preds_json)} video predictions -> {preds_path} and {std_preds_path}")
    print(f"Per-video step_segments_pred.json written under {args.out_dir}/<video_id>/")

    # Run standardized evaluation against ground truth
    try:
        from tools.eval_step_segment_predictions import load_ground_truth, evaluate_predictions
        repo_root = Path(__file__).resolve().parent.parent
        gt = load_ground_truth(repo_root / "data")
        eval_gt = {k: v for k, v in gt.items() if k in preds_json} or gt
        metrics_report = evaluate_predictions(eval_gt, preds_json, thresholds=[0.25, 0.5, 1.0], primary_window=0.5)
        metrics_report["model"] = "DiffGEBD"
        metrics_path = args.out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
        print(f"Standardized metrics report -> {metrics_path}")
        macro_f1 = metrics_report["primary_metrics"].get("macro_f1", 0.0)
        print(f"DiffGEBD Macro F1@0.5s: {macro_f1:.4f}")
    except Exception as e:
        print(f"Notice: Standardized evaluation skipped: {e}")



if __name__ == "__main__":
    main()
