#!/usr/bin/env python3
"""Pre-merge vs. post-merge boundary candidate audit (exp_002_premerge_boundary_audit).

Read-only, zero-inference diagnostic for Hypothesis 2 (Overlap+Merge
Interaction Nets Fewer Surviving Predictions). Loads an already-saved
`model_pred_dict_ep-1.pkl` (e.g. from the existing
`outputs/step_segment/iter_01/exp_002_val_overlap_context/main_overlap_run/`
directory) and reconstructs the PRE-merge `boundaries_by_source` dict using
the same logic as `tools/export_diffgebd_predictions.py::scores_to_preds_json`,
up to but NOT including the `merge_close_boundaries()` call.

Reports, dataset-wide across all val videos:
  1. Total pre-merge candidate count vs. total post-merge candidate count
     (aggregate reduction %), not just 2-3 examples.
  2. For each merged cluster with >1 pre-merge candidate: whether the
     retained/averaged timestamp is within `--gt-tolerance` seconds of a GT
     boundary (legitimately-duplicated TP-adjacent detection) vs. not
     (spurious candidate averaged into a wrong location, or a real candidate
     merged away from its correct GT-adjacent timestamp).
  3. Per-video breakdown, with `cd12_chuyen1` / `cd12_chuyen2` (or any
     `--worker-pair` videos) always reported explicitly (per Debater's
     standing per-worker verification requirement).

No re-inference, no GPU. Pure post-hoc analysis on existing artifacts.

Usage:
    python tools/audit_diffgebd_premerge_candidates.py \\
        --pred-pkl outputs/step_segment/iter_01/exp_002_val_overlap_context/main_overlap_run/model_pred_dict_ep-1.pkl \\
        --post-merge-preds outputs/step_segment/iter_01/exp_002_val_overlap_context/main_overlap_run/predictions.json \\
        --dataset-dir data/diff_gebd_dataset \\
        --split val \\
        --out-dir outputs/step_segment/iter_02/exp_002_premerge_boundary_audit
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
    get_boundary_frame_indices_with_scores,
    load_annotation,
    load_chunked_annotation,
    merge_close_boundaries,
)


def build_premerge_boundaries_by_source(model_pred_dict, annotation, chunked_annotation, threshold):
    """Same remapping logic as scores_to_preds_json(), stopping BEFORE the
    merge_close_boundaries() / suppress_min_peak_distance() call."""
    boundaries_by_source: dict[str, list[tuple[float, float]]] = {}
    for vid, pred in model_pred_dict.items():
        chunk_meta = chunked_annotation.get(vid)
        boundary_frames_scores = get_boundary_frame_indices_with_scores(threshold, pred["frame_idx"], pred["scores"])

        if chunk_meta is not None:
            source_vid = chunk_meta["source_vid"]
            chunk_start = int(chunk_meta["chunk_start_frame"])
            fps = float(chunk_meta["fps"])
            for f, score in boundary_frames_scores:
                global_frame = chunk_start + f - 1
                ts = round((global_frame - 1) / fps, 3)
                boundaries_by_source.setdefault(source_vid, []).append((ts, score))
        else:
            meta = annotation.get(vid)
            if meta is None:
                continue
            fps = float(meta["fps"])
            for f, score in boundary_frames_scores:
                ts = round((f - 1) / fps, 3)
                boundaries_by_source.setdefault(vid, []).append((ts, score))
    return boundaries_by_source


def cluster_premerge(timestamps: list[float], eps: float) -> list[list[float]]:
    """Same clustering as merge_close_boundaries(), but returns the raw
    cluster membership (list of lists) instead of collapsing to averages."""
    if not timestamps:
        return []
    ordered = sorted(timestamps)
    clusters = [[ordered[0]]]
    for t in ordered[1:]:
        if t - clusters[-1][-1] <= eps:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return clusters


def nearest_gt_distance(t: float, gt_times: list[float]) -> float | None:
    if not gt_times:
        return None
    return min(abs(t - g) for g in gt_times)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-pkl", type=Path, required=True, help="model_pred_dict_ep*.pkl (already saved on disk)")
    parser.add_argument("--post-merge-preds", type=Path, required=True, help="predictions.json (already saved, post-merge) for the same run")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/diff_gebd_dataset"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--merge-eps", type=float, default=0.35)
    parser.add_argument("--gt-tolerance", type=float, default=0.5, help="Seconds; matches the project's primary eval window")
    parser.add_argument("--worker-pair", nargs="*", default=["cd12_chuyen1", "cd12_chuyen2"],
                         help="Video IDs to always report individually (standing per-worker verification requirement)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.pred_pkl, "rb") as f:
        model_pred_dict = pickle.load(f)
    annotation = load_annotation(args.dataset_dir, args.split)
    chunked_annotation = load_chunked_annotation(args.dataset_dir, args.split)
    post_merge_preds = json.loads(args.post_merge_preds.read_text())

    premerge_by_source = build_premerge_boundaries_by_source(model_pred_dict, annotation, chunked_annotation, args.threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_video_report = {}
    total_premerge = 0
    total_postmerge = 0
    total_multi_candidate_clusters_gt_adjacent = 0
    total_multi_candidate_clusters_spurious = 0

    for source_vid, ts_scores in premerge_by_source.items():
        meta = annotation.get(source_vid)
        if meta is None:
            continue
        gt_times = meta.get("substages_timestamps", [[]])[0] if meta.get("substages_timestamps") else []

        premerge_ts = sorted(t for t, _ in ts_scores)
        clusters = cluster_premerge(premerge_ts, eps=args.merge_eps)
        postmerge_ts = post_merge_preds.get(source_vid, [])

        multi_clusters = [c for c in clusters if len(c) > 1]
        gt_adjacent_multi = 0
        spurious_multi = 0
        cluster_details = []
        for c in multi_clusters:
            retained_ts = round(sum(c) / len(c), 3)
            dist = nearest_gt_distance(retained_ts, gt_times)
            is_gt_adjacent = dist is not None and dist <= args.gt_tolerance
            if is_gt_adjacent:
                gt_adjacent_multi += 1
            else:
                spurious_multi += 1
            cluster_details.append({
                "cluster_raw_timestamps": c,
                "retained_averaged_timestamp": retained_ts,
                "nearest_gt_distance_s": round(dist, 3) if dist is not None else None,
                "classified_as": "gt_adjacent_legit_duplicate" if is_gt_adjacent else "spurious_or_merged_away_from_gt",
            })

        per_video_report[source_vid] = {
            "premerge_candidate_count": len(premerge_ts),
            "postmerge_candidate_count": len(postmerge_ts),
            "num_multi_candidate_clusters": len(multi_clusters),
            "num_multi_candidate_clusters_gt_adjacent": gt_adjacent_multi,
            "num_multi_candidate_clusters_spurious": spurious_multi,
            "cluster_details": cluster_details,
        }

        total_premerge += len(premerge_ts)
        total_postmerge += len(postmerge_ts)
        total_multi_candidate_clusters_gt_adjacent += gt_adjacent_multi
        total_multi_candidate_clusters_spurious += spurious_multi

    reduction_pct = round(100.0 * (1 - total_postmerge / total_premerge), 2) if total_premerge else None

    worker_pair_breakdown = {vid: per_video_report[vid] for vid in args.worker_pair if vid in per_video_report}

    report = {
        "dataset_wide_summary": {
            "total_premerge_candidates": total_premerge,
            "total_postmerge_candidates": total_postmerge,
            "aggregate_reduction_pct": reduction_pct,
            "total_multi_candidate_clusters_gt_adjacent": total_multi_candidate_clusters_gt_adjacent,
            "total_multi_candidate_clusters_spurious": total_multi_candidate_clusters_spurious,
        },
        "worker_pair_breakdown": worker_pair_breakdown,
        "per_video": per_video_report,
        "params": {
            "threshold": args.threshold,
            "merge_eps": args.merge_eps,
            "gt_tolerance": args.gt_tolerance,
            "pred_pkl": str(args.pred_pkl),
            "post_merge_preds": str(args.post_merge_preds),
        },
    }

    out_path = args.out_dir / "premerge_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote pre-merge/post-merge audit report -> {out_path}")
    print(f"Total pre-merge candidates: {total_premerge}, post-merge: {total_postmerge} "
          f"(reduction: {reduction_pct}%)")
    print(f"Multi-candidate clusters: {total_multi_candidate_clusters_gt_adjacent} GT-adjacent "
          f"(legit duplicates) vs. {total_multi_candidate_clusters_spurious} spurious/merged-away-from-GT")
    print(f"Worker-pair breakdown ({', '.join(args.worker_pair)}):")
    for vid, v in worker_pair_breakdown.items():
        print(f"  {vid}: premerge={v['premerge_candidate_count']}, postmerge={v['postmerge_candidate_count']}, "
              f"gt_adjacent_clusters={v['num_multi_candidate_clusters_gt_adjacent']}, "
              f"spurious_clusters={v['num_multi_candidate_clusters_spurious']}")


if __name__ == "__main__":
    main()
