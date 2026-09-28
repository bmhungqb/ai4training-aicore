#!/usr/bin/env python3
"""exp_000b_aggregate_error_histograms: dataset-wide FN/FP distribution probe.

Read-only diagnostic (no GPU, no re-inference). Consumes the existing
baseline `predictions.json` + GT (`data/.../step_segments.json`) plus
`val_annotation_chunked.pkl` (for chunk boundaries) to build, across ALL val
videos (not just the 2 anecdotal cases in 02_diagnosis.md):

  (a) Histogram of FN discrepancy distances bucketed by chunk-relative
      position (first 1.5s of chunk vs. rest of chunk).
  (b) Histogram of FP-to-nearest-other-prediction distances.

Blocking-gate verdict:
  - If FN rate is NOT materially higher near chunk-starts dataset-wide ->
    Hypothesis 1 / Experiment 1 de-prioritized.
  - If FP-to-nearest-prediction distances are NOT concentrated in the
    sub-1s range dataset-wide -> Hypothesis 2 / Experiment 2 de-prioritized.

Usage:
    python tools/analyze_diffgebd_error_distributions.py \\
        --pred outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/predictions.json \\
        --chunked-annotation data/diff_gebd_dataset/val_annotation_chunked.pkl \\
        --out-dir experiments/step_segment/iter_01/exp_000b_aggregate_error_histograms \\
        --tolerance 0.5 \\
        --near-chunk-start-s 1.5
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

from tools.eval_step_segment_predictions import load_ground_truth


def nearest_chunk_relative_position(t: float, source_vid: str, chunked_annotation: dict, chunk_seconds: float = 5.0) -> float | None:
    """Approximate the position of timestamp `t` (in source-video time) within
    its nearest non-overlapping val chunk (val chunking uses a single fixed
    grid, offset 0, chunk_seconds long). Returns seconds-from-chunk-start, or
    None if the source video has no chunk metadata (unchunked prediction)."""
    found = False
    for chunk_meta in chunked_annotation.values():
        if chunk_meta.get("source_vid") == source_vid:
            found = True
            break
    if not found:
        return None
    # val chunks are a fixed, non-overlapping grid starting at 0: chunk index = floor(t / chunk_seconds)
    return t - (int(t // chunk_seconds) * chunk_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="Baseline predictions.json {video_id: [timestamps]}")
    parser.add_argument("--chunked-annotation", type=Path, default=Path("data/diff_gebd_dataset/val_annotation_chunked.pkl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.5, help="Matching tolerance window (s), matches primary eval window")
    parser.add_argument("--near-chunk-start-s", type=float, default=1.5, help="Bucket boundary for 'near chunk-start' (seconds)")
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_all = load_ground_truth(args.data_dir)
    preds = json.loads(args.pred.read_text())
    # Restrict GT to only the videos actually present in predictions.json (the
    # val split evaluated by this baseline), matching
    # export_diffgebd_predictions.py's own eval_gt filtering -- otherwise FN
    # counts get inflated by train-split videos that were never run through
    # inference at all.
    gt = {k: v for k, v in gt_all.items() if k in preds} or gt_all

    chunked_annotation = {}
    if args.chunked_annotation.exists():
        with open(args.chunked_annotation, "rb") as f:
            chunked_annotation = pickle.load(f)

    fn_records = []  # {video, gt_t, discrepancy, chunk_relative_pos, near_start}
    fp_records = []  # {video, fp_t, nearest_other_pred_dist}

    all_fn_near_start = 0
    all_fn_total = 0
    all_fp_sub1s = 0
    all_fp_total = 0

    for vid, gt_bounds in gt.items():
        pred_bounds = sorted(preds.get(vid, []))
        remaining_pred = list(pred_bounds)

        # FN pass: greedy nearest-neighbour matching (mirrors eval_step_segment_predictions.match_f1)
        for g in gt_bounds:
            if remaining_pred:
                closest = min(remaining_pred, key=lambda p: abs(p - g))
                if abs(closest - g) <= args.tolerance:
                    remaining_pred.remove(closest)
                    continue
            # Unmatched GT boundary -> False Negative
            discrepancy = min((abs(p - g) for p in pred_bounds), default=None)
            chunk_pos = nearest_chunk_relative_position(g, vid, chunked_annotation, args.chunk_seconds)
            near_start = chunk_pos is not None and chunk_pos < args.near_chunk_start_s
            fn_records.append({
                "video": vid, "gt_t": g, "discrepancy_to_nearest_pred": discrepancy,
                "chunk_relative_pos_s": chunk_pos, "near_chunk_start": near_start,
            })
            all_fn_total += 1
            if near_start:
                all_fn_near_start += 1

        # FP pass: predictions that never matched any GT boundary within tolerance
        matched_pred = set()
        remaining_pred2 = list(pred_bounds)
        for g in gt_bounds:
            if remaining_pred2:
                closest = min(remaining_pred2, key=lambda p: abs(p - g))
                if abs(closest - g) <= args.tolerance:
                    matched_pred.add(closest)
                    remaining_pred2.remove(closest)
        for p in pred_bounds:
            if p in matched_pred:
                continue
            others = [o for o in pred_bounds if o != p]
            nearest_other = min((abs(p - o) for o in others), default=None)
            fp_records.append({"video": vid, "fp_t": p, "nearest_other_pred_dist": nearest_other})
            all_fp_total += 1
            if nearest_other is not None and nearest_other < 1.0:
                all_fp_sub1s += 1

    fn_near_start_frac = all_fn_near_start / all_fn_total if all_fn_total else 0.0
    fp_sub1s_frac = all_fp_sub1s / all_fp_total if all_fp_total else 0.0

    # Verdict thresholds: "materially higher" = near-chunk-start FN fraction
    # exceeds the naive expected fraction if FNs were uniformly distributed
    # across a chunk_seconds-long chunk (near_chunk_start_s / chunk_seconds).
    expected_uniform_frac = args.near_chunk_start_s / args.chunk_seconds
    h1_generalizes = fn_near_start_frac > expected_uniform_frac * 1.5  # >=50% relative excess
    h2_generalizes = fp_sub1s_frac >= 0.5  # majority of FPs are sub-1s from another prediction

    report = {
        "status": "PASS",
        "fn_total": all_fn_total,
        "fn_near_chunk_start": all_fn_near_start,
        "fn_near_chunk_start_fraction": round(fn_near_start_frac, 4),
        "expected_uniform_near_start_fraction": round(expected_uniform_frac, 4),
        "hypothesis_1_generalizes": h1_generalizes,
        "fp_total": all_fp_total,
        "fp_sub_1s_from_nearest": all_fp_sub1s,
        "fp_sub_1s_fraction": round(fp_sub1s_frac, 4),
        "hypothesis_2_generalizes": h2_generalizes,
        "verdict": {
            "exp_002_val_overlap_context": "PROCEED" if h1_generalizes else "DE-PRIORITIZE (FN rate not materially higher near chunk-starts dataset-wide)",
            "exp_003_min_peak_distance_suppression": "PROCEED" if h2_generalizes else "DE-PRIORITIZE (FP-to-nearest-prediction distances not concentrated sub-1s dataset-wide)",
        },
        "fn_records": fn_records,
        "fp_records": fp_records,
    }
    (args.out_dir / "error_distribution_report.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fn_positions = [r["chunk_relative_pos_s"] for r in fn_records if r["chunk_relative_pos_s"] is not None]
        axes[0].hist(fn_positions, bins=20, color="C3")
        axes[0].axvline(args.near_chunk_start_s, color="k", linestyle="--", label=f"near-start cutoff ({args.near_chunk_start_s}s)")
        axes[0].set_title(f"FN chunk-relative position (n={len(fn_positions)})")
        axes[0].set_xlabel("seconds from chunk start")
        axes[0].legend()

        fp_dists = [r["nearest_other_pred_dist"] for r in fp_records if r["nearest_other_pred_dist"] is not None]
        axes[1].hist(fp_dists, bins=20, color="C1")
        axes[1].axvline(1.0, color="k", linestyle="--", label="1.0s cutoff")
        axes[1].set_title(f"FP distance to nearest other prediction (n={len(fp_dists)})")
        axes[1].set_xlabel("seconds")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(args.out_dir / "error_histograms.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(json.dumps({k: v for k, v in report.items() if k not in ("fn_records", "fp_records")}, indent=2))
    print(f"\nFull report -> {args.out_dir / 'error_distribution_report.json'}")


if __name__ == "__main__":
    main()
