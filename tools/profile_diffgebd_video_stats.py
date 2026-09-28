#!/usr/bin/env python3
"""Per-video motion-energy / brightness-contrast profiling (exp_003_session_confound_profile).

Descriptive, non-inferential case-comparison for Hypothesis 3 (Session-Level
Heterogeneity Dominates). CPU-only (PIL + numpy), no GPU, no optical-flow
dependency. Reads each val video's UNTOUCHED raw source frames at
`data/diff_gebd_dataset/images/val/<video_id>/frame*.jpg` -- verified on disk,
NOT the non-existent `outputs/.../<video_id>/` path, and NOT the stateful,
repeatedly-regenerated `<video_id>__chunk<i>/` symlink directories.

Computes, over ALL frames per video (no subsampling, per Debater WARN
correction):
  1. Motion-energy proxy: mean absolute inter-frame pixel difference
     (grayscale, normalized to [0,1]).
  2. Brightness: mean luminance across all frames.
  3. Contrast: mean per-frame pixel standard deviation across all frames.

Outputs a single CSV/JSON table: video_id, motion_energy, brightness,
contrast, f1_at_0_5s (joined from an existing eval_report.json).

Revision 2 statistical framing: this script does NOT compute or report a
Pearson correlation coefficient (n=9 is not statistically meaningful for
population-level inference). It presents the 3 stats as a plain descriptive
table plus an explicit named-pair comparison, both labeled hypothesis-
generating / non-inferential.

Usage:
    python tools/profile_diffgebd_video_stats.py \\
        --dataset-dir data/diff_gebd_dataset \\
        --split val \\
        --eval-report experiments/step_segment/iter_02/eval_report.json \\
        --named-pair cd12_chuyen1 cd12_chuyen2 \\
        --out-dir outputs/step_segment/iter_02/exp_003_session_confound_profile
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _frame_paths_sorted(video_dir: Path) -> list[Path]:
    frames = list(video_dir.glob("frame*.jpg"))

    def _idx(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    frames.sort(key=_idx)
    return frames


def profile_video(video_dir: Path) -> dict:
    """Single sequential pass over all frames: running sums for motion
    energy (consecutive abs-diff), brightness, and contrast (per-frame std).
    Uses numpy for per-frame pixel ops (grayscale arrays), since a pure-Python
    per-pixel loop is infeasible at ~2500 frames x multi-megapixel frames per
    video across 9 videos."""
    frame_paths = _frame_paths_sorted(video_dir)
    if not frame_paths:
        return {"num_frames": 0, "motion_energy": None, "brightness": None, "contrast": None}

    diff_sum = 0.0
    diff_count = 0
    brightness_sum = 0.0
    contrast_sum = 0.0
    prev_arr = None

    for p in frame_paths:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0

        brightness_sum += float(arr.mean())
        contrast_sum += float(arr.std())

        if prev_arr is not None:
            diff_sum += float(np.abs(arr - prev_arr).mean())
            diff_count += 1
        prev_arr = arr

    num_frames = len(frame_paths)
    return {
        "num_frames": num_frames,
        "motion_energy": round(diff_sum / diff_count, 6) if diff_count else None,
        "brightness": round(brightness_sum / num_frames, 6),
        "contrast": round(contrast_sum / num_frames, 6),
    }


def load_f1_lookup(metrics_json_path: Path) -> dict:
    """Reads the standardized `metrics.json`'s `per_video_performance` dict
    (schema: {video_id: {..., "f1": ...}}), e.g.
    outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/metrics.json
    (the baseline referenced by 01_eval_report.md's per-video F1@0.5s table)."""
    if not metrics_json_path.exists():
        return {}
    data = json.loads(metrics_json_path.read_text())
    per_video = data.get("per_video_performance", {})
    return {vid: v.get("f1") for vid, v in per_video.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/diff_gebd_dataset"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--metrics-json", type=Path,
                         default=Path("outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/metrics.json"),
                         help="Baseline metrics.json with 'per_video_performance' (f1 @0.5s) to join alongside the 3 stats")
    parser.add_argument("--named-pair", nargs=2, default=["cd12_chuyen1", "cd12_chuyen2"],
                         help="Two video IDs for the explicit descriptive case-comparison")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    images_dir = args.dataset_dir / "images" / args.split
    # Only untouched, non-chunked source video directories (exclude "__chunk" symlinked dirs)
    video_dirs = sorted(
        d for d in images_dir.iterdir()
        if d.is_dir() and "__chunk" not in d.name
    )

    f1_lookup = load_f1_lookup(args.metrics_json)

    rows = []
    for video_dir in video_dirs:
        vid = video_dir.name
        stats = profile_video(video_dir)
        stats["video_id"] = vid
        stats["f1_at_0_5s"] = f1_lookup.get(vid)
        rows.append(stats)
        print(f"{vid}: num_frames={stats['num_frames']}, motion_energy={stats['motion_energy']}, "
              f"brightness={stats['brightness']}, contrast={stats['contrast']}, f1_at_0_5s={stats['f1_at_0_5s']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = args.out_dir / "video_stats_profile.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # CSV output
    csv_path = args.out_dir / "video_stats_profile.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "num_frames", "motion_energy", "brightness", "contrast", "f1_at_0_5s"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote {json_path} and {csv_path}")

    # --- Descriptive case-comparison (NOT a correlation / inferential test, n=9) ---
    named_pair = args.named_pair
    pair_rows = {r["video_id"]: r for r in rows if r["video_id"] in named_pair}
    rest_rows = [r for r in rows if r["video_id"] not in named_pair]

    comparison = {}
    for stat_key in ["motion_energy", "brightness", "contrast"]:
        rest_vals = [r[stat_key] for r in rest_rows if r[stat_key] is not None]
        rest_range = (min(rest_vals), max(rest_vals)) if rest_vals else (None, None)
        pair_stats = {vid: pair_rows[vid][stat_key] for vid in named_pair if vid in pair_rows}
        outside_range = {
            vid: (val is not None and rest_range[0] is not None and not (rest_range[0] <= val <= rest_range[1]))
            for vid, val in pair_stats.items()
        }
        comparison[stat_key] = {
            "rest_of_dataset_range": rest_range,
            "named_pair_values": pair_stats,
            "named_pair_outside_rest_range": outside_range,
        }

    comparison_report = {
        "note": "Descriptive, non-inferential case comparison (n=9). NOT a Pearson correlation or "
                "population-level inference. Hypothesis-generating observation only.",
        "named_pair": named_pair,
        "comparison": comparison,
    }
    comparison_path = args.out_dir / "named_pair_comparison.json"
    comparison_path.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
    print(f"Wrote descriptive named-pair comparison -> {comparison_path}")

    for stat_key, c in comparison.items():
        flags = [vid for vid, out in c["named_pair_outside_rest_range"].items() if out]
        if flags:
            print(f"NOTE: {', '.join(flags)} falls OUTSIDE the rest-of-dataset range on '{stat_key}' "
                  f"(range={c['rest_of_dataset_range']}, values={c['named_pair_values']}) "
                  f"-- candidate confound, NOT proof of causation.")


if __name__ == "__main__":
    main()
