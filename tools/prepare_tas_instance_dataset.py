#!/usr/bin/env python3
"""Restructure `dataset_tas_instance/` into the **instance-level** TAS format
described in `src/TAS-instance-style/problem_definition.md` (section 17).

Unlike `tools/prepare_tas_dataset.py` (which only writes coarse frame-wise
`groundTruth/*.txt`, silently merging consecutive same-category fine-grained
operations into a single run), this script keeps **every fine-grained
operation segment from `step_segments_clean.json` as its own instance**, even
when two consecutive segments map to the same coarse category. This preserves
exactly the information the problem definition requires: two adjacent
"Sewing #1" / "Sewing #2" instances must never collapse into one.

For each video, writes:

    annotations/<video_id>.json
        {
          "video_id", "fps", "num_frames", "duration",
          "instances": [ {id, start_frame, end_frame, start_time, end_time,
                           class_id, class_name}, ... ],
          "boundaries": [ {frame, type: same_class|class_change,
                            left_instance, right_instance}, ... ]
        }

and regenerates the legacy derived views from these instances (single source
of truth) so they stay consistent:

    groundTruth/<video_id>.txt   -- one coarse class name per frame
    boundaries/<video_id>.txt    -- 0/1 per frame, 1 at every instance start
                                     (same-class AND class-change), not just
                                     class-change frames.

Also writes:

    classes.json                          -- [{class_id, class_name}, ...]
    splits/{train,val,test}.txt           -- plain video_id lists (problem_definition.md §17.1)
    splits/{train,test}.split1.bundle     -- kept for BaFormer-style loaders (`<video_id>.txt` per line)

The video's own frame count is taken from the existing `groundTruth/*.txt`
line count (already consistent with the extracted `features/*.npy`), so no
ffprobe call on the source videos is needed.

Usage:
    python tools/prepare_tas_instance_dataset.py
    python tools/prepare_tas_instance_dataset.py --out-dir src/TAS-instance-style/dataset_tas_instance
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_prepare_tas_dataset():
    spec = importlib.util.spec_from_file_location(
        "prepare_tas_dataset", REPO_ROOT / "tools" / "prepare_tas_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_base = _load_prepare_tas_dataset()
CATEGORIES = _base.CATEGORIES  # ["background", "Sewing/Joining", ...]
classify = _base.classify
video_unique_id = _base.video_unique_id
find_videos = _base.find_videos

CLASS_TO_ID = {name: i for i, name in enumerate(CATEGORIES)}


def build_instances(step_segments: dict, n_frames: int, fps: float) -> list[dict]:
    """Convert fine-grained `step_segments_clean.json` segments into
    instance-level (start_frame, end_frame, class_id, class_name) records,
    filling any uncovered gaps (head/tail or between segments) with explicit
    `background` instances. Consecutive segments are NEVER merged, even if
    they map to the same coarse category -- that is the whole point.
    """
    segs = sorted(step_segments.get("segments", []), key=lambda s: s["start_time_s"])

    instances: list[dict] = []
    cursor = 0  # next uncovered frame

    def add_instance(start_frame: int, end_frame: int, class_name: str) -> None:
        if end_frame <= start_frame:
            return
        instances.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": round(start_frame / fps, 3),
                "end_time": round(end_frame / fps, 3),
                "class_id": CLASS_TO_ID[class_name],
                "class_name": class_name,
            }
        )

    for seg in segs:
        start_f = max(0, round(float(seg["start_time_s"]) * fps))
        end_f = min(n_frames, round(float(seg["end_time_s"]) * fps) + 1)  # inclusive end -> exclusive
        start_f = max(start_f, cursor)
        if end_f <= start_f:
            continue  # fully swallowed by previous segment / out of range

        # fill gap before this segment with background
        if start_f > cursor:
            add_instance(cursor, start_f, "background")

        class_name = classify(seg.get("operation_name"))
        add_instance(start_f, end_f, class_name)
        cursor = end_f

    # trailing gap
    if cursor < n_frames:
        add_instance(cursor, n_frames, "background")

    # assign sequential ids, ordered with "id" first
    ordered = []
    for i, inst in enumerate(instances):
        ordered.append(
            {
                "id": i,
                "start_frame": inst["start_frame"],
                "end_frame": inst["end_frame"],
                "start_time": inst["start_time"],
                "end_time": inst["end_time"],
                "class_id": inst["class_id"],
                "class_name": inst["class_name"],
            }
        )
    return ordered


def build_boundaries(instances: list[dict]) -> list[dict]:
    """One entry per internal transition between consecutive instances
    (problem_definition.md §17.4). The very first instance's start (frame 0)
    is not listed as a "boundary" since there's no left neighbor.
    """
    boundaries = []
    for left, right in zip(instances, instances[1:]):
        btype = "same_class" if left["class_id"] == right["class_id"] else "class_change"
        boundaries.append(
            {
                "frame": right["start_frame"],
                "type": btype,
                "left_instance": left["id"],
                "right_instance": right["id"],
            }
        )
    return boundaries


def instances_to_frame_labels(instances: list[dict], n_frames: int) -> list[str]:
    labels = ["background"] * n_frames
    for inst in instances:
        for i in range(inst["start_frame"], inst["end_frame"]):
            labels[i] = inst["class_name"]
    return labels


def instances_to_frame_boundaries(instances: list[dict], n_frames: int) -> list[int]:
    """0/1 per frame, 1 at the start of every instance (same-class AND
    class-change) -- unlike a naive class-change-only detector, this is a
    true instance-boundary signal."""
    boundary = [0] * n_frames
    for inst in instances:
        s = inst["start_frame"]
        if 0 <= s < n_frames:
            boundary[s] = 1
    return boundary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "src" / "TAS-instance-style" / "dataset_tas_instance"
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    ann_dir = out_dir / "annotations"
    gt_dir = out_dir / "groundTruth"
    boundary_dir = out_dir / "boundaries"
    splits_dir = out_dir / "splits"
    for d in (ann_dir, gt_dir, boundary_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    # reuse the existing train/val split (by video_id, independent of frame content)
    def read_bundle(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [line.strip()[:-4] for line in path.read_text().splitlines() if line.strip()]

    train_ids = set(read_bundle(out_dir / "splits" / "train.bundle"))
    val_ids = set(read_bundle(out_dir / "splits" / "val.bundle"))

    video_dirs = find_videos(args.data_dir)
    n_written = 0
    n_skipped = []

    for video_dir in video_dirs:
        vid = video_unique_id(video_dir)
        seg_path = video_dir / "step_segments_clean.json"
        gt_legacy_path = out_dir / "groundTruth" / f"{vid}.txt"
        if not seg_path.exists():
            continue
        if vid not in train_ids and vid not in val_ids:
            n_skipped.append(vid)
            continue
        if not gt_legacy_path.exists():
            print(f"WARNING: no existing groundTruth/{vid}.txt to source frame count from, skipping")
            n_skipped.append(vid)
            continue

        step_segments = json.loads(seg_path.read_text())
        fps = float(step_segments.get("fps") or 25.0)
        n_frames = sum(1 for _ in gt_legacy_path.open())

        instances = build_instances(step_segments, n_frames, fps)
        boundaries = build_boundaries(instances)

        annotation = {
            "video_id": vid,
            "fps": fps,
            "num_frames": n_frames,
            "duration": round(n_frames / fps, 3),
            "instances": instances,
            "boundaries": boundaries,
        }
        (ann_dir / f"{vid}.json").write_text(json.dumps(annotation, indent=2, ensure_ascii=False) + "\n")

        # regenerate legacy derived views from the instance annotation (single source of truth)
        labels = instances_to_frame_labels(instances, n_frames)
        (gt_dir / f"{vid}.txt").write_text("\n".join(labels) + "\n")

        frame_boundaries = instances_to_frame_boundaries(instances, n_frames)
        (boundary_dir / f"{vid}.txt").write_text("\n".join(str(b) for b in frame_boundaries) + "\n")

        n_written += 1

    # classes.json (problem_definition.md §17.1)
    classes_path = out_dir / "classes.json"
    classes_path.write_text(
        json.dumps([{"class_id": i, "class_name": name} for i, name in enumerate(CATEGORIES)], indent=2) + "\n"
    )

    # plain-text split lists (problem_definition.md §17.1), val == test here (no held-out test set exists)
    (splits_dir / "train.txt").write_text("\n".join(sorted(train_ids)) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(sorted(val_ids)) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(sorted(val_ids)) + "\n")

    print(f"Instance annotations written: {n_written} -> {ann_dir}")
    print(f"Skipped (no split assignment / missing data): {len(n_skipped)} {n_skipped if n_skipped else ''}")
    print(f"Regenerated derived views: {gt_dir}, {boundary_dir}")
    print(f"Wrote {classes_path}")
    print(f"Wrote splits/{{train,val,test}}.txt -> {splits_dir}")


if __name__ == "__main__":
    main()
