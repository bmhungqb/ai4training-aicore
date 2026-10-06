#!/usr/bin/env python3
"""Build a TAS-instance-style dataset (same layout as `dataset_tas_instance/`)
from `original_data/processed_data/` + the `balanced_split` train/valid lists.

Source layout (per clip, under original_data/processed_data/<cd>/<chuyen>/):
    <clip>.mp4   -- video for this clip (already cut to the segment's time range)
    <clip>.json  -- {"video_path", "source_video", "fps", "source_time_range_s",
                     "n_segments", "segments": [{id, start, end,
                     original_operation, category}, ...]}
    (there is also one shared <video>.mask.png per source video -- not used here)

`category` is already one of 4 coarse classes (no classification heuristics
needed, unlike the older `data/` + step_segments_clean.json pipeline):
    Preparation, Positioning/Adjustment/Alignment, Sewing/Joining/Handling,
    Inspection/Auxiliary
There is NO `background` class in this dataset -- the source `category`
field never contains one, and any uncovered gap (head/tail, or between two
consecutive segments) is instead ABSORBED into the nearest real segment
(never labeled as a separate class):
  - head/middle gap -> extends the *next* segment's start_frame backward to
    cover it.
  - tail gap (uncovered frames after the last segment, e.g. because the
    source video file runs slightly longer than the last annotated segment)
    -> extends the *previous* (last) segment's end_frame forward to cover it.
In practice inter-segment gaps are sub-frame rounding noise (<0.08s, i.e.
<1 frame at this dataset's ~15fps) and round away to 0 frames; the only
gaps that actually produce a frame-level extension are video-file tail gaps
(observed up to ~100 frames / ~6.8s in this dataset).

Output: <out-dir>/
    classes.json, mapping.txt
    videos/<video_id>.mp4            (symlink)
    annotations/<video_id>.json      (PRIMARY ground truth, instance-level)
    groundTruth/<video_id>.txt       (derived per-frame class-name view)
    boundaries/<video_id>.txt        (derived per-frame 0/1 instance-start view)
    splits/{train,val,test}.txt      (video_id lists; val == test, from balanced_split/valid.txt)
    splits/{train,val}.bundle        (legacy "<video_id>.txt" per line)

video_id is derived from the balanced_split path, e.g.
    cd9/chuyen3/cam-03_..._part1_0.00-98.88  ->  cd9_chuyen3_part1
    cd1/chuyen1/cam-03_..._cut_0_0-0_57      ->  cd1_chuyen1

Usage:
    python src/TAS-instance-style/scripts/build_dataset.py
    python src/TAS-instance-style/scripts/build_dataset.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = REPO_ROOT / "original_data" / "processed_data"
SPLIT_DIR = REPO_ROOT / "original_data" / "split" / "balanced_split"
DEFAULT_OUT_DIR = REPO_ROOT / "src" / "TAS-instance-style" / "dataset"

# Fixed class order. No background class -- see module docstring: every gap
# is absorbed into a neighboring real segment instead.
CATEGORIES = [
    "Preparation",
    "Positioning/Adjustment/Alignment",
    "Sewing/Joining/Handling",
    "Inspection/Auxiliary",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CATEGORIES)}


def video_id_from_split_line(line: str) -> str:
    """'cd9/chuyen3/cam-03_..._part1_0.00-98.88' -> 'cd9_chuyen3_part1'"""
    cd, chuyen, stem = line.strip().split("/", 2)
    m = re.search(r"(_part\d+)_[\d.]+-[\d.]+$", stem)
    suffix = m.group(1) if m else ""
    return f"{cd}_{chuyen}{suffix}"


def ffprobe_nb_frames(video_path: Path, fps: float, duration_fallback: float | None = None) -> int:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames,duration",
        "-of", "json", str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120).stdout
        info = json.loads(out)["streams"][0]
        nb = int(info.get("nb_read_frames", 0) or 0)
        if nb > 0:
            return nb
        duration = float(info.get("duration", 0.0) or 0.0)
        if duration:
            return max(1, round(duration * fps))
    except Exception as e:
        print(f"  WARNING: ffprobe failed for {video_path}: {e}")
    if duration_fallback:
        return max(1, round(duration_fallback * fps))
    return 0


def build_instances(segments: list[dict], n_frames: int, fps: float) -> list[dict]:
    """No background class: any uncovered gap is absorbed into the nearest
    real segment (extends the next segment backward for head/middle gaps,
    extends the last segment forward for a trailing/tail gap) rather than
    being labeled as a separate 'background' instance.
    """
    segs = sorted(segments, key=lambda s: s["start"])
    instances: list[dict] = []
    cursor = 0

    def add(start_frame: int, end_frame: int, class_name: str) -> None:
        if end_frame <= start_frame:
            return
        instances.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": round(start_frame / fps, 3),
            "end_time": round(end_frame / fps, 3),
            "class_id": CLASS_TO_ID[class_name],
            "class_name": class_name,
        })

    for seg in segs:
        start_f = max(0, round(float(seg["start"]) * fps))
        end_f = min(n_frames, round(float(seg["end"]) * fps) + 1)  # inclusive end -> exclusive
        # absorb any head/middle gap before this segment by starting earlier
        # (extend this segment backward to `cursor` instead of creating a
        # separate background instance for the gap)
        start_f = cursor
        if end_f <= start_f:
            continue
        class_name = seg["category"]
        if class_name not in CLASS_TO_ID:
            print(f"  WARNING: unknown category {class_name!r}, skipping segment")
            continue
        add(start_f, end_f, class_name)
        cursor = end_f

    # absorb any trailing/tail gap into the last real instance (extend its
    # end_frame forward to n_frames) instead of creating a background instance
    if cursor < n_frames and instances:
        last = instances[-1]
        last["end_frame"] = n_frames
        last["end_time"] = round(n_frames / fps, 3)

    return [{"id": i, **inst} for i, inst in enumerate(instances)]


def build_boundaries(instances: list[dict]) -> list[dict]:
    boundaries = []
    for left, right in zip(instances, instances[1:]):
        btype = "same_class" if left["class_id"] == right["class_id"] else "class_change"
        boundaries.append({
            "frame": right["start_frame"],
            "type": btype,
            "left_instance": left["id"],
            "right_instance": right["id"],
        })
    return boundaries


def instances_to_frame_labels(instances: list[dict], n_frames: int) -> list[str]:
    labels = ["background"] * n_frames
    for inst in instances:
        for i in range(inst["start_frame"], inst["end_frame"]):
            labels[i] = inst["class_name"]
    return labels


def instances_to_frame_boundaries(instances: list[dict], n_frames: int) -> list[int]:
    boundary = [0] * n_frames
    for inst in instances:
        s = inst["start_frame"]
        if 0 <= s < n_frames:
            boundary[s] = 1
    return boundary


def read_split(path: Path) -> list[tuple[str, str]]:
    """Returns [(video_id, relative_json_path_without_ext), ...]"""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append((video_id_from_split_line(line), line))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-dir", type=Path, default=SRC_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    ann_dir = out_dir / "annotations"
    gt_dir = out_dir / "groundTruth"
    boundary_dir = out_dir / "boundaries"
    videos_dir = out_dir / "videos"
    splits_dir = out_dir / "splits"
    for d in (ann_dir, gt_dir, boundary_dir, videos_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_entries = read_split(args.split_dir / "train.txt")
    val_entries = read_split(args.split_dir / "valid.txt")

    train_ids, val_ids = [], []
    n_written, n_skipped = 0, []

    for entries, id_list in ((train_entries, train_ids), (val_entries, val_ids)):
        for vid, rel in entries:
            json_path = args.src_dir / f"{rel}.json"
            mp4_path = args.src_dir / f"{rel}.mp4"
            if not json_path.exists() or not mp4_path.exists():
                print(f"WARNING: missing source for {vid} ({rel}), skipping")
                n_skipped.append(vid)
                continue

            data = json.loads(json_path.read_text())
            fps = float(data.get("fps") or 25.0)
            segments = data.get("segments", [])
            if not segments:
                print(f"WARNING: no segments for {vid}, skipping")
                n_skipped.append(vid)
                continue

            last_end = max(float(s["end"]) for s in segments)
            n_frames = ffprobe_nb_frames(mp4_path, fps, duration_fallback=last_end)
            if n_frames <= 0:
                print(f"WARNING: could not determine frame count for {mp4_path}, skipping")
                n_skipped.append(vid)
                continue

            instances = build_instances(segments, n_frames, fps)
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

            labels = instances_to_frame_labels(instances, n_frames)
            (gt_dir / f"{vid}.txt").write_text("\n".join(labels) + "\n")

            frame_boundaries = instances_to_frame_boundaries(instances, n_frames)
            (boundary_dir / f"{vid}.txt").write_text("\n".join(str(b) for b in frame_boundaries) + "\n")

            link_path = videos_dir / f"{vid}.mp4"
            if link_path.exists() or link_path.is_symlink():
                if args.overwrite:
                    link_path.unlink()
            if not link_path.exists() and not link_path.is_symlink():
                link_path.symlink_to(mp4_path.resolve())

            id_list.append(vid)
            n_written += 1

    # classes.json / mapping.txt
    (out_dir / "classes.json").write_text(
        json.dumps([{"class_id": i, "class_name": n} for i, n in enumerate(CATEGORIES)], indent=2) + "\n"
    )
    (out_dir / "mapping.txt").write_text(
        "\n".join(f"{i} {n}" for i, n in enumerate(CATEGORIES)) + "\n"
    )

    # splits (val == test, as in dataset_tas_instance)
    (splits_dir / "train.txt").write_text("\n".join(sorted(train_ids)) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(sorted(val_ids)) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(sorted(val_ids)) + "\n")
    (splits_dir / "train.bundle").write_text("\n".join(f"{v}.txt" for v in sorted(train_ids)) + "\n")
    (splits_dir / "val.bundle").write_text("\n".join(f"{v}.txt" for v in sorted(val_ids)) + "\n")

    print(f"\nWritten: {n_written}  (train={len(train_ids)}, val/test={len(val_ids)})")
    print(f"Skipped: {len(n_skipped)} {n_skipped if n_skipped else ''}")
    print(f"Out dir: {out_dir}")


if __name__ == "__main__":
    main()
