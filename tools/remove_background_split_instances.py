#!/usr/bin/env python3
"""Remove the `background` class from `dataset_tas_instance/` and cut it out
of the videos/features/annotations entirely.

Policy (per the task):
  - `background` instances at the head or tail of a video are simply
    trimmed off (the clip shrinks).
  - A `background` instance in the MIDDLE of a video (i.e. with real
    non-background instances both before and after it) causes the video to
    be SPLIT into two (or more) separate clips at that point -- each
    resulting clip contains zero background frames.

Operates on `dataset_tas_instance/` **in place** (regenerable from scratch
via `prepare_tas_dataset.py` + `prepare_tas_instance_dataset.py`, so no
backup is kept). For every original video, writes one or more new clips
named `<video_id>.mp4` (if a single run results) or `<video_id>_partN.mp4`
(if background in the middle caused a split), with:

  - `videos/<clip_id>.mp4`        -- actually re-encoded/cut with ffmpeg
  - `features/<clip_id>.npy`      -- sliced from the original (C, T) array
  - `annotations/<clip_id>.json`  -- instances/boundaries re-based to start
                                      at frame 0, class ids remapped to drop
                                      `background` (old id 0 removed, ids
                                      1..4 become 0..3)
  - `groundTruth/<clip_id>.txt` / `boundaries/<clip_id>.txt` -- regenerated
    derived views (same convention as prepare_tas_instance_dataset.py)

Old per-video files are removed and replaced by the new clip file(s).
`mapping.txt` / `classes.json` are rewritten without `background`.
`splits/{train,val,test}.{txt,bundle}` are rewritten so every new clip
inherits its parent video's original split assignment.

Usage:
    python tools/remove_background_split_instances.py
    python tools/remove_background_split_instances.py --out-dir src/TAS-instance-style/dataset_tas_instance
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

OLD_CATEGORIES = [
    "background",
    "Sewing/Joining",
    "Positioning/Handling",
    "Adjustment/Alignment/Preparation",
    "Inspection/Auxiliary",
]
NEW_CATEGORIES = OLD_CATEGORIES[1:]  # background dropped
# old_class_id -> new_class_id (background maps to None, never emitted)
OLD_TO_NEW = {1: 0, 2: 1, 3: 2, 4: 3}


def split_into_runs(instances: list[dict]) -> list[list[dict]]:
    """Split an instance list into maximal runs of non-background instances,
    dropping every background instance. Background only at the head/tail
    simply produces a single (trimmed) run; background in the middle
    produces 2+ runs.
    """
    runs: list[list[dict]] = []
    cur: list[dict] = []
    for inst in instances:
        if inst["class_id"] == 0:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(inst)
    if cur:
        runs.append(cur)
    return runs


def rebase_run(run: list[dict], fps: float) -> list[dict]:
    """Shift a run's instances so the first one starts at frame 0, and remap
    class ids to drop `background`."""
    offset = run[0]["start_frame"]
    rebased = []
    for i, inst in enumerate(run):
        start_frame = inst["start_frame"] - offset
        end_frame = inst["end_frame"] - offset
        rebased.append(
            {
                "id": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": round(start_frame / fps, 3),
                "end_time": round(end_frame / fps, 3),
                "class_id": OLD_TO_NEW[inst["class_id"]],
                "class_name": NEW_CATEGORIES[OLD_TO_NEW[inst["class_id"]]],
            }
        )
    return rebased


def build_boundaries(instances: list[dict]) -> list[dict]:
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
    labels = [None] * n_frames
    for inst in instances:
        for i in range(inst["start_frame"], inst["end_frame"]):
            labels[i] = inst["class_name"]
    assert all(l is not None for l in labels), "gap left after background removal -- bug"
    return labels


def instances_to_frame_boundaries(instances: list[dict], n_frames: int) -> list[int]:
    boundary = [0] * n_frames
    for inst in instances:
        s = inst["start_frame"]
        if 0 <= s < n_frames:
            boundary[s] = 1
    return boundary


def cut_video(src_video: Path, dst_video: Path, start_frame: int, end_frame: int, fps: float) -> None:
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_video),
        "-ss", f"{start_time:.6f}",
        "-t", f"{duration:.6f}",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst_video),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "src" / "TAS-instance-style" / "dataset_tas_instance",
    )
    parser.add_argument("--skip-video", action="store_true",
                         help="Skip ffmpeg video cutting (for fast dry-run validation of annotations/features only)")
    args = parser.parse_args()

    root: Path = args.out_dir
    ann_dir = root / "annotations"
    feat_dir = root / "features"
    video_dir = root / "videos"
    gt_dir = root / "groundTruth"
    boundary_dir = root / "boundaries"
    splits_dir = root / "splits"

    def read_ids(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]

    def read_bundle(path: Path) -> list[str]:
        return [l.strip()[:-4] for l in read_ids(path)]

    orig_train = set(read_ids(splits_dir / "train.txt")) or set(read_bundle(splits_dir / "train.bundle"))
    orig_val = set(read_ids(splits_dir / "val.txt")) or set(read_bundle(splits_dir / "val.bundle"))
    orig_test = set(read_ids(splits_dir / "test.txt")) or orig_val

    video_ids = sorted(p.stem for p in ann_dir.glob("*.json") if not re.search(r"_part\d+$", p.stem))

    # Clean up stale _part files from previous runs
    for d in (ann_dir, gt_dir, boundary_dir):
        for stale in d.glob("*_part*.*"):
            stale.unlink()

    new_train, new_val, new_test = [], [], []
    stats = {"unchanged": 0, "trimmed_single": 0, "split_multi": 0, "dropped_all_background": 0}

    old_files_to_remove = []

    for vid in video_ids:
        ann = json.loads((ann_dir / f"{vid}.json").read_text())
        fps = ann["fps"]
        # Check for original full video features in feat_dir or fallback to tmp backup
        feat_path = feat_dir / f"{vid}.npy"
        repo_root = Path(__file__).resolve().parent.parent
        backup_feat_path = repo_root / "tmp" / "dataset_tas_instance" / "features" / f"{vid}.npy"
        feat = None
        if feat_path.exists():
            feat = np.load(feat_path)
        elif backup_feat_path.exists():
            feat = np.load(backup_feat_path)
        else:
            print(f"NOTICE: {vid} has no .npy feature in features/ or tmp/ - skipping feature slicing for this video")
        src_video = (video_dir / f"{vid}.mp4").resolve()  # resolve symlink -> real source file

        runs = split_into_runs(ann["instances"])

        if not runs:
            stats["dropped_all_background"] += 1
            print(f"WARNING: {vid} is entirely background, dropping")
            old_files_to_remove.append(vid)
            continue

        if len(runs) == 1 and runs[0][0]["start_frame"] == 0 and runs[0][-1]["end_frame"] == ann["num_frames"]:
            stats["unchanged"] += 1
        elif len(runs) == 1:
            stats["trimmed_single"] += 1
        else:
            stats["split_multi"] += 1

        clip_ids = [vid] if len(runs) == 1 else [f"{vid}_part{i+1}" for i in range(len(runs))]

        for clip_id, run in zip(clip_ids, runs):
            orig_start = run[0]["start_frame"]
            orig_end = run[-1]["end_frame"]
            n_frames = orig_end - orig_start

            instances = rebase_run(run, fps)
            boundaries = build_boundaries(instances)

            annotation = {
                "video_id": clip_id,
                "fps": fps,
                "num_frames": n_frames,
                "duration": round(n_frames / fps, 3),
                "instances": instances,
                "boundaries": boundaries,
            }
            (ann_dir / f"{clip_id}.json").write_text(json.dumps(annotation, indent=2, ensure_ascii=False) + "\n")

            if feat is not None:
                clip_feat = feat[:, orig_start:orig_end]
                np.save(feat_dir / f"{clip_id}.npy", clip_feat)

            labels = instances_to_frame_labels(instances, n_frames)
            (gt_dir / f"{clip_id}.txt").write_text("\n".join(labels) + "\n")

            frame_boundaries = instances_to_frame_boundaries(instances, n_frames)
            (boundary_dir / f"{clip_id}.txt").write_text("\n".join(str(b) for b in frame_boundaries) + "\n")

            dst_video = video_dir / f"{clip_id}.mp4"
            if clip_id == vid and len(runs) == 1 and orig_start == 0 and orig_end == ann["num_frames"]:
                pass  # unchanged video, symlink already correct, nothing to do
            elif not args.skip_video:
                if dst_video.exists() or dst_video.is_symlink():
                    dst_video.unlink()
                cut_video(src_video, dst_video, orig_start, orig_end, fps)

            if vid in orig_train:
                new_train.append(clip_id)
            if vid in orig_val:
                new_val.append(clip_id)
            if vid in orig_test:
                new_test.append(clip_id)

        # only remove the OLD <vid>.* files when `vid` itself was NOT reused
        # as one of the new clip_ids (i.e. only the split-into-multiple-parts
        # case, where every new clip is named <vid>_partN instead of <vid>)
        if vid not in clip_ids:
            old_files_to_remove.append(vid)

    # remove stale old per-video files for videos that were split/trimmed/dropped
    for vid in old_files_to_remove:
        for d, ext in ((ann_dir, ".json"), (feat_dir, ".npy"), (gt_dir, ".txt"), (boundary_dir, ".txt")):
            p = d / f"{vid}{ext}"
            if p.exists():
                p.unlink()
        vp = video_dir / f"{vid}.mp4"
        if vp.exists() or vp.is_symlink():
            vp.unlink()

    # mapping.txt / classes.json without background, ids shifted down by 1
    (root / "mapping.txt").write_text("\n".join(f"{i} {c}" for i, c in enumerate(NEW_CATEGORIES)) + "\n")
    (root / "classes.json").write_text(
        json.dumps([{"class_id": i, "class_name": c} for i, c in enumerate(NEW_CATEGORIES)], indent=2) + "\n"
    )

    new_train = sorted(set(new_train))
    new_val = sorted(set(new_val))
    new_test = sorted(set(new_test)) or new_val

    # Only include clips with valid .npy features in train.bundle to prevent BaFormer crash
    ready_train = [v for v in new_train if (feat_dir / f"{v}.npy").exists()]

    (splits_dir / "train_all.txt").write_text("\n".join(new_train) + "\n")
    (splits_dir / "train_all.bundle").write_text("\n".join(f"{v}.txt" for v in new_train) + "\n")
    (splits_dir / "train.txt").write_text("\n".join(ready_train if ready_train else new_train) + "\n")
    (splits_dir / "train.bundle").write_text("\n".join(f"{v}.txt" for v in (ready_train if ready_train else new_train)) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(new_val) + "\n")
    (splits_dir / "val.bundle").write_text("\n".join(f"{v}.txt" for v in new_val) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(new_test) + "\n")

    print(f"Processed {len(video_ids)} original videos:")
    print(f"  unchanged (no background): {stats['unchanged']}")
    print(f"  trimmed (background at head/tail only): {stats['trimmed_single']}")
    print(f"  split into multiple clips (background in middle): {stats['split_multi']}")
    print(f"  dropped (entirely background): {stats['dropped_all_background']}")
    print(f"Resulting clips: train={len(new_train)}, val={len(new_val)}, test={len(new_test)}")
    print(f"Classes: {NEW_CATEGORIES}")


if __name__ == "__main__":
    main()
