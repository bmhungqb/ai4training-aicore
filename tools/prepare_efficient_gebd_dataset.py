#!/usr/bin/env python3
"""Build an EfficientGEBD training dataset from per-video `step_segments.json` files.

Mirrors `tools/prepare_ddm_dataset.py` but targets the EfficientGEBD repo
(`src/step_segment/EfficientGEBD/`), which expects data laid out as:

    <out-dir>/images/{train,val}/<video_id>/frame<N>.jpg   (offline extracted frames, N starting at 1)
    <out-dir>/train_annotation.pkl
    <out-dir>/val_annotation.pkl

Each `*_annotation.pkl` follows the Kinetics-GEBD / TAPOS schema expected by
`EfficientGEBD/datasets/dataset.py::prepare_gebd_annotations` (name='SEWING'):

    {
      "<video_id>": {
        "num_frames": int,
        "path_video": str,                 # original .mp4 path (relative to repo root)
        "fps": float,
        "video_duration": float,
        "path_frame": "<video_id>",         # matched to images/{split}/<video_id>/
        "f1_consis": [1.0],                 # single annotator (our GT)
        "f1_consis_avg": 1.0,
        "substages_myframeidx": [[frame_idx, ...]],   # 1 boundary list per annotator
        "substages_timestamps": [[seconds, ...]],
      },
      ...
    }

If a `<video_stem>.mask.png` exists next to the source video (see
`tools/mask_editor`), frames are cropped to the mask's non-zero bounding box
(with a margin) and multiplied by the binary mask before being saved, exactly
like the ROI pipeline used for DDM-Net (`datasets/roi_mask.py`).

Usage:
    python tools/prepare_efficient_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42
    python tools/prepare_efficient_gebd_dataset.py --split-mode by_folder --val-folders cd18,cd19,cd20
    python tools/prepare_efficient_gebd_dataset.py --max-videos 2 --overwrite   # quick sanity check
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Video discovery / split (shared logic with tools/prepare_ddm_dataset.py)
# --------------------------------------------------------------------------- #
def find_videos(data_dir: Path) -> list[Path]:
    return sorted(p.parent for p in data_dir.rglob("step_segments.json"))


def video_unique_id(video_dir: Path) -> str:
    return f"{video_dir.parent.name}_{video_dir.name}"


def find_mp4(video_dir: Path) -> Path | None:
    mp4s = list(video_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def is_unknown(name: str | None) -> bool:
    return not name or not name.strip() or name.strip().upper() in ["UNKNOWN", "NONE"]


def split_videos(
    video_dirs: list[Path],
    split_mode: str,
    val_ratio: float,
    val_folders: list[str],
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if split_mode == "by_folder":
        val_set = set(val_folders)
        val = [d for d in video_dirs if d.parent.name in val_set]
        train = [d for d in video_dirs if d.parent.name not in val_set]
        return train, val

    rng = random.Random(seed)
    shuffled = video_dirs.copy()
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


# --------------------------------------------------------------------------- #
# Boundary extraction from step_segments.json
# --------------------------------------------------------------------------- #
def trimmed_segments(step_segments: dict) -> list[dict]:
    """Same trimming rule as tools/prepare_ddm_dataset.py::to_ddm_events."""
    segs = step_segments.get("segments", [])
    if not segs:
        return []
    start_idx, end_idx = 0, len(segs)
    while start_idx < len(segs) and is_unknown(segs[start_idx].get("operation_name", "")):
        start_idx += 1
    while end_idx > start_idx and is_unknown(segs[end_idx - 1].get("operation_name", "")):
        end_idx -= 1
    return segs[start_idx:end_idx]


def boundary_timestamps(step_segments: dict) -> list[float]:
    """Internal boundaries: transition point between consecutive (trimmed) segments."""
    segs = trimmed_segments(step_segments)
    bounds = []
    for i in range(1, len(segs)):
        prev_end = float(segs[i - 1]["end_time_s"])
        cur_start = float(segs[i]["start_time_s"])
        bounds.append(round((prev_end + cur_start) / 2.0, 3))
    return bounds


# --------------------------------------------------------------------------- #
# ROI mask (crop-to-bbox + multiply), numpy-only re-implementation of
# EfficientGEBD/datasets/roi_mask.py so this tool has no torch dependency.
# --------------------------------------------------------------------------- #
def find_mask_for_video(video_path: Path) -> Path | None:
    cand = video_path.with_suffix("").with_suffix(".mask.png")
    if cand.exists():
        return cand
    any_mask = next(video_path.parent.glob("*.mask.png"), None)
    return any_mask


def compute_mask_bbox(mask_path: Path, margin: float = 0.05):
    arr = np.array(Image.open(mask_path).convert("L"))
    H, W = arr.shape
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    if margin > 0:
        mx = int((x_max - x_min) * margin)
        my = int((y_max - y_min) * margin)
        x_min, x_max = max(0, x_min - mx), min(W, x_max + mx + 1)
        y_min, y_max = max(0, y_min - my), min(H, y_max + my + 1)
    else:
        x_max, y_max = min(W, x_max + 1), min(H, y_max + 1)
    binary_mask = (arr[y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
    return x_min, x_max, y_min, y_max, binary_mask


def apply_roi_to_frame(frame_path: Path, bbox_mask) -> None:
    x_min, x_max, y_min, y_max, binary_mask = bbox_mask
    img = Image.open(frame_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    ymin, ymax = max(0, min(y_min, h - 1)), max(1, min(y_max, h))
    xmin, xmax = max(0, min(x_min, w - 1)), max(1, min(x_max, w))
    cropped = arr[ymin:ymax, xmin:xmax]
    mask = binary_mask[: cropped.shape[0], : cropped.shape[1], None]
    masked = cropped * mask
    Image.fromarray(masked.astype(np.uint8)).save(frame_path, quality=95)


# --------------------------------------------------------------------------- #
# Frame extraction
# --------------------------------------------------------------------------- #
def ffprobe_fps_duration(video_path: Path) -> tuple[float, float, int]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,duration",
        "-of", "json", str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    info = json.loads(out)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else float(num)
    duration = float(info.get("duration", 0.0) or 0.0)
    nb_frames = int(info.get("nb_frames", 0) or 0)
    return fps, duration, nb_frames


def extract_frames(video_path: Path, out_dir: Path, overwrite: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("frame*.jpg"))
    if existing and not overwrite:
        return len(existing)
    for f in existing:
        f.unlink()
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-f", "image2", "-qscale:v", "2", "-loglevel", "quiet",
        str(out_dir / "frame%d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("frame*.jpg")))


# --------------------------------------------------------------------------- #
# Main annotation build
# --------------------------------------------------------------------------- #
def build_annotation(video_dirs: list[Path], images_out_dir: Path, repo_root: Path,
                      overwrite: bool, mask_margin: float, apply_roi: bool) -> dict:
    annotation: dict = {}
    for video_dir in video_dirs:
        mp4_path = find_mp4(video_dir)
        if mp4_path is None:
            print(f"WARNING: no .mp4 found in {video_dir}, skipping")
            continue

        step_segments = json.loads((video_dir / "step_segments.json").read_text())
        bounds_s = boundary_timestamps(step_segments)
        if not bounds_s:
            print(f"WARNING: no internal boundaries in {video_dir}, skipping")
            continue

        vid = video_unique_id(video_dir)
        frame_dir = images_out_dir / vid

        # Prefer the fps recorded in step_segments.json: it is the *effective*
        # fps (num_frames / duration) used when the boundary timestamps were
        # computed, which can differ from the container's nominal r_frame_rate
        # (e.g. variable frame rate videos advertised as 30fps but averaging ~15fps).
        _, ffprobe_duration, _ = ffprobe_fps_duration(mp4_path)
        vlen = extract_frames(mp4_path, frame_dir, overwrite=overwrite)
        if vlen == 0:
            print(f"WARNING: failed to extract frames for {mp4_path}, skipping")
            continue
        duration = ffprobe_duration or (vlen / float(step_segments.get("fps", 25.0)))
        json_fps = step_segments.get("fps")
        fps = float(json_fps) if json_fps else (vlen / duration if duration else 25.0)

        if apply_roi:
            mask_path = find_mask_for_video(mp4_path)
            if mask_path is not None:
                bbox_mask = compute_mask_bbox(mask_path, margin=mask_margin)
                if bbox_mask is not None:
                    for frame_path in frame_dir.glob("frame*.jpg"):
                        apply_roi_to_frame(frame_path, bbox_mask)

        # 1-indexed frame numbers, matching the ffmpeg `frame%d.jpg` output and the
        # EfficientGEBD `template='frame{:d}.jpg'` convention.
        frame_idx = [min(vlen, max(1, round(t * fps) + 1)) for t in bounds_s]

        annotation[vid] = {
            "num_frames": vlen,
            "path_video": str(mp4_path.relative_to(repo_root)) if mp4_path.is_relative_to(repo_root) else str(mp4_path),
            "fps": fps,
            "video_duration": duration,
            "path_frame": vid,
            "f1_consis": [1.0],
            "f1_consis_avg": 1.0,
            "substages_myframeidx": [frame_idx],
            "substages_timestamps": [bounds_s],
        }
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/efficient_gebd_dataset"))
    parser.add_argument("--split-mode", choices=["random", "by_folder"], default="random")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-folders", type=str, default="", help="Comma-separated cd folder names, e.g. cd18,cd19,cd20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-margin", type=float, default=0.05)
    parser.add_argument("--no-roi-mask", action="store_true", help="Disable ROI crop/mask even if *.mask.png exists")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract frames even if the folder already exists")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit number of videos (0 = all); useful for a quick sanity check")
    args = parser.parse_args()

    repo_root = Path.cwd()
    video_dirs = find_videos(args.data_dir)
    if not video_dirs:
        print(f"No step_segments.json found under {args.data_dir}. Run tools/process_step_segments.py first.")
        return
    if args.max_videos:
        video_dirs = video_dirs[: args.max_videos]

    val_folders = [f.strip() for f in args.val_folders.split(",") if f.strip()]
    if args.split_mode == "by_folder" and not val_folders:
        raise ValueError("--split-mode by_folder requires --val-folders")

    train_dirs, val_dirs = split_videos(video_dirs, args.split_mode, args.val_ratio, val_folders, args.seed)
    print(f"Found {len(video_dirs)} videos -> train={len(train_dirs)}, val={len(val_dirs)}")

    images_dir = args.out_dir / "images"
    apply_roi = not args.no_roi_mask

    for split, dirs in [("train", train_dirs), ("val", val_dirs)]:
        split_images_dir = images_dir / split
        split_images_dir.mkdir(parents=True, exist_ok=True)
        annotation = build_annotation(dirs, split_images_dir, repo_root, args.overwrite, args.mask_margin, apply_roi)
        out_path = args.out_dir / f"{split}_annotation.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(annotation, f)
        print(f"Wrote {len(annotation)} {split} videos -> {out_path}")

    print(f"Frames extracted under {images_dir}")


if __name__ == "__main__":
    main()
