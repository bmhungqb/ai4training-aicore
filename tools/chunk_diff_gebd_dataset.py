#!/usr/bin/env python3
"""Chunk long Sewing videos into short (~10-15s) clips for DiffGEBD END_TO_END training.

Root cause (see docs/step_segment/diff_gebd.md): DiffGEBD's END_TO_END sampler does
    selected_indices = np.linspace(1, vlen, cfg.INPUT.SEQUENCE_LENGTH)
which assumes every video is already short (Kinetics-GEBD: ~10s / ~300 frames).
Sewing videos are multi-minute, so this downsamples so aggressively that boundary
frames are skipped entirely, starving the model of positive labels (-> low recall).

This script re-uses the already-extracted frames + full-video annotation produced by
`tools/prepare_diff_gebd_dataset.py` and slices each video into overlapping chunks of
`--chunk-seconds` (e.g. 5s, ~75 frames @ ~15fps to match SEQUENCE_LENGTH=75).
Boundary timestamps are re-expressed in each chunk's local frame/time reference.
Frames are symlinked (no re-encoding/copying) into `images/{split}/<vid>__chunk<i>/`
(or `images/{split}/<vid>__grid<g>_chunk<i>/` for augmented train grids).

Multi-grid temporal jittering (train only): `--train-offsets` lists K sliding-window
start offsets (seconds). Each offset re-runs the chunking grid over the same video, so
boundary events land at different positions (start/middle/end) across chunks, reducing
position bias. val always uses a single, fixed grid with offset 0.

Usage:
    python tools/prepare_diff_gebd_dataset.py          # if not already done
    python tools/chunk_diff_gebd_dataset.py
    python tools/chunk_diff_gebd_dataset.py --chunk-seconds 5.0 --overlap-seconds 1.0 \\
        --min-chunk-seconds 3.0 --train-offsets 0.0,1.5,3.0 \\
        --drop-negative-ratio 0.3   # subsample empty chunks in train split only

Output:
    <dataset-dir>/images/{train,val}/<vid>[__grid<g>]_chunk<i>/frame<n>.jpg  (symlinks)
    <dataset-dir>/{train,val}_annotation_chunked.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from pathlib import Path


def _chunk_grid(vid: str, meta: dict, chunk_frames: int, overlap_frames: int, min_chunk_frames: int,
                 grid_start: int, name_suffix: str):
    """Yield (chunk_vid, start_frame, end_frame, local_boundaries) for one video,
    for a single sliding-window grid starting at `grid_start` (1-indexed).

    start/end are 1-indexed, inclusive, in the *source* video's frame numbering.
    """
    vlen = meta["num_frames"]
    boundaries = meta["substages_myframeidx"][0]
    stride = max(1, chunk_frames - overlap_frames)

    start = grid_start
    chunk_idx = 0
    while start <= vlen:
        end = min(start + chunk_frames - 1, vlen)
        if end - start + 1 >= min_chunk_frames or chunk_idx == 0:
            local_boundaries = [int(b - start + 1) for b in boundaries if start <= b <= end]
            yield f"{vid}{name_suffix}__chunk{chunk_idx}", start, end, local_boundaries
        if end >= vlen:
            break
        start += stride
        chunk_idx += 1


def chunk_video(vid: str, meta: dict, chunk_frames: int, overlap_frames: int, min_chunk_frames: int,
                 fps: float, is_train: bool, train_offsets: list[float]):
    """Yield (chunk_vid, start_frame, end_frame, local_boundaries) for one video.

    For `is_train` videos, repeats the sliding-window grid once per offset in
    `train_offsets` (multi-grid temporal jittering augmentation), each grid
    getting a distinct `__grid<g>` name suffix. For val (or when only one
    offset is given), a single grid starting at frame 1 is used.
    """
    if not is_train:
        # val: single, fixed grid at offset 0 (no augmentation).
        yield from _chunk_grid(vid, meta, chunk_frames, overlap_frames, min_chunk_frames,
                                grid_start=1, name_suffix="")
        return

    if len(train_offsets) <= 1:
        offset_s = train_offsets[0] if train_offsets else 0.0
        yield from _chunk_grid(vid, meta, chunk_frames, overlap_frames, min_chunk_frames,
                                grid_start=1 + round(offset_s * fps), name_suffix="")
        return

    for g, offset_s in enumerate(train_offsets):
        grid_start = 1 + round(offset_s * fps)
        yield from _chunk_grid(vid, meta, chunk_frames, overlap_frames, min_chunk_frames,
                                grid_start=grid_start, name_suffix=f"__grid{g}")


def symlink_chunk_frames(src_dir: Path, dst_dir: Path, start: int, end: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, global_idx in enumerate(range(start, end + 1), start=1):
        link_path = dst_dir / f"frame{local_idx}.jpg"
        if link_path.exists() or link_path.is_symlink():
            continue
        target = Path("..") / src_dir.name / f"frame{global_idx}.jpg"
        os.symlink(target, link_path)


def build_chunked_annotation(
    annotation: dict,
    images_dir: Path,
    chunk_seconds: float,
    overlap_seconds: float,
    min_chunk_seconds: float,
    drop_negative_ratio: float,
    apply_drop: bool,
    rng: random.Random,
    is_train: bool,
    train_offsets: list[float],
) -> dict:
    chunked: dict = {}
    for vid, meta in annotation.items():
        fps = float(meta["fps"])
        chunk_frames = max(1, round(chunk_seconds * fps))
        overlap_frames = max(0, round(overlap_seconds * fps))
        min_chunk_frames = max(1, round(min_chunk_seconds * fps))

        src_dir = images_dir / meta["path_frame"]
        if not src_dir.exists():
            print(f"WARNING: frame dir {src_dir} missing, skipping {vid}")
            continue

        for chunk_vid, start, end, local_boundaries in chunk_video(
            vid, meta, chunk_frames, overlap_frames, min_chunk_frames, fps, is_train, train_offsets
        ):
            if apply_drop and not local_boundaries and drop_negative_ratio > 0:
                if rng.random() < drop_negative_ratio:
                    continue

            dst_dir = images_dir / chunk_vid
            symlink_chunk_frames(src_dir, dst_dir, start, end)

            num_frames = end - start + 1
            chunked[chunk_vid] = {
                "num_frames": num_frames,
                "path_video": meta["path_video"],
                "fps": fps,
                "video_duration": num_frames / fps,
                "path_frame": chunk_vid,
                "f1_consis": [1.0],
                "f1_consis_avg": 1.0,
                "substages_myframeidx": [local_boundaries],
                "substages_timestamps": [[round((b - 1) / fps, 3) for b in local_boundaries]],
                # extra fields used by tools/export_diffgebd_predictions.py to
                # aggregate chunk-level predictions back to the source video.
                "source_vid": vid,
                "chunk_start_frame": start,
            }
    return chunked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/diff_gebd_dataset"))
    parser.add_argument("--chunk-seconds", type=float, default=5.0, help="Target chunk length in seconds")
    parser.add_argument("--overlap-seconds", type=float, default=1.0, help="Overlap between consecutive chunks")
    parser.add_argument("--min-chunk-seconds", type=float, default=3.0, help="Drop trailing chunks shorter than this")
    parser.add_argument("--drop-negative-ratio", type=float, default=0.0,
                         help="Fraction of boundary-free (negative) train chunks to randomly drop, to rebalance pos/neg")
    parser.add_argument("--train-offsets", type=str, default="0.0",
                         help="Comma-separated list of grid start offsets (seconds), applied to the train split only "
                              "(multi-grid temporal jittering augmentation). val always uses a single offset=0 grid.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_offsets = [float(x) for x in args.train_offsets.split(",") if x.strip() != ""]

    for split in ["train", "val"]:
        ann_path = args.dataset_dir / f"{split}_annotation.pkl"
        if not ann_path.exists():
            print(f"WARNING: {ann_path} not found, skipping split '{split}'")
            continue
        with open(ann_path, "rb") as f:
            annotation = pickle.load(f)

        images_dir = args.dataset_dir / "images" / split
        chunked = build_chunked_annotation(
            annotation, images_dir,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            min_chunk_seconds=args.min_chunk_seconds,
            drop_negative_ratio=args.drop_negative_ratio,
            apply_drop=(split == "train"),
            rng=rng,
            is_train=(split == "train"),
            train_offsets=train_offsets,
        )

        n_pos = sum(1 for c in chunked.values() if c["substages_myframeidx"][0])
        out_path = args.dataset_dir / f"{split}_annotation_chunked.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(chunked, f)
        print(f"{split}: {len(annotation)} videos -> {len(chunked)} chunks "
              f"({n_pos} with >=1 boundary, {len(chunked) - n_pos} negative) -> {out_path}")


if __name__ == "__main__":
    main()
