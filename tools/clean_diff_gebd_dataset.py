#!/usr/bin/env python3
"""Filter out invalid segments (e.g. 'bỏ qua...', 'chưa tới...', 'unknown')
from DiffGEBD dataset annotations and remove chunks that overlap with them.

What this script does:
1. Scans source `data/**/step_segments.json` files to identify:
   - Invalid operations (matching keywords: 'bỏ qua', 'chưa tới', 'unknown', etc.).
   - Boundaries that touch these invalid operations (removes them so they are not treated as true boundaries).
   - Ignored time spans (e.g., 427.33s - 507.02s 'Bỏ qua đoạn bị giật').
2. Cleans `{train,val}_annotation.pkl`:
   - Strips boundaries touching invalid operations.
   - Saves cleaned annotations to `{split}_annotation_cleaned.pkl` (or overwrites `{split}_annotation.pkl`).
3. Cleans `{train,val}_annotation_chunked.pkl`:
   - Drops any chunk that overlaps with an invalid/ignored span (e.g. camera lag, unknown period).
   - Strips symlinked chunk directories for dropped chunks to keep disk tidy.
   - Saves cleaned chunked annotations to `{split}_annotation_chunked.pkl`.

Usage:
    # Dry-run / inspect only:
    python tools/clean_diff_gebd_dataset.py --dry-run

    # In-place clean:
    python tools/clean_diff_gebd_dataset.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path


DEFAULT_IGNORED_KEYWORDS = [
    "bỏ qua",
    "bo qua",
    "chưa tới",
    "chua toi",
    "unknown",
    "none",
    "chưa bắt đầu",
    "không may",
]


def is_ignored_operation(name: str | None, ignored_keywords: list[str]) -> bool:
    if not name or not name.strip():
        return True
    nl = name.strip().lower()
    return any(k in nl for k in ignored_keywords)


def get_video_spans_and_clean_boundaries(
    step_segments_path: Path, ignored_keywords: list[str]
) -> tuple[list[tuple[float, float, str]], list[float]]:
    """Return:
    - ignored_spans: list of (start_s, end_s, operation_name) for invalid operations.
    - valid_boundaries: list of boundary timestamps between two VALID consecutive operations.
    """
    with open(step_segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segs = data.get("segments", [])
    ignored_spans = []
    valid_boundaries = []

    for s in segs:
        op = s.get("operation_name", "")
        if is_ignored_operation(op, ignored_keywords):
            st = float(s.get("start_time_s", 0))
            et = float(s.get("end_time_s", 0))
            ignored_spans.append((st, et, op))

    for i in range(1, len(segs)):
        p_name = segs[i - 1].get("operation_name", "")
        c_name = segs[i].get("operation_name", "")
        if is_ignored_operation(p_name, ignored_keywords) or is_ignored_operation(c_name, ignored_keywords):
            continue
        prev_end = float(segs[i - 1]["end_time_s"])
        cur_start = float(segs[i]["start_time_s"])
        valid_boundaries.append(round((prev_end + cur_start) / 2.0, 3))

    return ignored_spans, valid_boundaries


def chunk_overlaps_span(c_start_s: float, c_end_s: float, spans: list[tuple[float, float, str]]) -> tuple[bool, str]:
    """Return (True, reason) if chunk interval intersects any invalid span."""
    for s_start, s_end, op_name in spans:
        if max(c_start_s, s_start) < min(c_end_s, s_end):
            return True, f"overlaps '{op_name}' ({s_start:.2f}s - {s_end:.2f}s)"
    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/diff_gebd_dataset"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--ignored-keywords",
        type=str,
        default=",".join(DEFAULT_IGNORED_KEYWORDS),
        help="Comma-separated keywords for operations to ignore",
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes in-place (overwriting pkl files)")
    args = parser.parse_args()

    ignored_keywords = [k.strip().lower() for k in args.ignored_keywords.split(",") if k.strip()]

    # 1. Collect all step_segments.json mapped by video_id
    video_info = {}
    for p in sorted(args.data_dir.rglob("step_segments.json")):
        vid = f"{p.parent.parent.name}_{p.parent.name}"
        ignored_spans, valid_bounds = get_video_spans_and_clean_boundaries(p, ignored_keywords)
        video_info[vid] = {
            "path": p,
            "ignored_spans": ignored_spans,
            "valid_bounds": valid_bounds,
        }

    print("=" * 65)
    print("STEP 1: CLEANING FULL-VIDEO ANNOTATIONS ({train,val}_annotation.pkl)")
    print("=" * 65)

    for split in ["train", "val"]:
        ann_path = args.dataset_dir / f"{split}_annotation.pkl"
        if not ann_path.exists():
            print(f"Skipping {ann_path} (not found)")
            continue

        with open(ann_path, "rb") as f:
            ann = pickle.load(f)

        modified_videos = 0
        total_bounds_removed = 0

        for vid, meta in ann.items():
            if vid not in video_info:
                continue
            info = video_info[vid]
            valid_bounds = info["valid_bounds"]
            fps = float(meta["fps"])
            vlen = int(meta["num_frames"])

            old_bounds = meta["substages_timestamps"][0]
            new_bounds = [b for b in old_bounds if any(abs(b - vb) <= 0.05 for vb in valid_bounds)]

            if len(new_bounds) != len(old_bounds):
                diff_count = len(old_bounds) - len(new_bounds)
                total_bounds_removed += diff_count
                modified_videos += 1
                new_frame_idx = [min(vlen, max(1, round(t * fps) + 1)) for t in new_bounds]
                meta["substages_timestamps"] = [new_bounds]
                meta["substages_myframeidx"] = [new_frame_idx]

        print(f"Split [{split}]: Removed {total_bounds_removed} invalid boundaries across {modified_videos} videos.")
        if args.apply:
            with open(ann_path, "wb") as f:
                pickle.dump(ann, f)
            print(f"  -> Saved cleaned annotation in-place to {ann_path}")

    print("\n" + "=" * 65)
    print("STEP 2: CLEANING CHUNKED DATASET ({train,val}_annotation_chunked.pkl)")
    print("=" * 65)

    for split in ["train", "val"]:
        chunked_ann_path = args.dataset_dir / f"{split}_annotation_chunked.pkl"
        if not chunked_ann_path.exists():
            print(f"Skipping {chunked_ann_path} (not found)")
            continue

        with open(chunked_ann_path, "rb") as f:
            chunked_ann = pickle.load(f)

        images_dir = args.dataset_dir / "images" / split
        kept_chunks = {}
        dropped_chunks = []

        for chunk_vid, meta in chunked_ann.items():
            source_vid = meta.get("source_vid")
            if source_vid not in video_info:
                kept_chunks[chunk_vid] = meta
                continue

            info = video_info[source_vid]
            fps = float(meta["fps"])
            chunk_start_frame = int(meta["chunk_start_frame"])
            num_frames = int(meta["num_frames"])
            chunk_end_frame = chunk_start_frame + num_frames - 1

            c_start_s = (chunk_start_frame - 1) / fps
            c_end_s = (chunk_end_frame - 1) / fps

            overlap, reason = chunk_overlaps_span(c_start_s, c_end_s, info["ignored_spans"])
            if overlap:
                dropped_chunks.append((chunk_vid, reason))
                continue

            # Update local boundaries in chunk
            valid_bounds_frames = [
                round(t * fps) + 1 for t in info["valid_bounds"]
            ]
            local_bounds = [
                int(b - chunk_start_frame + 1)
                for b in valid_bounds_frames
                if chunk_start_frame <= b <= chunk_end_frame
            ]
            meta["substages_myframeidx"] = [local_bounds]
            meta["substages_timestamps"] = [[round((b - 1) / fps, 3) for b in local_bounds]]
            kept_chunks[chunk_vid] = meta

        print(f"Split [{split}]: Total chunks: {len(chunked_ann)} -> Kept: {len(kept_chunks)} | Dropped: {len(dropped_chunks)}")
        if dropped_chunks:
            print(f"  Examples of dropped chunks in [{split}]:")
            for c_id, r in dropped_chunks[:4]:
                print(f"    - {c_id}: {r}")

        if args.apply:
            with open(chunked_ann_path, "wb") as f:
                pickle.dump(kept_chunks, f)
            print(f"  -> Saved cleaned chunked annotation to {chunked_ann_path}")

            # Remove dropped chunk symlink directories if present
            removed_dirs = 0
            for c_id, _ in dropped_chunks:
                chunk_dir = images_dir / c_id
                if chunk_dir.exists() and chunk_dir.is_dir():
                    shutil.rmtree(chunk_dir)
                    removed_dirs += 1
            if removed_dirs > 0:
                print(f"  -> Removed {removed_dirs} dropped chunk image directories in {images_dir}")

    print("\n" + "=" * 65)
    if not args.apply:
        print("DRY-RUN COMPLETE. Run with `--apply` to save all cleaned files.")
    else:
        print("ALL DATASET ANNOTATIONS & CHUNKS SUCCESSFULLY CLEANED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
