#!/usr/bin/env python3
"""
tools/place_masks.py

Finds all *.mask.png files from a staging directory (or source directory),
matches each mask to its corresponding video in data/cd*/chuyen* by video stem,
and copies/moves it next to the video.

Usage:
    python tools/place_masks.py --src /tmp/masks_staging
    python tools/place_masks.py --src /tmp/masks_staging --copy
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Place *.mask.png next to matching video in data/cd*/chuyen*")
    parser.add_argument("--src", default="/tmp/masks_staging", help="Source staging folder containing *.mask.png")
    parser.add_argument("--data-dir", default="data", help="Target data root directory (default: data)")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = parser.parse_args()

    src_dir = Path(args.src)
    data_dir = Path(args.data_dir)

    if not src_dir.exists():
        print(f"❌ Source directory does not exist: {src_dir}")
        return

    # Index all videos by stem
    video_map: dict[str, Path] = {}
    for vid in data_dir.glob("cd*/chuyen*/*.mp4"):
        video_map[vid.stem] = vid

    print(f"📹 Found {len(video_map)} videos under {data_dir}/cd*/chuyen*/")

    # Find all mask.png files in source
    mask_files = list(src_dir.rglob("*.mask.png"))
    print(f"🎭 Found {len(mask_files)} *.mask.png file(s) in {src_dir}")

    placed = 0
    missing = 0

    for mf in sorted(mask_files):
        # Name format: <stem>.mask.png
        video_stem = mf.name.replace(".mask.png", "")
        if video_stem in video_map:
            target_video = video_map[video_stem]
            dest_mask = target_video.parent / mf.name
            if args.copy:
                shutil.copy2(mf, dest_mask)
            else:
                shutil.move(mf, dest_mask)
            print(f"  ✅ Placed: {mf.name} -> {dest_mask.parent}")
            placed += 1
        else:
            print(f"  ⚠️ No matching video for: {mf.name}")
            missing += 1

    print(f"\n🎉 Successfully placed {placed} mask(s) into corresponding folders! ({missing} unmatched)")


if __name__ == "__main__":
    main()
