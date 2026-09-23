#!/usr/bin/env python3
"""Merge consecutive action segments that share the same operation into step segments.

Scans `data/**/action_segments_annotated.json`, merges consecutive segments with
identical `operation_name` into a single step segment, and writes the result to
`step_segments.json` alongside the source file.

Empty (`""`) and `"UNKNOWN"` operation names are kept as their own segments
(they are not merged with a *different* neighbouring name, but consecutive
runs of the same empty/UNKNOWN name are still merged like any other name).

Usage:
    python tools/process_step_segments.py
    python tools/process_step_segments.py --data-dir data --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_segments(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments sharing the same operation_name."""
    merged: list[dict] = []
    for seg in segments:
        name = seg.get("operation_name", "")
        start = seg["start_time_s"]
        end = seg["end_time_s"]
        if merged and merged[-1]["operation_name"] == name:
            merged[-1]["end_time_s"] = end
            merged[-1]["n_merged"] += 1
        else:
            merged.append(
                {
                    "operation_name": name,
                    "start_time_s": start,
                    "end_time_s": end,
                    "n_merged": 1,
                }
            )
    return merged


def process_file(annotated_path: Path, dry_run: bool = False) -> Path:
    data = json.loads(annotated_path.read_text())
    segments = merge_segments(data.get("segments", []))

    out = {
        "video_path": data.get("video_path"),
        "fps": data.get("fps"),
        "n_segments": len(segments),
        "segments": segments,
    }

    out_path = annotated_path.parent / "step_segments.json"
    if not dry_run:
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true", help="Don't write output files")
    args = parser.parse_args()

    annotated_files = sorted(args.data_dir.rglob("action_segments_annotated.json"))
    if not annotated_files:
        print(f"No action_segments_annotated.json files found under {args.data_dir}")
        return

    for path in annotated_files:
        data = json.loads(path.read_text())
        n_before = len(data.get("segments", []))
        n_after = len(merge_segments(data.get("segments", [])))
        out_path = process_file(path, dry_run=args.dry_run)
        print(f"{path} : {n_before} -> {n_after} segments -> {out_path}")


if __name__ == "__main__":
    main()
