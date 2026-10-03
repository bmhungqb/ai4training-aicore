#!/usr/bin/env python3
"""Build a TAS (Temporal Action Segmentation) dataset from `step_segments_clean.json`.

Produces the standard MS-TCN / ASFormer style layout (minus pre-extracted
features, which are intentionally left out per request — only videos +
per-frame ground-truth labels are generated here):

    <out-dir>/mapping.txt                  # "<id> <class_name>" per line
    <out-dir>/groundTruth/<video_id>.txt   # one coarse-category label per frame
    <out-dir>/videos/<video_id>.mp4        # symlink to source video
    <out-dir>/splits/train.bundle          # "<video_id>.txt" per line (train)
    <out-dir>/splits/val.bundle            # "<video_id>.txt" per line (val)

Label granularity: the 4 coarse categories from
`data/sheets/operation_duration_stats_lvl2_grouped_categorized.csv`:
    1. Sewing/Joining
    2. Positioning/Handling
    3. Adjustment/Alignment/Preparation
    4. Inspection/Auxiliary
    + background (noisy/unknown/catch-all segments, and any uncovered frames)

Source annotations: `data/cd*/chuyen*/step_segments_clean.json` (merged
consecutive same-name segments, unknown/empty already dropped upstream).

Frame rate: each video's own native fps as recorded in step_segments_clean.json
(no resampling) — frame count comes from ffprobe on the source .mp4.

Train/val split: reuses the exact same split as
`data/efficient_gebd_dataset/{train,val}_annotation.pkl` (by_folder holdout,
cd18-20 in val, consistent with diff_gebd_dataset) so all pipelines in this
repo train/evaluate on the same videos.

Usage:
    python tools/prepare_tas_dataset.py
    python tools/prepare_tas_dataset.py --out-dir dataset_tas --overwrite
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------- #
# Operation name -> coarse category classifier
# (mirrors the rules used to build operation_duration_stats_lvl2_grouped_categorized.csv)
# --------------------------------------------------------------------------- #
CATEGORIES = [
    "background",
    "Sewing/Joining",
    "Positioning/Handling",
    "Adjustment/Alignment/Preparation",
    "Inspection/Auxiliary",
]

# Noisy / catch-all / error labels (see dataset_insight.md section 2.1) -> background
NOISY_NAMES = {
    "may sửa lỗi", "sửa lỗi", "công đoạn khác", "chưa tới chu kì",
    "bỏ qua đoạn bị giật", "lấy lót túi lai lớn", "lỗi",
}

AUX_EXACT = {
    "bỏ", "cắt chỉ", "nhìn số", "nhìn kiểm tra", "nhấn nút", "bấm nút",
    "lấy kéo", "lấy phấn", "đưa kéo ra", "đưa phấn ra",
    "thao tách thừa", "lấy rập", "mở rập", "đóng rập", "chỉnh rập", "xoay rập",
    "đưa rập ra", "đưa rập vào chân vịt", "dời đô sau để kiểm",
    "lộn vai + kiểm tra", "mở miệng túi + kiểm tra",
}
AUX_KEYWORDS = ["kiểm", "nhìn"]
SEW_KEYWORDS = ["tra ", "tra", "may", "diễu", "ráp", "ghép", "dán", "đính",
                "khóa", "lại mũi", "gắn", "câu tay", "câu túi lót", "cài"]
ADJ_KEYWORDS = ["điều chỉnh", "chỉnh", "vuốt", "canh", "so mép", "so dây", "so ",
                "gấp", "bấm", "rẽ mép", "vén", "chấm", "doup", "vỗ"]
POS_KEYWORDS = ["lấy", "đưa", "lật", "xoay", "dời", "đặt", "kéo", "lộn", "cầm",
                "luồn", "xỏ", "treo", "trãi", "đẩy", "hất", "mở", "đóng", "hạ", "nâng", "quấn"]

# Manual overrides for phrases where a keyword matches mid-sentence out of context
OVERRIDES = {
    "đưa ra sau may": "Positioning/Handling",
    "đưa túi ra sau may": "Positioning/Handling",
    "đưa tay ra sau may": "Positioning/Handling",
    "đưa cổ ra sau may": "Positioning/Handling",
    "dời thân sau để ráp": "Positioning/Handling",
    "lấy nách để gắn dây câu": "Positioning/Handling",
    "lấy dây kéo ghép vào lót túi": "Positioning/Handling",
    "điều chỉnh khóa lưỡi gà": "Adjustment/Alignment/Preparation",
    "điều chỉnh khóa đầu bao túi": "Adjustment/Alignment/Preparation",
}


def is_unknown(name: str | None) -> bool:
    return not name or not name.strip() or name.strip().upper() in ["UNKNOWN", "NONE"]


def strip_suffix(name: str) -> str:
    """Strip trailing "( đoạn N )" style segment-index annotations."""
    return re.sub(r"\(\s*đoạn\s*\d+\s*\)", "", name, flags=re.IGNORECASE).strip()


def classify(name: str | None) -> str:
    if is_unknown(name):
        return "background"
    name = strip_suffix(name)
    low = name.lower()
    if low in NOISY_NAMES:
        return "background"
    if low in OVERRIDES:
        return OVERRIDES[low]
    if low in AUX_EXACT:
        return "Inspection/Auxiliary"
    for kw in AUX_KEYWORDS:
        if kw in low:
            return "Inspection/Auxiliary"
    for kw in SEW_KEYWORDS:
        if low.startswith(kw) or (" " + kw) in low:
            return "Sewing/Joining"
    for kw in ADJ_KEYWORDS:
        if low.startswith(kw) or (" " + kw.strip()) in (" " + low):
            return "Adjustment/Alignment/Preparation"
    for kw in POS_KEYWORDS:
        if low.startswith(kw):
            return "Positioning/Handling"
    return "background"


# --------------------------------------------------------------------------- #
# Video discovery / split (reuse exact split from data/efficient_gebd_dataset)
# --------------------------------------------------------------------------- #
def find_videos(data_dir: Path) -> list[Path]:
    return sorted(p.parent for p in data_dir.rglob("step_segments_clean.json"))


def video_unique_id(video_dir: Path) -> str:
    return f"{video_dir.parent.name}_{video_dir.name}"


def find_mp4(video_dir: Path) -> Path | None:
    mp4s = list(video_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def load_reference_split(ref_pkl_dir: Path) -> tuple[set[str], set[str]] | None:
    train_pkl = ref_pkl_dir / "train_annotation.pkl"
    val_pkl = ref_pkl_dir / "val_annotation.pkl"
    if not (train_pkl.exists() and val_pkl.exists()):
        return None
    with open(train_pkl, "rb") as f:
        train_ids = set(pickle.load(f).keys())
    with open(val_pkl, "rb") as f:
        val_ids = set(pickle.load(f).keys())
    return train_ids, val_ids


# --------------------------------------------------------------------------- #
# Frame count via ffprobe
# --------------------------------------------------------------------------- #
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
        print(f"WARNING: ffprobe frame count failed for {video_path}: {e}")
    if duration_fallback:
        return max(1, round(duration_fallback * fps))
    return 0


# --------------------------------------------------------------------------- #
# Per-frame ground truth labels
# --------------------------------------------------------------------------- #
def build_ground_truth(step_segments: dict, n_frames: int, fps: float,
                        split_repeated: bool = False) -> list[str]:
    """
    Per-frame coarse-category labels.

    If `split_repeated` is False (default/legacy behavior): consecutive
    fine-grained segments that map to the same coarse category are written
    with the *same* label string, so they end up indistinguishable in the
    output (two separate real operations collapse into one long segment --
    see dataset_insight / the TAS README "Known issue" section).

    If `split_repeated` is True: whenever a fine-grained segment maps to
    the *same* coarse category as the immediately preceding fine-grained
    segment, its label gets an alternating `#A`/`#B` suffix instead (never
    the same suffix as the previous segment). This means two consecutive
    real operations of the same category can never produce the same label
    string, so the classifier itself is forced to learn the boundary
    between them directly -- no separate boundary channel/head needed.
    This doubles the number of classes (5 -> 10) but does NOT blow up to
    fine-grained per-operation classes (~60), keeping the label space
    learnable on a 44-video dataset.
    """
    labels = ["background"] * n_frames
    prev_cat = None
    parity = "A"
    for seg in step_segments.get("segments", []):
        cat = classify(seg.get("operation_name"))

        if split_repeated and cat != "background":
            # only real operation categories get the #A/#B parity split;
            # "background" (noisy/catch-all/uncovered gaps) is always
            # written as the plain string "background" so it doesn't matter
            # whether a frame got there via the initial fill or via a
            # classify()-returned "background" segment.
            if cat == prev_cat:
                parity = "B" if parity == "A" else "A"
            else:
                parity = "A"
            out_label = f"{cat}#{parity}"
        else:
            out_label = cat

        if cat != "background":
            prev_cat = cat
        else:
            prev_cat = None  # a background gap always resets parity for the next real segment

        start_f = max(0, round(float(seg["start_time_s"]) * fps))
        end_f = min(n_frames, round(float(seg["end_time_s"]) * fps) + 1)  # inclusive end frame
        for i in range(start_f, end_f):
            labels[i] = out_label
    return labels


def strip_parity_suffix(label: str) -> str:
    """Inverse of the `#A`/`#B` suffixing in build_ground_truth(split_repeated=True)
    -- recovers the plain coarse category, e.g. for computing the standard
    5-class F1@IoU metric against predictions made in the 10-class space."""
    return label.split("#", 1)[0]


def build_instance_boundaries(step_segments: dict, n_frames: int, fps: float, tolerance: int = 2) -> list[int]:
    """Per-frame binary boundary labels from the *fine-grained* operation
    segments (every `step_segments_clean.json` transition, regardless of
    whether the coarse category changes across it).

    A boundary frame is the first frame of each fine-grained segment
    (including frame 0). The label is widened by `tolerance` frames on
    each side (set via max, no overlap merging needed since 1s are
    idempotent) to give the boundary-detection head a few frames of
    slack, following common GEBD/ASRF practice.
    """
    boundary = [0] * n_frames
    segs = step_segments.get("segments", [])
    starts = [0]  # video start is always a boundary
    for seg in segs:
        starts.append(max(0, round(float(seg["start_time_s"]) * fps)))
    for s in starts:
        lo = max(0, s - tolerance)
        hi = min(n_frames, s + tolerance + 1)
        for i in range(lo, hi):
            boundary[i] = 1
    return boundary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset_tas"))
    parser.add_argument("--ref-split-dir", type=Path, default=Path("data/efficient_gebd_dataset"),
                         help="Directory with train/val_annotation.pkl to reuse the exact same video split")
    parser.add_argument("--overwrite", action="store_true", help="Re-write existing groundTruth/video/boundary files")
    parser.add_argument("--boundary-tolerance", type=int, default=2,
                         help="Widen each instance-boundary label by +-N frames")
    parser.add_argument("--split-repeated", action="store_true",
                         help="Fix (at the dataset level): when 2+ consecutive fine-grained operations map to "
                              "the same coarse category, alternate an #A/#B suffix on the coarse label so they "
                              "are never written as the same groundTruth string. Doubles num_classes (5 -> 10). "
                              "Forces the classifier itself to learn these boundaries directly -- no separate "
                              "boundary head/channel needed. See strip_parity_suffix() for recovering the plain "
                              "5-class label at eval time.")
    args = parser.parse_args()

    video_dirs = find_videos(args.data_dir)
    if not video_dirs:
        print(f"No step_segments_clean.json found under {args.data_dir}.")
        return

    ref_split = load_reference_split(args.ref_split_dir)
    if ref_split is None:
        print(f"WARNING: no reference split found at {args.ref_split_dir}, falling back to by_folder cd18-20 holdout")
        val_folders = {"cd18", "cd19", "cd20"}
        train_ids = None
        val_ids = None
    else:
        train_ids, val_ids = ref_split
        val_folders = None

    out_dir = args.out_dir
    gt_dir = out_dir / "groundTruth"
    boundary_dir = out_dir / "boundaries"
    videos_dir = out_dir / "videos"
    splits_dir = out_dir / "splits"
    for d in (gt_dir, boundary_dir, videos_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_bundle, val_bundle, skipped = [], [], []

    for video_dir in video_dirs:
        vid = video_unique_id(video_dir)
        mp4_path = find_mp4(video_dir)
        if mp4_path is None:
            print(f"WARNING: no .mp4 found in {video_dir}, skipping")
            skipped.append(vid)
            continue

        step_segments = json.loads((video_dir / "step_segments_clean.json").read_text())
        segs = step_segments.get("segments", [])
        if not segs:
            print(f"WARNING: no segments in {video_dir}, skipping")
            skipped.append(vid)
            continue

        fps = float(step_segments.get("fps") or 25.0)
        last_end = max(float(s["end_time_s"]) for s in segs)
        n_frames = ffprobe_nb_frames(mp4_path, fps, duration_fallback=last_end)
        if n_frames <= 0:
            print(f"WARNING: could not determine frame count for {mp4_path}, skipping")
            skipped.append(vid)
            continue

        gt_path = gt_dir / f"{vid}.txt"
        if gt_path.exists() and not args.overwrite:
            pass
        else:
            labels = build_ground_truth(step_segments, n_frames, fps, split_repeated=args.split_repeated)
            gt_path.write_text("\n".join(labels) + "\n")

        boundary_path = boundary_dir / f"{vid}.txt"
        if boundary_path.exists() and not args.overwrite:
            pass
        else:
            boundaries = build_instance_boundaries(step_segments, n_frames, fps, tolerance=args.boundary_tolerance)
            boundary_path.write_text("\n".join(str(b) for b in boundaries) + "\n")

        link_path = videos_dir / f"{vid}.mp4"
        if link_path.exists() or link_path.is_symlink():
            if args.overwrite:
                link_path.unlink()
        if not link_path.exists() and not link_path.is_symlink():
            link_path.symlink_to(mp4_path.resolve())

        # split assignment
        if train_ids is not None:
            if vid in train_ids:
                train_bundle.append(vid)
            elif vid in val_ids:
                val_bundle.append(vid)
            else:
                # Newly annotated video: assign to train
                train_bundle.append(vid)
        else:
            if video_dir.parent.name in val_folders:
                val_bundle.append(vid)
            else:
                train_bundle.append(vid)

    # mapping.txt
    mapping_path = out_dir / "mapping.txt"
    if args.split_repeated:
        # "background" is always written plain (no #A/#B -- see
        # build_ground_truth()), only the 4 real operation categories get
        # both parity variants.
        real_cats = [c for c in CATEGORIES if c != "background"]
        mapping_classes = ["background"] + [f"{c}#A" for c in real_cats] + [f"{c}#B" for c in real_cats]
    else:
        mapping_classes = CATEGORIES
    mapping_path.write_text("\n".join(f"{i} {c}" for i, c in enumerate(mapping_classes)) + "\n")

    # split bundles
    (splits_dir / "train.bundle").write_text("\n".join(f"{v}.txt" for v in sorted(train_bundle)) + "\n")
    (splits_dir / "val.bundle").write_text("\n".join(f"{v}.txt" for v in sorted(val_bundle)) + "\n")

    print(f"Videos found: {len(video_dirs)}")
    print(f"  train: {len(train_bundle)}")
    print(f"  val:   {len(val_bundle)}")
    print(f"  skipped: {len(skipped)} {skipped if skipped else ''}")
    print(f"Wrote mapping.txt -> {mapping_path}")
    print(f"Wrote groundTruth/*.txt -> {gt_dir}")
    print(f"Wrote boundaries/*.txt -> {boundary_dir} (fine-grained instance boundaries, 0/1 per frame, tolerance=+-{args.boundary_tolerance})")
    print(f"Wrote videos/*.mp4 (symlinks) -> {videos_dir}")
    print(f"Wrote splits/{{train,val}}.bundle -> {splits_dir}")


if __name__ == "__main__":
    main()
