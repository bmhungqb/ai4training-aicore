#!/usr/bin/env python3
"""Extract per-frame features for `dataset/` using DINOv2 (ViT-B/14, 768-dim
CLS token), run FRAME-BY-FRAME (one 2D ViT forward pass per frame) -- as
opposed to `extract_chunk_features.py`, which runs a temporal/3D backbone
over a `clip_len`-frame window per output frame.

ROI mask cropping (on by default, same convention as extract_chunk_features.py):
`dataset/videos/<video_id>.mp4` is a symlink into
`original_data/processed_data/<cd>/<chuyen>/`, where each source video has
one shared `<source_video>.mask.png` (binary 0/255, same resolution as the
video, marking the relevant sewing-station ROI). Before resizing to 224x224,
every decoded frame is cropped to the bounding box of that mask's nonzero
region. Pass --no-mask-crop to disable this and use the full frame instead.

Output: <dataset>/features_dinov2/<video_id>.npy, shape (768, T) float32,
Fortran order -- consistent with features_videomae/features_s3d/etc.

T is taken from groundTruth/<video_id>.txt line count (authoritative) so
features always align with frame-wise labels. Frames are processed in a
STREAMING fashion (no full video buffered in RAM) to avoid OOM on long
videos. cv2 frame-drop near EOF is handled by seek-based recovery +
last-vector padding.

Usage:
    python src/TAS-instance-style/scripts/extract_frame_features_dinov2.py
    python src/TAS-instance-style/scripts/extract_frame_features_dinov2.py --video-ids cd9_chuyen3_part1
    python src/TAS-instance-style/scripts/extract_frame_features_dinov2.py --overwrite
    python src/TAS-instance-style/scripts/extract_frame_features_dinov2.py --no-mask-crop
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = REPO_ROOT / "src" / "TAS-instance-style" / "dataset"
DEFAULT_VID_DIR = DATASET_DIR / "videos"
DEFAULT_GT_DIR = DATASET_DIR / "groundTruth"
DEFAULT_FEAT_DIR = DATASET_DIR / "features_dinov2"


# ---------------------------------------------------------------------------
# Frame-count helper (authoritative T comes from groundTruth)
# ---------------------------------------------------------------------------

def gt_frame_count(gt_dir: Path, vid_id: str) -> int | None:
    gt = gt_dir / f"{vid_id}.txt"
    if not gt.exists():
        return None
    return len(gt.read_text().strip().splitlines())


# ---------------------------------------------------------------------------
# ROI mask bbox (same convention as extract_chunk_features.py)
# ---------------------------------------------------------------------------

def resolve_mask_bbox(mp4_path: Path) -> tuple[int, int, int, int] | None:
    """Find the shared `<source_video>.mask.png` next to the (symlink-resolved)
    source video for `mp4_path`, and return its nonzero region's bounding box
    as (x0, y0, x1, y1) (x1/y1 exclusive), or None if no mask is found.
    """
    real_path = mp4_path.resolve()
    mask_candidates = sorted(real_path.parent.glob("*.mask.png"))
    if not mask_candidates:
        return None
    mask = cv2.imread(str(mask_candidates[0]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# DINOv2 backbone
# ---------------------------------------------------------------------------

def _imagenet_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_dinov2(device: torch.device):
    print("  Loading DINOv2 ViT-B/14 (downloads ~330 MB on first run) ...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model = model.to(device).eval()
    tf = _imagenet_transform(224)

    def infer(batch: torch.Tensor) -> list[np.ndarray]:
        with torch.no_grad():
            out = model(batch)  # (B, 768) CLS token
        return list(out.cpu().numpy())

    return tf, infer, 768


# ---------------------------------------------------------------------------
# Extraction -- STREAMING, one frame -> one feature vector (no clip window)
# ---------------------------------------------------------------------------

def extract_frames_streaming(
    mp4_path: Path,
    expected_T: int | None,
    tf: transforms.Compose,
    infer,
    device: torch.device,
    batch_size: int,
    mask_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Stream frames one batch at a time (safe for very long videos), run a
    single-frame 2D ViT forward pass per frame, optionally cropping to
    `mask_bbox` = (x0, y0, x1, y1) BEFORE the backbone's own resize/crop
    transform. Returns (C, T) float32, Fortran order.
    """
    def crop(rgb: np.ndarray) -> np.ndarray:
        if mask_bbox is None:
            return rgb
        x0, y0, x1, y1 = mask_bbox
        return rgb[y0:y1, x0:x1]

    cap = cv2.VideoCapture(str(mp4_path))

    feats: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        feats.extend(infer(x))
        batch.clear()

    while True:
        if expected_T is not None and (len(feats) + len(batch)) >= expected_T:
            break
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch.append(tf(crop(rgb)))
        if len(batch) >= batch_size:
            flush()
    flush()

    # seek-based recovery for codec-dropped tail frames
    if expected_T is not None and len(feats) < expected_T:
        n_missing = expected_T - len(feats)
        recovered = 0
        for idx in range(len(feats), expected_T):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(tf(crop(rgb)))
            recovered += 1
            if len(batch) >= batch_size:
                flush()
        flush()
        if recovered:
            print(f"    recovered {recovered}/{n_missing} tail frames via seek")

    cap.release()

    if not feats:
        raise RuntimeError(f"No frames decoded from {mp4_path}")

    if expected_T is not None and len(feats) < expected_T:
        pad = expected_T - len(feats)
        print(f"    padding {pad} frame(s) with last feature vector")
        last = feats[-1]
        feats.extend([last] * pad)
    elif expected_T is not None and len(feats) > expected_T:
        feats = feats[:expected_T]

    arr = np.stack(feats, axis=1).astype(np.float32)  # (C, T)
    return np.asfortranarray(arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--vid-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--feat-dir", type=Path, default=None)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mask-crop", action="store_true",
                        help="Disable ROI mask cropping; use the full frame (old behavior)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Frames per GPU batch — reduce if OOM")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    vid_dir = args.vid_dir or (args.dataset_dir / "videos")
    gt_dir = args.gt_dir or (args.dataset_dir / "groundTruth")
    feat_dir = args.feat_dir or DEFAULT_FEAT_DIR
    feat_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    tf, infer, feat_dim = build_dinov2(device)
    print(f"Backbone : dinov2 (frame-by-frame, dim={feat_dim})")
    print(f"Device   : {device}")
    print(f"Feat dir : {feat_dir}")

    gt_ids = {f[:-4] for f in os.listdir(gt_dir) if f.endswith(".txt")}
    feat_ids = {f[:-4] for f in os.listdir(feat_dir) if f.endswith(".npy")}

    if args.video_ids:
        targets = sorted(args.video_ids)
    elif args.overwrite:
        targets = sorted(gt_ids)
    else:
        targets = sorted(gt_ids - feat_ids)

    print(f"\nVideos total : {len(gt_ids)}")
    print(f"Already have : {len(feat_ids)}")
    print(f"To extract   : {len(targets)}")

    if not targets:
        print("Nothing to do. Use --overwrite to re-extract.")
        return

    for i, vid_id in enumerate(targets, 1):
        out = feat_dir / f"{vid_id}.npy"
        if out.exists() and not args.overwrite:
            print(f"[{i}/{len(targets)}] SKIP {vid_id} — already exists")
            continue

        mp4 = vid_dir / f"{vid_id}.mp4"
        if not mp4.exists():
            print(f"[{i}/{len(targets)}] SKIP {vid_id} — video not found")
            continue

        expected_T = gt_frame_count(gt_dir, vid_id)
        mask_bbox = None if args.no_mask_crop else resolve_mask_bbox(mp4)
        bbox_info = f", crop={mask_bbox}" if mask_bbox else ", crop=none (full frame)"
        print(f"[{i}/{len(targets)}] {vid_id}  (expected T={expected_T}{bbox_info}) ...", flush=True)

        try:
            arr = extract_frames_streaming(mp4, expected_T, tf, infer, device, args.batch_size, mask_bbox=mask_bbox)
            np.save(out, arr)
            print(f"    -> saved  shape={arr.shape}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
