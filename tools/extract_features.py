#!/usr/bin/env python3
"""Extract per-frame features for videos in dataset_tas_instance/features/.

Supported backbones:
  resnet50  — ResNet-50 avgpool (2048-dim). Default. No extra deps.
  dinov2    — DINOv2 ViT-B/14 CLS token (768-dim). Better generalisation.
              Requires: internet (first run downloads ~330 MB weights).
  videomae  — VideoMAE-Base (768-dim). Best temporal understanding.
              Requires: pip install transformers

Output shape: (C, T) float32, Fortran order — consistent with existing .npy files.

T is taken from groundTruth/<video_id>.txt line count (authoritative) so features
always align with frame-wise labels. cv2 frame-drop is handled by seek-based
recovery + last-vector padding. All frames are processed in a STREAMING fashion
(no full video buffered in RAM) to avoid OOM on long videos.

Usage:
    python tools/extract_features.py                          # all missing, resnet50
    python tools/extract_features.py --backbone dinov2        # DINOv2
    python tools/extract_features.py --backbone videomae      # VideoMAE (pip install transformers)
    python tools/extract_features.py --fix-mismatches         # fix existing broken .npy
    python tools/extract_features.py --video-ids cd11_chuyen1 cd16_chuyen2
    python tools/extract_features.py --backbone dinov2 \\
        --feat-dir src/TAS-instance-style/dataset_tas_instance/features_dinov2
    python tools/extract_features.py --overwrite
"""
from __future__ import annotations

import argparse
import os
import re
import struct
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VID_DIR  = REPO_ROOT / "src" / "TAS-instance-style" / "dataset_tas_instance" / "videos"
DEFAULT_FEAT_DIR = REPO_ROOT / "src" / "TAS-instance-style" / "dataset_tas_instance" / "features"
DEFAULT_GT_DIR   = REPO_ROOT / "src" / "TAS-instance-style" / "dataset_tas_instance" / "groundTruth"


# ---------------------------------------------------------------------------
# Frame-count helpers
# ---------------------------------------------------------------------------

def gt_frame_count(gt_dir: Path, vid_id: str) -> int | None:
    gt = gt_dir / f"{vid_id}.txt"
    if not gt.exists():
        return None
    return len(gt.read_text().strip().splitlines())


def npy_frame_count(npy_path: Path) -> int:
    with open(npy_path, "rb") as f:
        f.read(6); f.read(2)
        hlen = struct.unpack("<H", f.read(2))[0]
        hdr  = f.read(hlen).decode("latin1")
    m = re.search(r"'shape':\s*\((\d+),\s*(\d+)\)", hdr)
    return int(m.group(2)) if m else -1


def find_mismatches(feat_dir: Path, gt_dir: Path) -> list[tuple]:
    out = []
    for npy in sorted(os.listdir(feat_dir)):
        if not npy.endswith(".npy"):
            continue
        vid = npy[:-4]
        T_gt = gt_frame_count(gt_dir, vid)
        if T_gt is None:
            continue
        T_npy = npy_frame_count(feat_dir / npy)
        if T_npy != T_gt:
            out.append((vid, T_npy, T_gt))
    return out


# ---------------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------------

def _imagenet_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_resnet50(device: torch.device):
    from torchvision import models
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device).eval()
    tf = _imagenet_transform(224)

    def infer(batch: torch.Tensor) -> list[np.ndarray]:
        with torch.no_grad():
            out = model(batch).squeeze(-1).squeeze(-1)  # (B, 2048)
        return list(out.cpu().numpy())

    return tf, infer, 2048


def build_dinov2(device: torch.device):
    print("  Loading DINOv2 ViT-B/14 (downloads ~330 MB on first run) ...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model = model.to(device).eval()
    tf = _imagenet_transform(224)

    def infer(batch: torch.Tensor) -> list[np.ndarray]:
        with torch.no_grad():
            out = model(batch)   # (B, 768)
        return list(out.cpu().numpy())

    return tf, infer, 768


def build_videomae(device: torch.device):
    try:
        from transformers import VideoMAEModel, VideoMAEImageProcessor
    except ImportError:
        raise ImportError("Run: pip install transformers")
    ckpt = "MCG-NJU/videomae-base"
    print(f"  Loading VideoMAE ({ckpt}) ...")
    processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    model = VideoMAEModel.from_pretrained(ckpt).to(device).eval()
    CLIP_LEN = 16

    def infer_clip(clip_frames_rgb: list) -> np.ndarray:
        inputs = processor(images=clip_frames_rgb, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        with torch.no_grad():
            out = model(pixel_values=pv)
        return out.last_hidden_state.squeeze(0).mean(0).cpu().numpy()  # (768,)

    return processor, infer_clip, 768, CLIP_LEN


# ---------------------------------------------------------------------------
# Extraction — STREAMING for resnet50 / dinov2 (no full-video RAM buffer)
# ---------------------------------------------------------------------------

def extract_streaming(
    mp4_path: Path,
    expected_T: int | None,
    tf: transforms.Compose,
    infer: Callable,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Stream frames one batch at a time — safe for very long videos (no OOM).

    If cv2 drops frames near EOF (codec bug), uses seek-based recovery.
    If still short, pads with the last valid feature vector.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    feats: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        feats.extend(infer(x))
        batch.clear()

    # ---- normal sequential read ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch.append(tf(rgb))   # only 3×224×224 float stored, not full-res
        if len(batch) >= batch_size:
            flush()
    flush()

    # ---- seek-based recovery for codec-dropped tail frames ----
    if expected_T is not None and len(feats) < expected_T:
        n_missing = expected_T - len(feats)
        recovered = 0
        for idx in range(len(feats), expected_T):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(tf(rgb))
            recovered += 1
            if len(batch) >= batch_size:
                flush()
        flush()
        if recovered:
            print(f"    recovered {recovered}/{n_missing} tail frames via seek")

    cap.release()

    if not feats:
        raise RuntimeError(f"No frames decoded from {mp4_path}")

    # ---- pad remaining gap with last feature vector ----
    if expected_T is not None and len(feats) < expected_T:
        pad = expected_T - len(feats)
        print(f"    ⚠ padding {pad} frame(s) with last feature vector")
        last = feats[-1]
        feats.extend([last] * pad)
    elif expected_T is not None and len(feats) > expected_T:
        feats = feats[:expected_T]

    arr = np.stack(feats, axis=1).astype(np.float32)   # (C, T)
    return np.asfortranarray(arr)


# ---------------------------------------------------------------------------
# Extraction — VideoMAE (needs random access, but stores small resized frames)
# ---------------------------------------------------------------------------

def extract_videomae(
    mp4_path: Path,
    expected_T: int | None,
    infer_clip: Callable,
    clip_len: int,
    resize: int = 224,
) -> np.ndarray:
    """Read all frames resized to `resize×resize` (saves ~10–50× RAM vs full-res),
    then run sliding-window VideoMAE per frame.
    """
    # ---- read all frames at small size to stay in RAM ----
    cap = cv2.VideoCapture(str(mp4_path))
    frames: list[np.ndarray] = []   # each (resize, resize, 3) uint8 — ~150 KB not ~6 MB
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize to save RAM (VideoMAE processor will resize again to its own size)
        small = cv2.resize(rgb, (resize, resize), interpolation=cv2.INTER_LINEAR)
        frames.append(small)

    # seek recovery for dropped tail frames
    if expected_T is not None and len(frames) < expected_T:
        for idx in range(len(frames), expected_T):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (resize, resize)))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {mp4_path}")

    T = len(frames)
    half = clip_len // 2
    feats: list[np.ndarray] = []
    for t in range(T):
        indices = [max(0, min(T - 1, t - half + k)) for k in range(clip_len)]
        clip = [frames[i] for i in indices]
        feats.append(infer_clip(clip))

    # pad / trim
    if expected_T is not None and len(feats) < expected_T:
        pad = expected_T - len(feats)
        print(f"    ⚠ padding {pad} frame(s)")
        feats.extend([feats[-1]] * pad)
    elif expected_T is not None and len(feats) > expected_T:
        feats = feats[:expected_T]

    arr = np.stack(feats, axis=1).astype(np.float32)
    return np.asfortranarray(arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=["resnet50", "dinov2", "videomae"],
                        default="resnet50")
    parser.add_argument("--vid-dir",  type=Path, default=DEFAULT_VID_DIR)
    parser.add_argument("--feat-dir", type=Path, default=DEFAULT_FEAT_DIR)
    parser.add_argument("--gt-dir",   type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--overwrite",      action="store_true")
    parser.add_argument("--fix-mismatches", action="store_true",
                        help="Re-extract only videos where npy T != groundTruth T")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Frames per GPU batch — reduce to 4 or 2 if OOM")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    args.feat_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---- build backbone ----
    clip_len = None
    if args.backbone == "resnet50":
        tf, infer, feat_dim = build_resnet50(device)
    elif args.backbone == "dinov2":
        tf, infer, feat_dim = build_dinov2(device)
    elif args.backbone == "videomae":
        _, infer_clip, feat_dim, clip_len = build_videomae(device)
        tf = infer = None
    else:
        raise ValueError(args.backbone)

    print(f"Backbone : {args.backbone}  (dim={feat_dim})")
    print(f"Device   : {device}")
    print(f"Feat dir : {args.feat_dir}")

    # ---- discover targets ----
    vid_ids  = {f[:-4] for f in os.listdir(args.vid_dir)  if f.endswith(".mp4")}
    feat_ids = {f[:-4] for f in os.listdir(args.feat_dir) if f.endswith(".npy")}

    if args.video_ids:
        targets = sorted(args.video_ids)
    elif args.fix_mismatches:
        mm = find_mismatches(args.feat_dir, args.gt_dir)
        if not mm:
            print("No frame-count mismatches — all .npy files are correct ✓")
            return
        print(f"Frame-count mismatches ({len(mm)}):")
        for vid, T_npy, T_gt in mm:
            print(f"  {vid:35s}  npy={T_npy}  gt={T_gt}  diff={T_npy-T_gt:+d}")
        targets = [vid for vid, _, _ in mm]
    elif args.overwrite:
        targets = sorted(vid_ids)
    else:
        targets = sorted(vid_ids - feat_ids)

    print(f"\nVideos total : {len(vid_ids)}")
    print(f"Already have : {len(feat_ids)}")
    print(f"To extract   : {len(targets)}")

    if not targets:
        print("Nothing to do. Use --overwrite or --fix-mismatches.")
        return

    for i, vid_id in enumerate(targets, 1):
        mp4 = args.vid_dir / f"{vid_id}.mp4"
        out = args.feat_dir / f"{vid_id}.npy"

        if not mp4.exists():
            print(f"[{i}/{len(targets)}] SKIP {vid_id} — mp4 not found")
            continue
        if out.exists() and not args.overwrite and not args.fix_mismatches:
            print(f"[{i}/{len(targets)}] SKIP {vid_id} — already exists")
            continue

        expected_T = gt_frame_count(args.gt_dir, vid_id)
        print(f"[{i}/{len(targets)}] {vid_id}  (expected T={expected_T}) ...", flush=True)

        try:
            if args.backbone == "videomae":
                arr = extract_videomae(mp4, expected_T, infer_clip, clip_len)
            else:
                arr = extract_streaming(mp4, expected_T, tf, infer, device, args.batch_size)

            np.save(out, arr)
            print(f"    → saved  shape={arr.shape}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
