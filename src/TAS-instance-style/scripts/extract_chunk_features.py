#!/usr/bin/env python3
"""Extract per-frame features for `dataset/` using TEMPORAL (chunk/clip-based)
backbones -- as opposed to `tools/extract_features.py`, which (for resnet50 /
dinov2 / dinov3) runs a 2D image backbone on a single frame at a time.

For a video with T frames, this script still outputs one feature vector per
frame (so the result aligns 1:1 with `groundTruth/<video_id>.txt` /
`annotations/<video_id>.json`, exactly like the existing `features*/` dirs),
but each of the T vectors is produced by running a genuine spatio-temporal
("chunk") backbone over a short clip of `clip_len` consecutive frames
CENTERED on that frame index, instead of a single-frame 2D CNN/ViT. I.e.
T frames -> T chunks (sliding window, stride 1) -> T feature vectors.

Supported backbones (all run a real 3D/temporal encoder over the whole clip,
then global-average-pool to one vector per clip):
  videomae   -- VideoMAE-Base (768-dim). transformers. clip_len=16.
  mvit_v1_b  -- MViT-v1-B ("Base", 768-dim). torchvision.models.video, Kinetics-400
                pretrained. clip_len=16, 224x224. (torchvision has no mvit_v2_b;
                this is the closest available "Base"-sized MViT checkpoint.)
  s3d        -- S3D (1024-dim). torchvision.models.video, Kinetics-400 pretrained. clip_len=16, 224x224.
  i3d_r50    -- I3D ResNet-50 (2048-dim). pytorchvideo (`pip install pytorchvideo`),
                Kinetics-400 pretrained ([Carreira & Zisserman, 2017] architecture).
                clip_len=8, 224x224. First run downloads the checkpoint from
                dl.fbaipublicfiles.com.

Not included (not available locally without extra downloads/deps):
  InternVideo  -- HF checkpoints exist but are gated and/or need the official
                  InternVideo GitHub repo's custom modeling code (not a plain
                  AutoModel.from_pretrained() load); not installed here.
  VideoPrism   -- Google JAX model, no official PyTorch port.
These can be added later following the same `build_<name>()` -> (infer_clip,
feat_dim, clip_len, resize) pattern used below. See
src/TAS-instance-style/scripts/SETUP_INTERNVIDEO_VIDEOPRISM.md for manual
setup steps.

ROI mask cropping (on by default): `dataset/videos/<video_id>.mp4` is a
symlink into `original_data/processed_data/<cd>/<chuyen>/`, where each source
video has one shared `<source_video>.mask.png` (binary 0/255, same
resolution as the video, marking the relevant sewing-station ROI). Before
resizing to the backbone's input size, every decoded frame is cropped to the
bounding box of that mask's nonzero region -- so the backbone only ever sees
the cropped ROI, not full irrelevant background/borders. Pass
--no-mask-crop to disable this and use the full frame instead (old behavior).

Output: <dataset>/features_<backbone>/<video_id>.npy, shape (C, T) float32,
Fortran order -- consistent with features_videomae/features_dinov2/etc. in
dataset_tas_instance/.

Usage:
    python src/TAS-instance-style/scripts/extract_chunk_features.py --backbone mvit_v1_b
    python src/TAS-instance-style/scripts/extract_chunk_features.py --backbone videomae
    python src/TAS-instance-style/scripts/extract_chunk_features.py --backbone s3d --video-ids cd9_chuyen3_part1
    python src/TAS-instance-style/scripts/extract_chunk_features.py --backbone mvit_v1_b --overwrite
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = REPO_ROOT / "src" / "TAS-instance-style" / "dataset"
DEFAULT_VID_DIR = DATASET_DIR / "videos"
DEFAULT_GT_DIR = DATASET_DIR / "groundTruth"


# ---------------------------------------------------------------------------
# Frame-count helper (authoritative T comes from groundTruth, same convention
# as tools/extract_features.py)
# ---------------------------------------------------------------------------

def gt_frame_count(gt_dir: Path, vid_id: str) -> int | None:
    gt = gt_dir / f"{vid_id}.txt"
    if not gt.exists():
        return None
    return len(gt.read_text().strip().splitlines())


# ---------------------------------------------------------------------------
# Backbone builders.
# Each returns (infer_clip, feat_dim, clip_len, resize) where
#   infer_clip(clip: np.ndarray[clip_len, resize, resize, 3] uint8 RGB) -> np.ndarray[feat_dim]
# ---------------------------------------------------------------------------

def build_videomae(device: torch.device):
    try:
        from transformers import VideoMAEModel, VideoMAEImageProcessor
    except ImportError:
        raise ImportError("Run: pip install transformers")
    ckpt = "MCG-NJU/videomae-base"
    print(f"  Loading VideoMAE ({ckpt}) ...")
    processor = VideoMAEImageProcessor.from_pretrained(ckpt)
    model = VideoMAEModel.from_pretrained(ckpt).to(device).eval()
    clip_len = 16
    resize = 224

    def infer_clip(clip_frames_rgb: list) -> np.ndarray:
        inputs = processor(images=clip_frames_rgb, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        with torch.no_grad():
            out = model(pixel_values=pv)
        # mean-pool over all spatio-temporal tokens -> one vector per clip
        return out.last_hidden_state.squeeze(0).mean(0).cpu().numpy()  # (768,)

    return infer_clip, 768, clip_len, resize


def build_mvit_v1_b(device: torch.device):
    from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
    print("  Loading MViT-v1-B (Kinetics-400 pretrained) ...")
    weights = MViT_V1_B_Weights.KINETICS400_V1
    model = mvit_v1_b(weights=weights).to(device).eval()
    model.head = torch.nn.Identity()  # drop classifier -> pooled 768-dim feature
    tf = weights.transforms()
    clip_len = 16
    resize = 224

    def infer_clip(clip_frames_rgb: list) -> np.ndarray:
        # (T, H, W, C) uint8 -> (T, C, H, W) uint8 tensor, as expected by VideoClassification transform
        x = torch.from_numpy(np.stack(clip_frames_rgb)).permute(0, 3, 1, 2)
        x = tf(x).unsqueeze(0).to(device)  # (1, C, T, H, W)
        with torch.no_grad():
            out = model(x)
        return out.squeeze(0).cpu().numpy()  # (768,)

    return infer_clip, 768, clip_len, resize


def build_s3d(device: torch.device):
    from torchvision.models.video import s3d, S3D_Weights
    print("  Loading S3D (Kinetics-400 pretrained) ...")
    weights = S3D_Weights.KINETICS400_V1
    model = s3d(weights=weights).to(device).eval()
    tf = weights.transforms()
    clip_len = 16
    resize = 224

    def infer_clip(clip_frames_rgb: list) -> np.ndarray:
        x = torch.from_numpy(np.stack(clip_frames_rgb)).permute(0, 3, 1, 2)
        x = tf(x).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.features(x)
            pooled = model.avgpool(feat).flatten(1)
        return pooled.squeeze(0).cpu().numpy()  # (1024,)

    return infer_clip, 1024, clip_len, resize


def build_i3d_r50(device: torch.device):
    try:
        from pytorchvideo.models.hub import i3d_r50
    except ImportError:
        raise ImportError("Run: pip install pytorchvideo")
    print("  Loading I3D-R50 (pytorchvideo, Kinetics-400 pretrained) ...")
    model = i3d_r50(pretrained=True)
    # drop the classifier projection -> pooled 2048-dim feature (keep the
    # pool + output_pool steps of ResNetBasicHead, skip proj + activation)
    model.blocks[-1].proj = torch.nn.Identity()
    model.blocks[-1].activation = None
    model = model.to(device).eval()
    clip_len = 8
    resize = 224
    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)

    def infer_clip(clip_frames_rgb: list) -> np.ndarray:
        # (T, H, W, C) uint8 -> (1, C, T, H, W) float, ImageNet-style norm (SlowFast/I3D convention)
        x = torch.from_numpy(np.stack(clip_frames_rgb)).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        x = (x - mean) / std
        x = x.to(device)
        with torch.no_grad():
            out = model(x)
        return out.squeeze(0).cpu().numpy()  # (2048,)

    return infer_clip, 2048, clip_len, resize


BACKBONES = {
    "videomae": build_videomae,
    "mvit_v1_b": build_mvit_v1_b,
    "s3d": build_s3d,
    "i3d_r50": build_i3d_r50,
}


# ---------------------------------------------------------------------------
# T frames -> T chunks (sliding window, stride 1, centered) -> T feature vectors
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


def extract_chunks(
    mp4_path: Path,
    expected_T: int | None,
    infer_clip: Callable,
    clip_len: int,
    resize: int,
    mask_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Decode all frames (optionally cropped to `mask_bbox` = (x0, y0, x1, y1)
    BEFORE resizing, to keep RAM low), then for every frame index t build a
    `clip_len`-frame window centered on t (edges clamped / repeated), run the
    temporal backbone on that chunk, and use the resulting pooled vector as
    frame t's feature. Returns (C, T) float32.
    """
    def crop(rgb: np.ndarray) -> np.ndarray:
        if mask_bbox is None:
            return rgb
        x0, y0, x1, y1 = mask_bbox
        return rgb[y0:y1, x0:x1]

    cap = cv2.VideoCapture(str(mp4_path))
    frames: list[np.ndarray] = []
    while True:
        if expected_T is not None and len(frames) >= expected_T:
            break
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(crop(rgb), (resize, resize), interpolation=cv2.INTER_LINEAR))

    # seek-based recovery for codec-dropped tail frames
    if expected_T is not None and len(frames) < expected_T:
        for idx in range(len(frames), expected_T):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(crop(rgb), (resize, resize)))
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

    if expected_T is not None and len(feats) < expected_T:
        pad = expected_T - len(feats)
        print(f"    padding {pad} frame(s) with last feature vector")
        feats.extend([feats[-1]] * pad)
    elif expected_T is not None and len(feats) > expected_T:
        feats = feats[:expected_T]

    arr = np.stack(feats, axis=1).astype(np.float32)  # (C, T)
    return np.asfortranarray(arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="mvit_v1_b")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--vid-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--feat-dir", type=Path, default=None)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-mask-crop", action="store_true",
                        help="Disable ROI mask cropping; use the full frame (old behavior)")
    args = parser.parse_args()

    vid_dir = args.vid_dir or (args.dataset_dir / "videos")
    gt_dir = args.gt_dir or (args.dataset_dir / "groundTruth")
    feat_dir = args.feat_dir or (args.dataset_dir / f"features_{args.backbone}")
    feat_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    infer_clip, feat_dim, clip_len, resize = BACKBONES[args.backbone](device)
    print(f"Backbone : {args.backbone}  (dim={feat_dim}, clip_len={clip_len}, resize={resize})")
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
            arr = extract_chunks(mp4, expected_T, infer_clip, clip_len, resize, mask_bbox=mask_bbox)
            np.save(out, arr)
            print(f"    -> saved  shape={arr.shape}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
