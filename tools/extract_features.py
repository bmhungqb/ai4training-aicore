#!/usr/bin/env python3
"""Extract per-frame features for videos in dataset_tas_instance/features/.

Supported backbones:
  resnet50  — ResNet-50 avgpool (2048-dim). Default. No extra deps.
  dinov2    — DINOv2 ViT-B/14 CLS token (768-dim). Better generalisation.
              Requires: internet (first run downloads ~330 MB weights).
  dinov3    — DINOv3 ViT-B/16 CLS token (768-dim). Meta AI foundation model (Aug 2025).
              Supports: Hugging Face (facebook/dinov3-vitb16-pretrain-lvd1689m) or torch.hub.
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
    python tools/extract_features.py --backbone dinov3        # DINOv3 (ViT-B/16, 768-dim)
    python tools/extract_features.py --backbone videomae      # VideoMAE (pip install transformers)
    python tools/extract_features.py --fix-mismatches         # fix existing broken .npy
    python tools/extract_features.py --video-ids cd11_chuyen1 cd16_chuyen2
    python tools/extract_features.py --backbone dinov3 \
        --feat-dir src/TAS-instance-style/dataset_tas_instance/features_dinov3
    python tools/extract_features.py --overwrite
"""
from __future__ import annotations

import argparse
import json
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


def build_dinov3(device: torch.device, model_name: str | None = None):
    """Build DINOv3 ViT-B/16 (768-dim) backbone.
    Supports Hugging Face (facebook/dinov3-vitb16-pretrain-lvd1689m) or torch.hub fallback.
    """
    model_id = model_name or "facebook/dinov3-vitb16-pretrain-lvd1689m"
    print(f"  Loading DINOv3 ({model_id}) ...")
    tf = _imagenet_transform(224)

    err_hf = None
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id).to(device).eval()

        def infer_hf(batch: torch.Tensor) -> list[np.ndarray]:
            with torch.no_grad():
                out = model(pixel_values=batch.to(device))
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    feat = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    feat = out.last_hidden_state[:, 0, :]
                else:
                    feat = out[0][:, 0, :]
            return list(feat.cpu().numpy())

        return tf, infer_hf, 768
    except Exception as e:
        err_hf = e
        print(f"  [Notice] Hugging Face transformers load failed: {e}\n  Attempting torch.hub fallback...")

    try:
        hub_name = model_name if (model_name and not model_name.startswith("facebook/")) else "dinov3_vitb16"
        model = torch.hub.load("facebookresearch/dinov3", hub_name, pretrained=True, verbose=False)
        model = model.to(device).eval()

        def infer_hub(batch: torch.Tensor) -> list[np.ndarray]:
            with torch.no_grad():
                out = model(batch)
            return list(out.cpu().numpy())

        return tf, infer_hub, 768
    except Exception as err_hub:
        raise RuntimeError(
            f"Failed to load DINOv3 ({model_id}) via transformers ({err_hf}) and torch.hub ({err_hub}).\n"
            "Note: DINOv3 is a gated model. To download weights, please request access at "
            "https://huggingface.co/facebook/dinov3 and run `huggingface-cli login`, "
            "or pass a local checkpoint with --model-name /path/to/checkpoint."
        )


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
    start_frame: int = 0,
) -> np.ndarray:
    """Stream frames one batch at a time — safe for very long videos (no OOM).

    If start_frame > 0, seeks to start_frame before reading up to expected_T frames.
    If cv2 drops frames near EOF (codec bug), uses seek-based recovery.
    If still short, pads with the last valid feature vector.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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
        if expected_T is not None and (len(feats) + len(batch)) >= expected_T:
            break
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
        for idx in range(start_frame + len(feats), start_frame + expected_T):
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
        raise RuntimeError(f"No frames decoded from {mp4_path} (start_frame={start_frame})")

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
    start_frame: int = 0,
) -> np.ndarray:
    """Read all frames resized to `resize×resize` (saves ~10–50× RAM vs full-res),
    then run sliding-window VideoMAE per frame.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []   # each (resize, resize, 3) uint8 — ~150 KB not ~6 MB
    while True:
        if expected_T is not None and len(frames) >= expected_T:
            break
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (resize, resize), interpolation=cv2.INTER_LINEAR)
        frames.append(small)

    # seek recovery for dropped tail frames
    if expected_T is not None and len(frames) < expected_T:
        for idx in range(start_frame + len(frames), start_frame + expected_T):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (resize, resize)))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {mp4_path} (start_frame={start_frame})")

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
# Video resolution & offset helper
# ---------------------------------------------------------------------------

def get_part_start_frame(clip_id: str, parent_id: str) -> int:
    """Calculate the exact starting frame of clip_id within parent video using step_segments_clean.json."""
    if clip_id == parent_id or "_part" not in clip_id:
        return 0
    cd, chuyen = parent_id.split("_", 1)
    clean_p = REPO_ROOT / "data" / cd / chuyen / "step_segments_clean.json"
    if not clean_p.exists():
        return 0
    try:
        import importlib.util
        spec_inst = importlib.util.spec_from_file_location("prep_inst", REPO_ROOT / "tools" / "prepare_tas_instance_dataset.py")
        prep_inst = importlib.util.module_from_spec(spec_inst)
        spec_inst.loader.exec_module(prep_inst)

        with open(clean_p) as f:
            clean_data = json.load(f)
        fps = clean_data.get("fps", 15.0)
        instances = prep_inst.build_instances(clean_data, 100000, fps)
        runs, curr = [], []
        for inst in instances:
            if inst["class_id"] != 0:
                curr.append(inst)
            else:
                if curr:
                    runs.append(curr)
                    curr = []
        if curr:
            runs.append(curr)

        part_idx = int(clip_id.split("_part")[-1]) - 1
        if 0 <= part_idx < len(runs):
            return int(runs[part_idx][0]["start_frame"])
    except Exception:
        pass
    return 0


def is_valid_mp4(path: Path) -> bool:
    """Quickly check if an MP4 file has a valid 'moov' atom and is not truncated/corrupt."""
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(1024 * 1024)
            if b"moov" in head:
                return True
            f.seek(-min(path.stat().st_size, 1024 * 1024), 2)
            tail = f.read(1024 * 1024)
            return b"moov" in tail
    except Exception:
        return False


def resolve_video_and_offset(vid_id: str, vid_dir: Path) -> tuple[Path | None, int]:
    """Find video file and start frame offset for vid_id (including sliced parts)."""
    direct = vid_dir / f"{vid_id}.mp4"
    if direct.exists() and is_valid_mp4(direct):
        return direct, 0

    parent = re.sub(r"_part\d+$", "", vid_id)
    parent_mp4 = vid_dir / f"{parent}.mp4"
    if not parent_mp4.exists() or not is_valid_mp4(parent_mp4):
        if "_" in parent:
            cd, chuyen = parent.split("_", 1)
            cands = [p for p in (REPO_ROOT / "data" / cd / chuyen).glob("*.mp4") if is_valid_mp4(p)]
            if cands:
                parent_mp4 = cands[0]

    if parent_mp4.exists() and is_valid_mp4(parent_mp4):
        start_frame = get_part_start_frame(vid_id, parent)
        return parent_mp4, start_frame

    return None, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=["resnet50", "dinov2", "dinov3", "videomae"],
                        default="resnet50")
    parser.add_argument("--model-name", type=str, default=None,
                        help="HuggingFace model ID or local path (e.g. facebook/dinov3-vitb16-pretrain-lvd1689m)")
    parser.add_argument("--vid-dir",  type=Path, default=DEFAULT_VID_DIR)
    parser.add_argument("--feat-dir", type=Path, default=None)
    parser.add_argument("--gt-dir",   type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--overwrite",      action="store_true")
    parser.add_argument("--fix-mismatches", action="store_true",
                        help="Re-extract only videos where npy T != groundTruth T")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Frames per GPU batch — reduce to 4 or 2 if OOM")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.feat_dir is None:
        if args.backbone == "resnet50":
            args.feat_dir = DEFAULT_FEAT_DIR
        else:
            args.feat_dir = DEFAULT_FEAT_DIR.parent / f"features_{args.backbone}"

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
    elif args.backbone == "dinov3":
        tf, infer, feat_dim = build_dinov3(device, args.model_name)
    elif args.backbone == "videomae":
        _, infer_clip, feat_dim, clip_len = build_videomae(device)
        tf = infer = None
    else:
        raise ValueError(args.backbone)

    print(f"Backbone : {args.backbone}  (dim={feat_dim})")
    print(f"Device   : {device}")
    print(f"Feat dir : {args.feat_dir}")

    # ---- discover targets from authoritative groundTruth files ----
    gt_ids   = {f[:-4] for f in os.listdir(args.gt_dir) if f.endswith(".txt")}
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
        targets = sorted(gt_ids)
    else:
        targets = sorted(gt_ids - feat_ids)

    print(f"\nVideos total : {len(gt_ids)}")
    print(f"Already have : {len(feat_ids)}")
    print(f"To extract   : {len(targets)}")

    if not targets:
        print("Nothing to do. Use --overwrite or --fix-mismatches.")
        return

    for i, vid_id in enumerate(targets, 1):
        out = args.feat_dir / f"{vid_id}.npy"

        if out.exists() and not args.overwrite and not args.fix_mismatches:
            print(f"[{i}/{len(targets)}] SKIP {vid_id} — already exists")
            continue

        mp4, start_frame = resolve_video_and_offset(vid_id, args.vid_dir)
        if not mp4 or not mp4.exists():
            print(f"[{i}/{len(targets)}] SKIP {vid_id} — source video not found")
            continue

        expected_T = gt_frame_count(args.gt_dir, vid_id)
        print(f"[{i}/{len(targets)}] {vid_id}  (expected T={expected_T}, offset={start_frame}) ...", flush=True)

        try:
            if args.backbone == "videomae":
                arr = extract_videomae(mp4, expected_T, infer_clip, clip_len, start_frame=start_frame)
            else:
                arr = extract_streaming(mp4, expected_T, tf, infer, device, args.batch_size, start_frame=start_frame)

            np.save(out, arr)
            print(f"    → saved  shape={arr.shape}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
