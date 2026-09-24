"""
ROI Mask Processor for DDM-Net datasets.

Implements the pipeline:
    Original Image => Crop follow Mask BBox => Apply Mask into Cropped Image => Resize

- Finds `<video_stem>.mask.png` for each video file.
- Computes non-zero Bounding Box with configurable margin (default: 5%).
- Crops the frames to the Bounding Box (eliminating surrounding margins).
- Multiplies with the binary mask (masking out background noise inside the Bounding Box to 0).
- Preserves uint8 dtype so subsequent transforms (Resize, ToDtype, Normalize) work identically.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
from PIL import Image


def find_mask_for_video(video_path: str | Path) -> Optional[Path]:
    """Find the corresponding *.mask.png file for a given video path (supporting symlinks)."""
    p = Path(video_path)
    real_p = p.resolve()

    # 1. Match stem of real video: <video_stem>.mask.png
    cand1 = real_p.with_suffix("").with_suffix(".mask.png")
    if cand1.exists():
        return cand1

    # 2. Match stem of original path if different
    cand2 = p.with_suffix("").with_suffix(".mask.png")
    if cand2.exists():
        return cand2

    # 3. Any *.mask.png in the same directory as the real video
    if real_p.parent.exists():
        cand3 = next(real_p.parent.glob("*.mask.png"), None)
        if cand3 and cand3.exists():
            return cand3

    return None


class MaskROI:
    """Stores bounding box coordinates and cropped binary mask for a video."""

    def __init__(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        cropped_mask: torch.Tensor,
        mask_path: Path,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        # Shape: (1, 1, H_crop, W_crop) uint8 with values 0 or 1
        self.cropped_mask = cropped_mask
        self.mask_path = mask_path

    def apply(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply Crop follow Mask BBox, then Apply Mask into Cropped Image.

        Args:
            frames: torch.Tensor of shape (N, C, H, W) or (C, H, W) in uint8.

        Returns:
            torch.Tensor of cropped & masked frames with same dtype (uint8).
        """
        orig_h, orig_w = frames.shape[-2], frames.shape[-1]
        
        # Guard against dimension mismatch (e.g. if mask resolution differs from video)
        # Bounding box coordinates are clamped to actual frame boundaries
        ymin = max(0, min(self.y_min, orig_h - 1))
        ymax = max(ymin + 1, min(self.y_max, orig_h))
        xmin = max(0, min(self.x_min, orig_w - 1))
        xmax = max(xmin + 1, min(self.x_max, orig_w))

        # 1. Crop to bounding box
        cropped = frames[..., ymin:ymax, xmin:xmax]

        # 2. Match cropped mask device/size
        mask = self.cropped_mask
        if mask.device != cropped.device:
            mask = mask.to(cropped.device)

        ch, cw = cropped.shape[-2], cropped.shape[-1]
        if mask.shape[-2] != ch or mask.shape[-1] != cw:
            # Resize mask if bounding box was clamped differently
            mask = mask[..., :ch, :cw]

        if frames.ndim == 3 and mask.ndim == 4:
            mask = mask.squeeze(0)  # (1, H, W) to preserve 3D shape

        # 3. Apply mask (pixels outside the polygon become 0)
        return cropped * mask


_ROI_CACHE: Dict[str, Optional[MaskROI]] = {}


def get_video_mask_roi(
    video_path: str | Path,
    margin: float = 0.05,
    use_cache: bool = True,
) -> Optional[MaskROI]:
    """Retrieve (and cache) MaskROI for a video.

    Returns None if no mask file is found.
    """
    key = str(video_path)
    if use_cache and key in _ROI_CACHE:
        return _ROI_CACHE[key]

    mask_path = find_mask_for_video(video_path)
    if mask_path is None:
        if use_cache:
            _ROI_CACHE[key] = None
        return None

    try:
        # Load mask image in grayscale
        img = Image.open(mask_path).convert("L")
        arr = np.array(img)
        H, W = arr.shape

        ys, xs = np.where(arr > 0)
        if len(xs) == 0:
            if use_cache:
                _ROI_CACHE[key] = None
            return None

        # Bounding box of non-zero mask pixels
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # Add optional margin
        if margin > 0:
            margin_x = int((x_max - x_min) * margin)
            margin_y = int((y_max - y_min) * margin)
            x_min = max(0, x_min - margin_x)
            x_max = min(W, x_max + margin_x + 1)
            y_min = max(0, y_min - margin_y)
            y_max = min(H, y_max + margin_y + 1)
        else:
            x_max = min(W, x_max + 1)
            y_max = min(H, y_max + 1)

        # Cropped binary mask (uint8: 0 or 1)
        cropped_mask_np = (arr[y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
        cropped_mask_t = torch.from_numpy(cropped_mask_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        roi = MaskROI(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            cropped_mask=cropped_mask_t,
            mask_path=mask_path,
        )

        if use_cache:
            _ROI_CACHE[key] = roi
            # Also cache by resolved real path
            real_key = str(Path(video_path).resolve())
            _ROI_CACHE[real_key] = roi

        return roi

    except Exception as e:
        print(f"Warning: Failed to load mask ROI for {video_path}: {e}")
        if use_cache:
            _ROI_CACHE[key] = None
        return None
