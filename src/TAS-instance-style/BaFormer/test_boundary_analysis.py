#!/usr/bin/env python3
"""Deep Diagnostic Tool: Action Boundary Precision & Recall Analysis.

Analyzes the boundary prediction head of existing checkpoints (Run 7, Run 8)
under various post-processing filters (NMS, prominence, threshold) and compares
raw local maxima vs NMS-filtered peaks against Ground Truth boundaries.

Usage:
    python test_boundary_analysis.py --checkpoint experiments/tas_instance/bk_fde_tde/final_exp08/1/checkpoint_best.pth
    python test_boundary_analysis.py --checkpoint experiments/tas_instance/bk_fde_tde/final_exp07/1/checkpoint_best.pth
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

baformer_root = Path(__file__).resolve().parent
if str(baformer_root) not in sys.path:
    sys.path.insert(0, str(baformer_root))

import numpy as np
import torch
import torch.nn.functional as F

from action_segmentation import (
    create_model,
    get_default_config,
    update_config,
)
from action_segmentation.datasets import create_instance_dataset


def extract_peaks_raw(bd_prob: np.ndarray, threshold: float = 0.25) -> list[int]:
    """Legacy 3-frame local maxima above threshold (used in current main.py)."""
    if len(bd_prob) < 3:
        return []
    cond = (bd_prob[:-2] < bd_prob[1:-1]) & (bd_prob[2:] < bd_prob[1:-1]) & (bd_prob[1:-1] >= threshold)
    return (np.where(cond)[0] + 1).tolist()


def extract_peaks_nms(
    bd_prob: np.ndarray,
    threshold: float = 0.25,
    min_distance: int = 6,
    prominence: float = 0.035,
) -> list[int]:
    """NMS + Prominence peak extraction.

    1. Finds local maxima above threshold that exceed local 7-frame floor by prominence.
    2. Enforces min_distance by Non-Maximum Suppression (keeps highest peak within window).
    """
    L = len(bd_prob)
    if L < 3:
        return []

    # Local floor via 7-frame min pool
    pad_bd = np.pad(bd_prob, (3, 3), mode='edge')
    local_min = np.array([np.min(pad_bd[i : i + 7]) for i in range(L)])
    prom = bd_prob - local_min

    # Candidate peaks
    cand_cond = (bd_prob[1:-1] >= threshold) & \
                (bd_prob[1:-1] >= bd_prob[:-2]) & \
                (bd_prob[1:-1] >= bd_prob[2:]) & \
                (prom[1:-1] >= prominence)
    raw_peaks = (np.where(cand_cond)[0] + 1).tolist()

    if not raw_peaks:
        return []

    # Sort peaks by probability descending for greedy NMS
    sorted_peaks = sorted(raw_peaks, key=lambda p: bd_prob[p], reverse=True)
    kept_peaks = []
    suppressed = np.zeros(L, dtype=bool)

    for p in sorted_peaks:
        if suppressed[p]:
            continue
        kept_peaks.append(p)
        # Suppress neighborhood within min_distance
        low = max(0, p - min_distance)
        high = min(L, p + min_distance + 1)
        suppressed[low:high] = True

    return sorted(kept_peaks)


def evaluate_boundary_peaks(
    all_pred_peaks: list[list[int]],
    all_gt_boundaries: list[list[int]],
    all_fps: list[float] | None = None,
    tol_seconds: float = 0.25,
) -> tuple[float, float, float, int, int, int]:
    """Computes Precision, Recall, F1@tol_seconds, TP, FP, FN.

    Tolerance is a fixed time window (+/-tol_seconds), converted per-clip to a
    frame tolerance via that clip's fps (`all_fps[i]`), replacing the old fixed
    +/-3 frame window (which corresponds to a different real-world time window
    depending on fps). If `all_fps` is not given, falls back to a fps=25
    assumption for every clip.
    """
    tp, fp, fn = 0, 0, 0
    total_gt = sum(len(g) for g in all_gt_boundaries)

    if all_fps is None:
        all_fps = [25.0] * len(all_pred_peaks)

    for pred_peaks, gt_bounds, fps in zip(all_pred_peaks, all_gt_boundaries, all_fps):
        tol = tol_seconds * fps
        matched_gt = set()
        for p in pred_peaks:
            matched = False
            for gi, g in enumerate(gt_bounds):
                if abs(p - g) <= tol and gi not in matched_gt:
                    matched_gt.add(gi)
                    matched = True
                    break
            if matched:
                tp += 1
            else:
                fp += 1
        fn += (len(gt_bounds) - len(matched_gt))

    prec = (tp / max(1, tp + fp)) * 100.0
    rec = (tp / max(1, tp + fn)) * 100.0
    f1 = (2 * prec * rec / max(1e-8, prec + rec))
    return prec, rec, f1, tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tas_instance.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/tas_instance/bk_fde_tde/final_exp13/1/checkpoint_best.pth",
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.merge_from_file(args.config, allow_unsafe=True)
    cfg = update_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = create_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    val_dataset = create_instance_dataset(cfg, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Extracting validation boundary probabilities for {len(val_loader)} clips...")
    cached_bd_probs = []
    all_gt_bounds = []
    all_ins_bounds = []
    all_fps = []

    import json as _json
    dataset_dir = Path(cfg.dataset.dataset_dir)

    def _lookup_fps(fname_str: str) -> float:
        ann_path = dataset_dir / 'annotations' / f'{fname_str}.json'
        try:
            if ann_path.exists():
                with open(ann_path, 'r', encoding='utf-8') as f:
                    return float(_json.load(f).get('fps', 25.0))
        except Exception:
            pass
        return 25.0

    with torch.no_grad():
        for batch in val_loader:
            data, frame_target, instances, fname = batch
            data = data.to(device)
            outputs = model(data)

            bd_prob = outputs["pred_boundarys"][0].sigmoid().squeeze().cpu().numpy()
            gt_cls = frame_target.squeeze().cpu().numpy()
            gt_bounds = (np.where(gt_cls[1:] != gt_cls[:-1])[0] + 1).tolist()

            if instances is not None and instances.numel() > 3:
                ins_arr = instances.squeeze(0).cpu().numpy() if instances.dim() > 1 else instances.cpu().numpy()
                if ins_arr.ndim == 2 and ins_arr.shape[0] > 1:
                    ins_bounds = ins_arr[1:, 0].tolist()
                else:
                    ins_bounds = gt_bounds
            else:
                ins_bounds = gt_bounds

            fname_str = fname[0] if isinstance(fname, (list, tuple)) else fname
            cached_bd_probs.append(bd_prob)
            all_gt_bounds.append(gt_bounds)
            all_ins_bounds.append(ins_bounds)
            all_fps.append(_lookup_fps(fname_str))

    total_gt = sum(len(g) for g in all_gt_bounds)
    total_ins = sum(len(g) for g in all_ins_bounds)
    print(f"Total Ground Truth: Class-change boundaries = {total_gt} | Instance boundaries = {total_ins}")

    # Fixed time-window tolerances (+/-0.1s, +/-0.25s, +/-0.5s), replacing the
    # old fixed +/-3 frame tolerance (which corresponds to a different
    # real-world time window depending on each clip's fps). The grid search
    # below is scored/sorted at the middle tolerance (+/-0.25s); the two
    # single-config comparisons report all three for context.
    TOL_SECONDS = [0.1, 0.25, 0.5]
    GRID_TOL = 0.25

    for target_name, gt_list in [("Instance Boundaries (443 GT)", all_ins_bounds), ("Class-Change Boundaries (177 GT)", all_gt_bounds)]:
        print("\n" + "=" * 150)
        print(f"EVALUATION TARGET: {target_name}")
        print("=" * 150)
        header_cols = " | ".join(f"F1@{t}s (%)".ljust(9) for t in TOL_SECONDS)
        print(f"{'Method / Configuration':<52} | {header_cols}")
        print("-" * 150)

        # 1. Baseline: Raw local maxima
        for raw_th in [0.15, 0.18, 0.25]:
            pred_raw = [extract_peaks_raw(p, threshold=raw_th) for p in cached_bd_probs]
            results_by_tol = [evaluate_boundary_peaks(pred_raw, gt_list, all_fps, tol_seconds=t) for t in TOL_SECONDS]
            f1_cols = " | ".join(f"{r[2]:>8.2f}%" for r in results_by_tol)
            print(f"{f'Raw Local Maxima (th={raw_th:.2f})':<52} | {f1_cols}")

        # 2. Exp13 Config
        pred_exp13 = [extract_peaks_nms(p, threshold=0.18, min_distance=8, prominence=0.065) for p in cached_bd_probs]
        results_by_tol = [evaluate_boundary_peaks(pred_exp13, gt_list, all_fps, tol_seconds=t) for t in TOL_SECONDS]
        f1_cols = " | ".join(f"{r[2]:>8.2f}%" for r in results_by_tol)
        print(f"{'Exp13 Config (th=0.18, d=8, prom=0.065)':<52} | {f1_cols}")
        detail_cols = " | ".join(f"P={r[0]:.1f}%/R={r[1]:.1f}%/TP={r[3]}/FP={r[4]}/FN={r[5]}" for r in results_by_tol)
        print(f"  (detail @0.1s/0.25s/0.5s: {detail_cols})")
        print("-" * 150)

        # 3. NMS Grid Search (scored at +/-0.25s)
        all_results = []
        for th in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]:
            for d_min in [4, 6, 8]:
                for prom in [0.015, 0.025, 0.035, 0.050, 0.065]:
                    pred_peaks = [extract_peaks_nms(p, threshold=th, min_distance=d_min, prominence=prom) for p in cached_bd_probs]
                    prec, rec, f1, tp, fp, fn = evaluate_boundary_peaks(pred_peaks, gt_list, all_fps, tol_seconds=GRID_TOL)
                    all_results.append((f1, prec, rec, tp, fp, fn, th, d_min, prom))

                    # Print select interesting anchor points
                    if (d_min == 6 and prom == 0.035 and th in [0.10, 0.12, 0.15, 0.18]) or \
                       (d_min == 6 and prom == 0.025 and th in [0.10, 0.12, 0.15]) or \
                       (d_min == 8 and prom == 0.035 and th in [0.12, 0.15, 0.18]):
                        name = f"NMS (th={th:.2f}, d={d_min}, prom={prom:.3f})"
                        print(f"{name:<52} | F1@{GRID_TOL}s={f1:>8.2f}% | Prec={prec:>8.2f}% | Rec={rec:>8.2f}% | TP={tp:>5d} | FP={fp:>5d} | FN={fn:>5d}")

        print("-" * 150)
        # Sort and print Top 3 configs
        all_results.sort(key=lambda x: x[0], reverse=True)
        print(f">> TOP 3 CONFIGURATIONS BY F1@{GRID_TOL}s:")
        for rank, (f1, prec, rec, tp, fp, fn, th, d_min, prom) in enumerate(all_results[:3], 1):
            print(f"   #{rank}: th={th:.2f}, d_min={d_min}, prom={prom:.3f} -> F1@{GRID_TOL}s={f1:.2f}% (Prec={prec:.2f}%, Rec={rec:.2f}%, TP={tp}, FP={fp}, FN={fn})")
        print("=" * 150)


if __name__ == "__main__":
    main()
