#!/usr/bin/env python3
"""Inference & Video Visualization Tool for BaFormer.

Runs inference on the validation or test split of dataset_tas_instance,
computes sample-level and aggregate metrics (Frame Acc, Edit Score, F1, and
Instance tIoU), and renders annotated side-by-side / dual-timeline .mp4 videos
for every sample comparing Ground Truth against Model Predictions.

Usage:
    cd /home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer
    python infer_and_visualize.py \\
        --config configs/tas_instance.yaml \\
        --checkpoint experiments/tas_instance/bk_fde_tde/final/1/checkpoint_best.pth \\
        --out-dir experiments/tas_instance/bk_fde_tde/final/1/visualizations \\
        --split val
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure BaFormer root is in sys.path
baformer_root = Path(__file__).resolve().parent
if str(baformer_root) not in sys.path:
    sys.path.insert(0, str(baformer_root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from action_segmentation import (
    create_model,
    get_default_config,
    update_config,
)
from action_segmentation.datasets import create_instance_dataset
from action_segmentation.utils.metrics import (
    accuracy,
    compute_metrics,
    edit_score,
    evaluate_instance_metrics,
    f_score,
)

# ---------------------------------------------------------------------------
# Color & Style Definitions
# ---------------------------------------------------------------------------
CLASS_COLORS_RGB = {
    0: (46, 204, 113),    # Sewing/Joining -> Emerald Green
    1: (52, 152, 219),    # Positioning/Handling -> Dodger Blue
    2: (243, 156, 18),    # Adjustment/Alignment/Preparation -> Amber / Orange
    3: (155, 89, 182),    # Inspection/Auxiliary -> Amethyst Purple
    -1: (127, 140, 141),  # Background / Unknown -> Neutral Gray
}

BOUNDARY_COLORS_RGB = {
    "same_class": (0, 240, 255),    # Cyan
    "class_change": (255, 220, 40), # Vivid Gold
    "pred_boundary": (255, 75, 75), # Coral Red
}

FONT_PATHS = [
    "/home/hungbm/ai4training/venv/lib/python3.12/site-packages/cv2/qt/fonts/DejaVuSans-Bold.ttf",
    "/home/hungbm/ai4training/venv/lib/python3.12/site-packages/cv2/qt/fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    target_idx = 0 if bold else 1
    for p in [FONT_PATHS[target_idx]] + FONT_PATHS:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def compute_f1(f_scores):
    """Computes F1 tensor of shape (3,) from f_scores tuple [tp, fp, fn]."""
    tp, fp, fn = f_scores
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    f1 = torch.nan_to_num(f1) * 100
    return f1


def load_config_and_model(config_path: str, checkpoint_path: str, device: torch.device):
    config = get_default_config()
    config.merge_from_file(config_path, allow_unsafe=True)
    config = update_config(config)
    config.device = str(device)
    config.freeze()

    model = create_model(config)
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        # strip module. prefix if multi-gpu
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k[7:] if k.startswith("module.") else k
            clean_state_dict[clean_k] = v
        model.load_state_dict(clean_state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model.to(device)
    model.eval()
    return config, model


def decode_predictions(
    outputs: dict,
    threshold: float = 0.2,
    relabel_theta: int = 5,
) -> Tuple[np.ndarray, List[dict], np.ndarray, List[int]]:
    """Decodes BaFormer outputs into frame-wise predictions, predicted instances,
    boundary probability curve, and peak cut locations.
    """
    mask_cls_raw = outputs["pred_logits"][0]  # (Q, C+1)
    mask_pred_raw = outputs["pred_masks"][0]  # (Q, L)
    mask_bd_raw = outputs["pred_boundarys"][0]  # (1, L)

    mask_bd_prob = mask_bd_raw.sigmoid().float().squeeze(dim=0)  # (L,)
    bd_curve_np = mask_bd_prob.detach().cpu().numpy()

    # Find peaks above threshold
    th_bd = mask_bd_prob.clone()
    th_bd[th_bd < threshold] = 0.0
    peak = torch.where((th_bd[:-2] < th_bd[1:-1]) & (th_bd[2:] < th_bd[1:-1]))[0]
    indices = [0] + (peak + 1).tolist() + [th_bd.shape[-1]]

    L = mask_bd_prob.shape[-1]
    mask_ref = torch.zeros_like(mask_pred_raw)
    pred_instances = []

    # Assign winning query to each interval
    mask_cls = F.softmax(mask_cls_raw, dim=-1)[..., :-1]  # (Q, C)
    for i in range(len(indices) - 1):
        s, e = indices[i], indices[i + 1]
        if e <= s:
            continue
        q_idx = mask_pred_raw[:, s:e].sigmoid().sum(dim=-1).argmax(0).item()
        mask_ref[q_idx, s:e] = 1.0

        c_idx = mask_cls[q_idx].argmax(dim=-1).item()
        pred_instances.append({
            "id": i,
            "start_frame": s,
            "end_frame": e,
            "class_id": c_idx,
            "query_id": q_idx,
        })

    # Per-frame class scores: (L, C)
    dense_scores = torch.einsum("qc,ql->cl", mask_cls, mask_ref).transpose(0, 1)

    # Optional minimal relabeling for short spikes
    if relabel_theta > 0:
        preds = dense_scores.argmax(dim=1)
        last = preds[0]
        cnt = 1
        for j in range(1, len(preds)):
            if last == preds[j]:
                cnt += 1
            else:
                if cnt > relabel_theta:
                    cnt = 1
                    last = preds[j]
                else:
                    dense_scores[j - cnt : j, :] = dense_scores[j - cnt - 1, :]
                    cnt = 1
                    last = preds[j]
        if cnt <= relabel_theta and len(preds) > cnt:
            dense_scores[len(preds) - cnt : len(preds), :] = dense_scores[len(preds) - cnt - 1, :]

    frame_preds = dense_scores.argmax(dim=1).detach().cpu().numpy()
    return frame_preds, pred_instances, bd_curve_np, indices


def render_sample_mp4(
    video_path: Path,
    output_path: Path,
    video_id: str,
    gt_instances: List[dict],
    pred_instances: List[dict],
    gt_frame_labels: np.ndarray,
    pred_frame_labels: np.ndarray,
    bd_curve: np.ndarray,
    classes_map: Dict[int, str],
    target_width: int = 1280,
    target_height: int = 720,
    fps_override: Optional[float] = None,
):
    """Renders dual-timeline visual comparison video for a sample."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Cannot open video file {video_path}, skipping .mp4 render.")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = fps_override or cap_fps

    total_frames = min(len(gt_frame_labels), len(pred_frame_labels), cap_frame_count)
    if total_frames <= 0:
        total_frames = max(len(gt_frame_labels), cap_frame_count)
    total_duration = total_frames / video_fps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, (target_width, target_height))

    # Fonts
    font_title = get_font(18, bold=True)
    font_badge = get_font(13, bold=True)
    font_regular = get_font(12, bold=False)
    font_small = get_font(11, bold=False)
    font_track = get_font(11, bold=True)

    # Timeline Layout Metrics
    tl_margin_x = 50
    tl_w = target_width - 2 * tl_margin_x
    tl_h_track = 24
    tl_gap = 14
    tl_bottom_margin = 18

    # Track 2: Predictions
    tl_y_pred = target_height - tl_bottom_margin - tl_h_track
    # Track 1: Ground Truth
    tl_y_gt = tl_y_pred - tl_gap - tl_h_track
    tl_panel_y = tl_y_gt - 42
    tl_panel_h = target_height - tl_panel_y - 6

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_bgr.shape[1] != target_width or frame_bgr.shape[0] != target_height:
            frame_resized = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame_bgr.copy()

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        overlay = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cur_time = frame_idx / video_fps
        gt_class_id = int(gt_frame_labels[frame_idx]) if frame_idx < len(gt_frame_labels) else -1
        pred_class_id = int(pred_frame_labels[frame_idx]) if frame_idx < len(pred_frame_labels) else -1

        # Current active GT instance
        active_gt = None
        for inst in gt_instances:
            if inst["start_frame"] <= frame_idx < inst["end_frame"]:
                active_gt = inst
                break

        # Current active Pred instance
        active_pred = None
        for inst in pred_instances:
            if inst["start_frame"] <= frame_idx < inst["end_frame"]:
                active_pred = inst
                break

        # ---------------------------------------------------------
        # 1. TOP HEADER HUD
        # ---------------------------------------------------------
        top_h = 100
        draw.rectangle([0, 0, target_width, top_h], fill=(15, 23, 42, 225))
        draw.line([0, top_h, target_width, top_h], fill=(51, 65, 85, 255), width=2)

        # Title & Video Info
        draw.text((25, 10), f"SAMPLE: {video_id}  |  BAFORMER INFERENCE VISUALIZER", fill=(255, 255, 255), font=font_title)
        info_str = f"Time: {cur_time:05.2f}s / {total_duration:05.2f}s   •   Frame: {frame_idx:04d} / {total_frames:04d}   •   FPS: {video_fps:.1f}"
        draw.text((25, 34), info_str, fill=(148, 163, 184), font=font_regular)

        # Match Status Indicator
        is_match = (gt_class_id == pred_class_id) and (gt_class_id != -1)
        status_bg = (34, 197, 94, 230) if is_match else (239, 68, 68, 230)
        status_text = "CLASS MATCH" if is_match else "CLASS MISMATCH"
        draw.rounded_rectangle([25, 60, 160, 88], radius=6, fill=status_bg)
        draw.text((36, 66), status_text, fill=(255, 255, 255), font=font_badge)

        # GT Badge
        gt_col = CLASS_COLORS_RGB.get(gt_class_id, CLASS_COLORS_RGB[-1])
        gt_name = classes_map.get(gt_class_id, "Unknown").split("/")[0]
        gt_badge_txt = f"GT: #{active_gt['id'] if active_gt else '?'} {gt_name}"
        draw.rounded_rectangle([175, 60, 390, 88], radius=6, fill=(*gt_col, 240))
        draw.text((185, 66), gt_badge_txt, fill=(255, 255, 255), font=font_badge)

        # Pred Badge
        pred_col = CLASS_COLORS_RGB.get(pred_class_id, CLASS_COLORS_RGB[-1])
        pred_name = classes_map.get(pred_class_id, "Unknown").split("/")[0]
        pred_badge_txt = f"PRED: #{active_pred['id'] if active_pred else '?'} {pred_name}"
        draw.rounded_rectangle([405, 60, 620, 88], radius=6, fill=(*pred_col, 240))
        draw.text((415, 66), pred_badge_txt, fill=(255, 255, 255), font=font_badge)

        # ---------------------------------------------------------
        # 2. CLASS COLOR LEGEND (Top Right)
        # ---------------------------------------------------------
        leg_w = 340
        leg_x = target_width - leg_w - 20
        leg_y = 10
        leg_h = 80
        draw.rounded_rectangle([leg_x, leg_y, leg_x + leg_w, leg_y + leg_h], radius=8, fill=(10, 15, 30, 220), outline=(51, 65, 85, 200), width=1)
        draw.text((leg_x + 12, leg_y + 5), "ACTION CLASSES", fill=(203, 213, 225), font=font_small)

        for cid in sorted(classes_map.keys())[:4]:
            row = cid // 2
            col = cid % 2
            c_x = leg_x + 12 + col * 165
            c_y = leg_y + 24 + row * 22
            ccol = CLASS_COLORS_RGB.get(cid, (200, 200, 200))
            draw.rectangle([c_x, c_y + 2, c_x + 10, c_y + 12], fill=(*ccol, 255))
            short_c = classes_map[cid].replace("Adjustment/Alignment/Preparation", "Adjustment/Prep").replace("Positioning/Handling", "Positioning").replace("Inspection/Auxiliary", "Inspection")
            draw.text((c_x + 16, c_y), short_c, fill=(241, 245, 249), font=font_small)

        # ---------------------------------------------------------
        # 3. BOTTOM DUAL-TRACK TIMELINE
        # ---------------------------------------------------------
        draw.rounded_rectangle([tl_margin_x - 12, tl_panel_y, target_width - tl_margin_x + 12, tl_panel_y + tl_panel_h], radius=10, fill=(15, 23, 42, 230), outline=(51, 65, 85, 255), width=1)
        draw.text((tl_margin_x, tl_panel_y + 8), "TIMELINE COMPARISON", fill=(226, 232, 240), font=font_track)

        # Track 1: Ground Truth
        draw.text((tl_margin_x, tl_y_gt - 15), "Ground Truth Instances", fill=(148, 163, 184), font=font_small)
        draw.rectangle([tl_margin_x, tl_y_gt, tl_margin_x + tl_w, tl_y_gt + tl_h_track], fill=(30, 41, 59, 255))
        for inst in gt_instances:
            s_f = max(0, inst["start_frame"])
            e_f = min(total_frames, inst["end_frame"])
            if e_f <= s_f:
                continue
            x1 = tl_margin_x + int(s_f / total_frames * tl_w)
            x2 = tl_margin_x + int(e_f / total_frames * tl_w)
            x2 = max(x1 + 1, x2)
            c_rgb = CLASS_COLORS_RGB.get(inst["class_id"], CLASS_COLORS_RGB[-1])
            draw.rectangle([x1, tl_y_gt, x2, tl_y_gt + tl_h_track], fill=(*c_rgb, 220), outline=(15, 23, 42, 255), width=1)
            # Boundary line
            draw.line([x1, tl_y_gt, x1, tl_y_gt + tl_h_track], fill=(255, 255, 255, 220), width=1)

        # Track 2: Predictions
        draw.text((tl_margin_x, tl_y_pred - 15), "BaFormer Predicted Instances", fill=(148, 163, 184), font=font_small)
        draw.rectangle([tl_margin_x, tl_y_pred, tl_margin_x + tl_w, tl_y_pred + tl_h_track], fill=(30, 41, 59, 255))
        for inst in pred_instances:
            s_f = max(0, inst["start_frame"])
            e_f = min(total_frames, inst["end_frame"])
            if e_f <= s_f:
                continue
            x1 = tl_margin_x + int(s_f / total_frames * tl_w)
            x2 = tl_margin_x + int(e_f / total_frames * tl_w)
            x2 = max(x1 + 1, x2)
            c_rgb = CLASS_COLORS_RGB.get(inst["class_id"], CLASS_COLORS_RGB[-1])
            draw.rectangle([x1, tl_y_pred, x2, tl_y_pred + tl_h_track], fill=(*c_rgb, 220), outline=(15, 23, 42, 255), width=1)
            # Boundary cut marker
            draw.line([x1, tl_y_pred, x1, tl_y_pred + tl_h_track], fill=BOUNDARY_COLORS_RGB["pred_boundary"], width=1)

        # Moving Playhead Cursor (Red Line across both tracks)
        playhead_x = tl_margin_x + int(min(1.0, frame_idx / max(1, total_frames - 1)) * tl_w)
        draw.line([playhead_x, tl_y_gt - 6, playhead_x, tl_y_pred + tl_h_track + 6], fill=(255, 50, 50, 255), width=2)
        # Playhead handle
        draw.polygon([(playhead_x - 5, tl_y_gt - 6), (playhead_x + 5, tl_y_gt - 6), (playhead_x, tl_y_gt - 1)], fill=(255, 50, 50, 255))

        # Composite and write frame
        out_frame = Image.alpha_composite(img.convert("RGBA"), overlay)
        out_bgr = cv2.cvtColor(np.array(out_frame), cv2.COLOR_RGBA2BGR)
        writer.write(out_bgr)
        frame_idx += 1

    cap.release()
    writer.release()


def run_inference_and_visualization(
    config_path: str,
    checkpoint_path: str,
    out_dir: str,
    split: str = "val",
    dataset_dir_override: Optional[str] = None,
    max_samples: Optional[int] = None,
    render_video: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config, model = load_config_and_model(config_path, checkpoint_path, device)
    dataset_root = Path(dataset_dir_override or config.dataset.dataset_dir)
    print(f"Dataset root: {dataset_root}")

    # Load classes mapping
    classes_json_path = dataset_root / "classes.json"
    if classes_json_path.exists():
        with open(classes_json_path, "r", encoding="utf-8") as f:
            classes_list = json.load(f)
        classes_map = {item["class_id"]: item["class_name"] for item in classes_list}
    else:
        classes_map = {0: "Sewing/Joining", 1: "Positioning/Handling", 2: "Adjustment/Prep", 3: "Inspection/Aux"}

    # Load Dataset
    is_train_split = (split == "train")
    dataset = create_instance_dataset(config, is_train=is_train_split)
    if is_train_split and isinstance(dataset, tuple):
        dataset = dataset[0]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    video_out_dir = out_path / "videos"
    if render_video:
        video_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(dataset)} samples (split: {split}) ...")
    all_sample_results = []
    overall_accs, overall_edits = [], []
    overall_f1_10, overall_f1_25, overall_f1_50 = [], [], []
    overall_tiou_25, overall_tiou_50, overall_tiou_75 = [], [], []
    total_tp = torch.zeros(3)
    total_fp = torch.zeros(3)
    total_fn = torch.zeros(3)

    num_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for idx in range(num_samples):
        item = dataset[idx]
        if is_train_split:
            data, frame_target, instances_arr, fname, _ = item
        else:
            data, frame_target, instances_arr, fname = item

        video_id = fname
        print(f"\n[{idx + 1}/{num_samples}] Processing sample: {video_id} ...")

        data_t = torch.from_numpy(data).unsqueeze(0).to(device)  # (1, C, L)
        frame_target_t = torch.from_numpy(frame_target).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(data_t)

            # Handle sample rate upsampling if applicable
            if config.dataset.sample_rate != 1:
                L = frame_target_t.shape[-1]
                outputs["pred_masks"] = outputs["pred_masks"].repeat_interleave(config.dataset.sample_rate, dim=-1)[:, :, :L]
                outputs["pred_boundarys"] = F.interpolate(outputs["pred_boundarys"], size=L, mode="linear")

            frame_preds, pred_instances, bd_curve, indices = decode_predictions(
                outputs,
                threshold=config.dataset.threshold,
                relabel_theta=5,
            )

        # Standard TAS metrics
        seg_pred_t = torch.from_numpy(frame_preds).unsqueeze(0)
        num_correct, acc, edit, f_scores = compute_metrics(
            config.dataset.name,
            F.one_hot(torch.from_numpy(frame_preds), num_classes=len(classes_map)).float(),
            torch.from_numpy(frame_target),
        )

        acc_val = float(acc.item()) * 100 if torch.is_tensor(acc) else float(acc) * 100
        edit_val = float(edit.item()) if torch.is_tensor(edit) else float(edit)

        # Compute F1 for this sample
        sample_f1 = compute_f1(f_scores)
        f1_10 = float(sample_f1[0].item())
        f1_25 = float(sample_f1[1].item())
        f1_50 = float(sample_f1[2].item())
        f1_mean = (f1_10 + f1_25 + f1_50) / 3.0

        # Accumulate dataset-level TP, FP, FN
        total_tp += f_scores[0]
        total_fp += f_scores[1]
        total_fn += f_scores[2]

        # Instance tIoU evaluation
        gt_inst_tuples = [(int(r[0]), int(r[1]), int(r[2])) for r in instances_arr]
        pred_inst_tuples = [(p["start_frame"], p["end_frame"], p["class_id"]) for p in pred_instances]
        tiou_res = evaluate_instance_metrics(pred_inst_tuples, gt_inst_tuples, thresholds=(0.25, 0.5, 0.75))

        overall_accs.append(acc_val)
        overall_edits.append(edit_val)
        overall_f1_10.append(f1_10)
        overall_f1_25.append(f1_25)
        overall_f1_50.append(f1_50)
        overall_tiou_25.append(tiou_res[0.25]["f1"] * 100)
        overall_tiou_50.append(tiou_res[0.50]["f1"] * 100)
        overall_tiou_75.append(tiou_res[0.75]["f1"] * 100)

        print(
            f"   Frame Acc: {acc_val:.2f}% | Edit Score: {edit_val:.2f} | "
            f"F1@10/25/50: {f1_10:.1f} / {f1_25:.1f} / {f1_50:.1f} (Mean: {f1_mean:.1f}) | "
            f"tIoU F1@50: {tiou_res[0.50]['f1'] * 100:.1f}%"
        )

        # Ground truth instances dictionary for rendering
        gt_instances_dicts = [
            {
                "id": i,
                "start_frame": int(r[0]),
                "end_frame": int(r[1]),
                "class_id": int(r[2]),
                "class_name": classes_map.get(int(r[2]), "Unknown"),
            }
            for i, r in enumerate(instances_arr)
        ]

        # Render video if requested
        rendered_mp4_path = None
        if render_video:
            mp4_src = dataset_root / "videos" / f"{video_id}.mp4"
            if mp4_src.exists():
                rendered_mp4 = video_out_dir / f"{video_id}_annotated.mp4"
                print(f"   Rendering comparison video -> {rendered_mp4} ...")
                render_sample_mp4(
                    video_path=mp4_src,
                    output_path=rendered_mp4,
                    video_id=video_id,
                    gt_instances=gt_instances_dicts,
                    pred_instances=pred_instances,
                    gt_frame_labels=frame_target,
                    pred_frame_labels=frame_preds,
                    bd_curve=bd_curve,
                    classes_map=classes_map,
                )
                rendered_mp4_path = str(rendered_mp4)

        all_sample_results.append({
            "video_id": video_id,
            "metrics": {
                "accuracy": acc_val,
                "edit_score": edit_val,
                "f1@10": f1_10,
                "f1@25": f1_25,
                "f1@50": f1_50,
                "f1_mean": f1_mean,
                "tiou_f1@25": tiou_res[0.25]["f1"] * 100,
                "tiou_f1@50": tiou_res[0.50]["f1"] * 100,
                "tiou_f1@75": tiou_res[0.75]["f1"] * 100,
            },
            "num_gt_instances": len(gt_inst_tuples),
            "num_pred_instances": len(pred_instances),
            "video_mp4": rendered_mp4_path,
            "predicted_instances": pred_instances,
        })

    # Summary metrics across split
    dataset_f1 = compute_f1((total_tp, total_fp, total_fn))
    summary = {
        "split": split,
        "num_samples": len(all_sample_results),
        "mean_accuracy": float(np.mean(overall_accs)),
        "mean_edit_score": float(np.mean(overall_edits)),
        "dataset_f1@10": float(dataset_f1[0].item()),
        "dataset_f1@25": float(dataset_f1[1].item()),
        "dataset_f1@50": float(dataset_f1[2].item()),
        "dataset_f1_mean": float(dataset_f1.mean().item()),
        "mean_f1@10": float(np.mean(overall_f1_10)),
        "mean_f1@25": float(np.mean(overall_f1_25)),
        "mean_f1@50": float(np.mean(overall_f1_50)),
        "mean_f1_macro": float(np.mean([np.mean(overall_f1_10), np.mean(overall_f1_25), np.mean(overall_f1_50)])),
        "instance_tiou_f1@25": float(np.mean(overall_tiou_25)),
        "instance_tiou_f1@50": float(np.mean(overall_tiou_50)),
        "instance_tiou_f1@75": float(np.mean(overall_tiou_75)),
    }

    # Save JSON files
    with open(out_path / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_sample_results, f, indent=2)

    with open(out_path / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"INFERENCE COMPLETE ({split.upper()} SET - {summary['num_samples']} SAMPLES)")
    print("=" * 80)
    print(f"Mean Frame Accuracy:   {summary['mean_accuracy']:.2f}%")
    print(f"Mean Edit Score:       {summary['mean_edit_score']:.2f}")
    print(f"Dataset F1@10/25/50:   {summary['dataset_f1@10']:.2f} / {summary['dataset_f1@25']:.2f} / {summary['dataset_f1@50']:.2f} (Mean: {summary['dataset_f1_mean']:.2f})")
    print(f"Sample-Avg F1@10/25/50:{summary['mean_f1@10']:.2f} / {summary['mean_f1@25']:.2f} / {summary['mean_f1@50']:.2f} (Macro: {summary['mean_f1_macro']:.2f})")
    print(f"Instance tIoU F1@25:   {summary['instance_tiou_f1@25']:.2f}%")
    print(f"Instance tIoU F1@50:   {summary['instance_tiou_f1@50']:.2f}%")
    print(f"Instance tIoU F1@75:   {summary['instance_tiou_f1@75']:.2f}%")
    print(f"\nArtifacts saved to:    {out_path}")
    if render_video:
        print(f"Annotated Videos:      {video_out_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Infer BaFormer model and render .mp4 visualization.")
    parser.add_argument("--config", default="configs/tas_instance.yaml", type=str, help="Path to config yaml")
    parser.add_argument(
        "--checkpoint",
        default="experiments/tas_instance/bk_fde_tde/final/1/checkpoint_best.pth",
        type=str,
        help="Path to checkpoint_best.pth",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/tas_instance/bk_fde_tde/final/1/visualizations",
        type=str,
        help="Output directory for predictions and mp4 videos",
    )
    parser.add_argument("--split", default="val", type=str, choices=["val", "train"], help="Split to evaluate")
    parser.add_argument("--dataset-dir", default=None, type=str, help="Optional override for dataset_tas_instance path")
    parser.add_argument("--max-samples", default=None, type=int, help="Limit number of samples to process")
    parser.add_argument("--no-render", action="store_true", help="Skip .mp4 video rendering")
    args = parser.parse_args()

    run_inference_and_visualization(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        split=args.split,
        dataset_dir_override=args.dataset_dir,
        max_samples=args.max_samples,
        render_video=not args.no_render,
    )


if __name__ == "__main__":
    main()
