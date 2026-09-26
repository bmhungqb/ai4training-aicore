#!/usr/bin/env python3
"""Comprehensive Evaluation and Visual Error Inspection Tool for the AI Research Loop.

Fulfills two distinct evaluation pillars:
  1. Training Log Health: Loss curve, metrics progression, learning rate schedule,
     and anomaly detection (overfitting gap, plateau, NaN/divergence, oscillation).
  2. Validation Metrics & Visual Severe Error Cases:
     - Multi-tolerance F1@0.25s, 0.5s, 1.0s, per-video ranking.
     - Identifies and prioritizes the Top 3-4 most severe error cases (severe False Positives
       and severe False Negatives).
     - Extracts physical video frames and 3-frame temporal strips around error timestamps
       so the Evaluator and Diagnoser agents have direct visual evidence.
     - Generates `01_eval_report.md` and `eval_report.json`.

Usage:
  python tools/eval_and_inspect_errors.py \\
      --output-dir outputs/step_segment/diff_gebd/iter_01/exp_001_baseline \\
      --report-dir experiments/step_segment/diff_gebd/iter_01 \\
      --top-k-errors 4 --viz
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_step_segment_results import (
    load_all_ground_truth,
    match_boundaries,
    render_score_curve_plt,
    render_score_curve_svg,
    render_annotated_video,
)


# =============================================================================
# Pillar 1: Training Log Analysis & Anomaly Detection
# =============================================================================

def parse_training_log(log_path: Path) -> dict:
    """Parses train.log or run.log and extracts loss/metrics history and anomalies."""
    if not log_path.exists():
        return {
            "status": "not_found",
            "log_path": str(log_path),
            "message": "No training log file found in experiment directory.",
        }

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    
    epochs_data = []
    has_nan_or_inf = False
    
    # Patterns for common logs (EfficientGEBD / DiffGEBD / Lightning)
    loss_patterns = [
        re.compile(r"Epoch[:\s]+(?P<epoch>\d+).*?(?:loss|total_loss)[:\s=]+(?P<loss>[\d\.]+)", re.IGNORECASE),
        re.compile(r"loss[:\s=]+(?P<loss>[\d\.]+).*?Epoch[:\s]+(?P<epoch>\d+)", re.IGNORECASE),
        re.compile(r"Epoch\s*(?P<epoch>\d+)/\d+.*?loss:\s*(?P<loss>[\d\.]+)", re.IGNORECASE),
    ]
    f1_pattern = re.compile(r"(?:F1|f1_at_0\.5s|macro_f1)[:\s=]+(?P<f1>[\d\.]+)", re.IGNORECASE)
    lr_pattern = re.compile(r"(?:lr|learning_rate)[:\s=]+(?P<lr>[\d\.eE\-]+)", re.IGNORECASE)
    rec_pattern = re.compile(r"(?:Rec|recall)[:\s=]+(?P<rec>[\d\.]+)", re.IGNORECASE)
    prec_pattern = re.compile(r"(?:Prec|precision)[:\s=]+(?P<prec>[\d\.]+)", re.IGNORECASE)

    current_epoch = -1
    epoch_entry = {}

    for line in lines:
        if "nan" in line.lower() or "inf" in line.lower():
            if "loss" in line.lower() or "grad" in line.lower():
                has_nan_or_inf = True

        # Check for epoch & loss
        loss_val = None
        epoch_idx = None
        for p in loss_patterns:
            m = p.search(line)
            if m:
                epoch_idx = int(m.group("epoch"))
                loss_val = float(m.group("loss"))
                break

        if epoch_idx is not None:
            if epoch_idx != current_epoch:
                if epoch_entry:
                    epochs_data.append(epoch_entry)
                current_epoch = epoch_idx
                epoch_entry = {"epoch": current_epoch, "loss": loss_val}
            else:
                epoch_entry["loss"] = loss_val

        # Check F1 / Recall / Precision / LR
        f1_m = f1_pattern.search(line)
        if f1_m and epoch_entry:
            epoch_entry["f1"] = float(f1_m.group("f1"))

        rec_m = rec_pattern.search(line)
        if rec_m and epoch_entry:
            epoch_entry["recall"] = float(rec_m.group("rec"))

        prec_m = prec_pattern.search(line)
        if prec_m and epoch_entry:
            epoch_entry["precision"] = float(prec_m.group("prec"))

        lr_m = lr_pattern.search(line)
        if lr_m and epoch_entry:
            try:
                epoch_entry["lr"] = float(lr_m.group("lr"))
            except ValueError:
                pass

    if epoch_entry and (not epochs_data or epochs_data[-1] != epoch_entry):
        epochs_data.append(epoch_entry)

    # Anomaly checks
    anomalies = []
    if has_nan_or_inf:
        anomalies.append({
            "type": "loss_divergence",
            "severity": "CRITICAL",
            "description": "Detected NaN or Inf in loss/gradients during training run.",
        })

    losses = [e["loss"] for e in epochs_data if "loss" in e and e["loss"] is not None]
    f1s = [e["f1"] for e in epochs_data if "f1" in e and e["f1"] is not None]

    initial_loss = losses[0] if losses else None
    final_loss = losses[-1] if losses else None
    min_loss = min(losses) if losses else None
    max_f1 = max(f1s) if f1s else None
    final_f1 = f1s[-1] if f1s else None

    # Check for loss plateau / non-decreasing
    if initial_loss and final_loss:
        loss_drop = (initial_loss - final_loss) / (initial_loss + 1e-7)
        if loss_drop < 0.05 and len(losses) >= 10:
            anomalies.append({
                "type": "loss_stagnation",
                "severity": "HIGH",
                "description": f"Loss barely decreased over {len(losses)} epochs ({initial_loss:.4f} -> {final_loss:.4f}, drop: {loss_drop*100:.1f}%). Possible vanishing gradient or learning rate too low.",
            })

    # Check for Overfitting divergence (Train loss drops strongly, but Val F1 drops or degrades)
    if len(f1s) >= 10 and max_f1 is not None and final_f1 is not None:
        peak_epoch = [e["epoch"] for e in epochs_data if e.get("f1") == max_f1][0]
        if final_f1 < max_f1 - 0.08:
            anomalies.append({
                "type": "overfitting_divergence",
                "severity": "HIGH",
                "description": f"Validation F1 degraded from peak {max_f1:.4f} (epoch {peak_epoch}) down to {final_f1:.4f} at final epoch, while training continued. Clear sign of overfitting.",
            })

    # Check for heavy oscillation
    if len(f1s) >= 8:
        deltas = [abs(f1s[i] - f1s[i-1]) for i in range(1, len(f1s))]
        max_delta = max(deltas)
        if max_delta > 0.20:
            anomalies.append({
                "type": "training_instability",
                "severity": "MEDIUM",
                "description": f"Large jump in validation F1 observed between consecutive epochs (max delta: {max_delta:.4f}). Training dynamics may be unstable.",
            })

    return {
        "status": "success",
        "log_path": str(log_path),
        "total_epochs_recorded": len(epochs_data),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "peak_f1": max_f1,
        "final_f1": final_f1,
        "anomalies": anomalies,
        "epochs_summary": epochs_data[-5:] if epochs_data else [],
    }


# =============================================================================
# Pillar 2: Validation Metrics & Severe Error Identification
# =============================================================================

def extract_severe_error_cases(
    gt_map: dict[str, dict],
    pred_data: dict[str, list[float]],
    top_k: int = 4,
    tolerance: float = 0.5,
) -> list[dict]:
    """Identifies the most severe error cases:
    - Critical False Positives (Spurious detections > 1.5s away from any GT)
    - Critical False Negatives (GT boundaries completely missed > 1.5s)
    - Significant Timing Drift (> 0.5s)
    """
    candidates = []

    for vid, preds in pred_data.items():
        if vid not in gt_map:
            continue
        gt_info = gt_map[vid]
        gt_bounds = gt_info.get("boundaries", [])
        mp4_path = gt_info.get("mp4_path")

        # 1. Check False Negatives (Missed GTs)
        for g_idx, g in enumerate(gt_bounds):
            if not preds:
                dist = 999.0
            else:
                dist = min(abs(g - p) for p in preds)

            if dist > tolerance:
                severity = dist if dist < 999.0 else 5.0
                candidates.append({
                    "case_id": f"{vid}_FN_t{g:.2f}s",
                    "video_id": vid,
                    "category": "false_negative",
                    "error_timestamp": g,
                    "nearest_pred_distance": round(dist, 3) if dist < 999.0 else None,
                    "severity_score": round(severity, 3),
                    "description": f"Ground-truth boundary at {g:.2f}s completely missed by model (closest prediction was {dist:.2f}s away)." if dist < 999.0 else f"Ground-truth boundary at {g:.2f}s completely missed (no nearby predictions).",
                    "mp4_path": mp4_path,
                })

        # 2. Check False Positives (Spurious Predictions)
        for p in preds:
            if not gt_bounds:
                dist = 999.0
            else:
                dist = min(abs(p - g) for g in gt_bounds)

            if dist > tolerance:
                severity = dist if dist < 999.0 else 5.0
                candidates.append({
                    "case_id": f"{vid}_FP_t{p:.2f}s",
                    "video_id": vid,
                    "category": "false_positive",
                    "error_timestamp": p,
                    "nearest_gt_distance": round(dist, 3) if dist < 999.0 else None,
                    "severity_score": round(severity, 3),
                    "description": f"Spurious boundary predicted at {p:.2f}s where no ground-truth boundary exists (closest GT is {dist:.2f}s away).",
                    "mp4_path": mp4_path,
                })

    # Sort descending by severity score
    candidates.sort(key=lambda c: c["severity_score"], reverse=True)

    # Balance top cases between False Positives and False Negatives
    fps = [c for c in candidates if c["category"] == "false_positive"]
    fns = [c for c in candidates if c["category"] == "false_negative"]

    selected = []
    # Pick top FP and top FN alternately up to top_k
    i = 0
    while len(selected) < top_k and (i < len(fps) or i < len(fns)):
        if i < len(fns) and len(selected) < top_k:
            selected.append(fns[i])
        if i < len(fps) and len(selected) < top_k:
            selected.append(fps[i])
        i += 1

    return selected


# =============================================================================
# Pillar 3: Visual Frame & Temporal Strip Extraction
# =============================================================================

def extract_visual_frames_for_error(case: dict, out_dir: Path) -> dict:
    """Extracts the exact event frame and a 3-frame temporal strip [t-0.3s, t, t+0.3s]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = case.get("mp4_path")
    if not mp4_path or not Path(mp4_path).exists():
        return {"visual_extracted": False, "reason": "MP4 video not found"}

    t = case["error_timestamp"]
    vid = case["video_id"]
    cat = case["category"]
    prefix = f"{vid}_{cat[:2].upper()}_t{t:.2f}s"
    event_frame_file = out_dir / f"{prefix}_event.jpg"
    strip_file = out_dir / f"{prefix}_strip.jpg"

    # Extract 3 timestamps: t - 0.3s, t, t + 0.3s
    t_prev = max(0.0, t - 0.35)
    t_cur = t
    t_next = t + 0.35

    temp_prev = out_dir / f"{prefix}_temp_prev.jpg"
    temp_cur = out_dir / f"{prefix}_temp_cur.jpg"
    temp_next = out_dir / f"{prefix}_temp_next.jpg"

    def extract_single_frame(time_s: float, target: Path):
        cmd = [
            "ffmpeg", "-y", "-ss", f"{time_s:.3f}",
            "-i", str(mp4_path),
            "-vframes", "1",
            "-q:v", "2",
            str(target)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return target.exists()

    ok_cur = extract_single_frame(t_cur, temp_cur)
    ok_prev = extract_single_frame(t_prev, temp_prev)
    ok_next = extract_single_frame(t_next, temp_next)

    if not ok_cur:
        return {"visual_extracted": False, "reason": "Failed to extract frame via ffmpeg"}

    # Copy current frame as the main event frame
    if temp_cur.exists():
        temp_cur.replace(event_frame_file)

    # Combine into a 3-frame horizontal strip if Image/cv2 is available, or keep event frame
    try:
        from PIL import Image, ImageDraw, ImageFont
        if ok_prev and event_frame_file.exists() and ok_next:
            im_prev = Image.open(temp_prev)
            im_cur = Image.open(event_frame_file)
            im_next = Image.open(temp_next)

            # Resize to uniform height 360
            h_target = 360
            def resize_keep_ratio(img):
                w, h = img.size
                new_w = int(w * (h_target / h))
                return img.resize((new_w, h_target), Image.Resampling.LANCZOS)

            p_img = resize_keep_ratio(im_prev)
            c_img = resize_keep_ratio(im_cur)
            n_img = resize_keep_ratio(im_next)

            total_w = p_img.width + c_img.width + n_img.width + 20
            strip = Image.new("RGB", (total_w, h_target + 50), color=(30, 30, 30))
            strip.paste(p_img, (0, 45))
            strip.paste(c_img, (p_img.width + 10, 45))
            strip.paste(n_img, (p_img.width + c_img.width + 20, 45))

            draw = ImageDraw.Draw(strip)
            banner_text = f"[{cat.upper()}] Video: {vid} | Event at {t:.2f}s | Sequence: {t_prev:.2f}s -> {t:.2f}s -> {t_next:.2f}s"
            draw.text((15, 12), banner_text, fill=(255, 255, 255))

            strip.save(strip_file, quality=90)
    except Exception:
        pass

    # Clean up temp files
    for tmp in [temp_prev, temp_next]:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

    return {
        "visual_extracted": True,
        "event_frame_path": str(event_frame_file.resolve()),
        "strip_frame_path": str(strip_file.resolve()) if strip_file.exists() else str(event_frame_file.resolve()),
        "event_frame_rel": str(event_frame_file.relative_to(REPO_ROOT)),
        "strip_frame_rel": str(strip_file.relative_to(REPO_ROOT)) if strip_file.exists() else str(event_frame_file.relative_to(REPO_ROOT)),
    }


# =============================================================================
# Pillar 4: Generate Markdown Report & JSON Fact Sheet
# =============================================================================

def generate_markdown_report(
    eval_data: dict,
    train_log_data: dict,
    severe_errors: list[dict],
    out_file: Path,
):
    """Formats full structured evaluation report following Research Loop standards."""
    pm = eval_data["primary_metrics"]
    pw = eval_data["primary_window"]
    
    md_lines = [
        f"# Structured Evaluation Report: {eval_data.get('iteration_id', 'iter_01')}",
        "",
        f"- **Track**: `{eval_data.get('track', 'step_segment')}`",
        f"- **Experiment ID**: `{eval_data.get('exp_id', 'exp_001_baseline')}`",
        f"- **Model Type**: `{eval_data.get('model_type', 'unspecified')}`",
        f"- **Evaluated Output Directory**: `{eval_data.get('output_dir', '')}`",
        f"- **Timestamp**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Executive Performance Fact Sheet",
        "",
        f"| Tolerance Window | Macro F1 | Recall | Precision | Status |",
        f"| :--- | :--- | :--- | :--- | :--- |",
    ]

    for th in [0.25, 0.5, 1.0]:
        th_key = f"{th}".replace(".", "_")
        f1 = pm.get(f"f1_at_{th_key}s", 0.0) * 100
        rec = pm.get(f"recall_at_{th_key}s", 0.0) * 100
        prec = pm.get(f"precision_at_{th_key}s", 0.0) * 100
        icon = "🎯" if f1 >= 70.0 else ("⚠️" if f1 >= 50.0 else "❌")
        primary_marker = " **(Primary)**" if th == pw else ""
        md_lines.append(f"| **+/- {th:.2f}s**{primary_marker} | **{f1:.2f}%** | {rec:.2f}% | {prec:.2f}% | {icon} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Training Log Health & Dynamics Analysis",
        "",
    ])

    if train_log_data.get("status") == "success":
        init_l = f"{train_log_data['initial_loss']:.4f}" if train_log_data.get("initial_loss") is not None else "N/A"
        final_l = f"{train_log_data['final_loss']:.4f}" if train_log_data.get("final_loss") is not None else "N/A"
        min_l = f"{train_log_data['min_loss']:.4f}" if train_log_data.get("min_loss") is not None else "N/A"
        peak_f1 = f"{train_log_data['peak_f1']*100:.2f}%" if train_log_data.get("peak_f1") is not None else "N/A"

        md_lines.extend([
            f"- **Recorded Epochs**: {train_log_data.get('total_epochs_recorded', 'N/A')}",
            f"- **Loss Trajectory**: Initial `{init_l}` -> Final `{final_l}` (Minimum: `{min_l}`)",
            f"- **Peak Validation F1**: `{peak_f1}`",
            "",
            "### Detected Training Anomalies:",
        ])
        anomalies = train_log_data.get("anomalies", [])
        if anomalies:
            for an in anomalies:
                md_lines.append(f"- > [!WARNING]")
                md_lines.append(f"  > **[{an['severity']}] {an['type']}**: {an['description']}")
        else:
            md_lines.append("- ✓ No critical training anomalies detected (smooth loss convergence, no NaN/divergence).")
    else:
        md_lines.append(f"- *Note*: {train_log_data.get('message', 'Training log not available.')}")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Top Severe Error Cases (Visual Evidence)",
        "",
        "These cases represent the most severe failure modes (missed transitions or false triggers) prioritized for root-cause diagnosis:",
        "",
    ])

    for idx, case in enumerate(severe_errors, 1):
        cat_badge = "🔴 FALSE NEGATIVE (MISSED GT)" if case["category"] == "false_negative" else "🟠 FALSE POSITIVE (SPURIOUS)"
        dist_info = f"Nearest prediction is {case['nearest_pred_distance']}s away" if case.get("nearest_pred_distance") is not None else f"Nearest GT is {case['nearest_gt_distance']}s away"

        md_lines.extend([
            f"### Case {idx}: `{case['case_id']}` ({cat_badge})",
            f"- **Video ID**: `{case['video_id']}`",
            f"- **Event Timestamp**: `{case['error_timestamp']:.2f}s`",
            f"- **Discrepancy**: {dist_info} (Severity Score: `{case['severity_score']}`)",
            f"- **Factual Observation**: {case['description']}",
        ])

        visual = case.get("visual_info", {})
        if visual.get("visual_extracted"):
            strip_path = visual.get("strip_frame_rel", "")
            event_path = visual.get("event_frame_rel", "")
            md_lines.extend([
                f"- **Visual Frame Sequence**: [View Strip Image](file:///{REPO_ROOT / strip_path})",
                f"  ![Visual Frame Strip]({REPO_ROOT / strip_path})",
            ])
        else:
            md_lines.append(f"- **Visual Frame**: *(Video file not located for direct extraction)*")

        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 4. Per-Video Breakdown Ranking (Worst to Best)",
        "",
        "| Rank | Video ID | F1 @ 0.5s | Recall | Precision | GT Count | Pred Count |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    pw_vids = eval_data.get("per_video_performance", {})
    sorted_vids = sorted(pw_vids.items(), key=lambda x: (x[1]["f1"], x[1]["recall"]))
    for rk, (vid, st) in enumerate(sorted_vids, 1):
        md_lines.append(f"| {rk} | `{vid}` | **{st['f1']*100:.1f}%** | {st['recall']*100:.1f}% | {st['precision']*100:.1f}% | {st['num_gt']} | {st['num_pred']} |")

    md_lines.extend([
        "",
        "---",
        "",
        "> [!CAUTION]",
        "> **Evaluator Protocol Rule**: This report contains strictly observable metrics and extracted visual evidence. Speculation on root causes (why it happened) is deferred to `@research-diagnoser`, and solutions belong to `@research-planner`.",
        "",
    ])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(md_lines), encoding="utf-8")


# =============================================================================
# Main Controller
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Step Segmentation Evaluation and Error Inspection Tool.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to experiment outputs (containing predictions.json and train.log)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Path to ground truth data/ directory")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory to save 01_eval_report.md and eval_report.json")
    parser.add_argument("--track", type=str, default="step_segment")
    parser.add_argument("--iter-id", type=str, default="iter_01")
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="unspecified")
    parser.add_argument("--top-k-errors", type=int, default=4, help="Number of severe error cases to prioritize (default: 4)")
    parser.add_argument("--viz", action="store_true", help="Also generate score_curve.png and annotated.mp4 for error videos")
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    exp_id = args.exp_id or out_dir.name
    report_dir = args.report_dir or (REPO_ROOT / "experiments" / args.track / args.iter_id)

    # 1. Locate predictions
    pred_file = out_dir / "predictions.json"
    if not pred_file.exists():
        # Look for alternate names
        candidates = list(out_dir.glob("*preds*.json"))
        if candidates:
            pred_file = candidates[0]
        else:
            print(f"[ERROR] Could not find predictions.json inside {out_dir}")
            sys.exit(1)

    print("=" * 70)
    print(" COMPREHENSIVE EXPERIMENT EVALUATOR & VISUAL INSPECTOR")
    print(f" Experiment: {exp_id} | Track: {args.track} | Iteration: {args.iter_id}")
    print(f" Source:     {out_dir}")
    print("=" * 70)

    # 2. Parse Training Logs (Khía cạnh 1)
    print("\n [1/4] Analyzing Training Log Health...")
    log_candidates = [out_dir / "train.log", out_dir / "run.log"]
    chosen_log = None
    for lc in log_candidates:
        if lc.exists() and lc.stat().st_size > 0:
            chosen_log = lc
            break
    train_log_data = parse_training_log(chosen_log) if chosen_log else {"status": "not_found", "message": "No log file found."}
    if train_log_data.get("anomalies"):
        print(f"   ⚠️  Found {len(train_log_data['anomalies'])} training anomalies!")
    else:
        print("   ✓ Training log healthy (loss trajectory normal).")

    # 3. Compute Validation Metrics (Khía cạnh 2)
    print("\n [2/4] Computing Multi-tolerance Validation Metrics...")
    pred_data = json.loads(pred_file.read_text(encoding="utf-8"))
    gt_map = load_all_ground_truth(args.data_dir)

    thresholds = [0.25, 0.5, 1.0]
    primary_window = 0.5

    results_by_th = {}
    for th in thresholds:
        tp_all = num_gt_all = num_pred_all = 0
        per_video_th = {}
        for vid, p_list in pred_data.items():
            g_info = gt_map.get(vid)
            if not g_info or not g_info["boundaries"]:
                continue
            g_list = g_info["boundaries"]
            tp_pairs, fp_list, fn_list = match_boundaries(g_list, p_list, th)
            tp = len(tp_pairs)
            num_gt = len(g_list)
            num_pred = len(p_list)
            rec = tp / num_gt if num_gt else 0.0
            prec = tp / num_pred if num_pred else 0.0
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
            per_video_th[vid] = {
                "tp": tp,
                "num_gt": num_gt,
                "num_pred": num_pred,
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
            }
            tp_all += tp
            num_gt_all += num_gt
            num_pred_all += num_pred

        ov_rec = tp_all / num_gt_all if num_gt_all else 0.0
        ov_prec = tp_all / num_pred_all if num_pred_all else 0.0
        ov_f1 = 2 * ov_rec * ov_prec / (ov_rec + ov_prec) if (ov_rec + ov_prec) else 0.0
        results_by_th[str(th)] = {
            "recall": round(ov_rec, 4),
            "precision": round(ov_prec, 4),
            "f1": round(ov_f1, 4),
            "tp": tp_all,
            "num_gt": num_gt_all,
            "num_pred": num_pred_all,
            "per_video": per_video_th,
        }

    primary_metrics = {}
    for th in thresholds:
        th_key = f"{th}".replace(".", "_")
        primary_metrics[f"f1_at_{th_key}s"] = results_by_th[str(th)]["f1"]
        primary_metrics[f"recall_at_{th_key}s"] = results_by_th[str(th)]["recall"]
        primary_metrics[f"precision_at_{th_key}s"] = results_by_th[str(th)]["precision"]

    primary_metrics["macro_f1"] = results_by_th[str(primary_window)]["f1"]
    primary_metrics["macro_recall"] = results_by_th[str(primary_window)]["recall"]
    primary_metrics["macro_precision"] = results_by_th[str(primary_window)]["precision"]

    print(f"   ✓ Macro F1@0.5s: {primary_metrics['macro_f1']*100:.2f}% | Recall: {primary_metrics['macro_recall']*100:.2f}% | Precision: {primary_metrics['macro_precision']*100:.2f}%")

    # 4. Extract Top Severe Error Cases (Khía cạnh 2 - Visual Evidence)
    print(f"\n [3/4] Extracting Top {args.top_k_errors} Severe Error Cases & Visual Frames...")
    severe_errors = extract_severe_error_cases(gt_map, pred_data, top_k=args.top_k_errors, tolerance=primary_window)
    error_frames_dir = out_dir / "error_cases"

    for case in severe_errors:
        visual_info = extract_visual_frames_for_error(case, error_frames_dir)
        case["visual_info"] = visual_info
        if visual_info.get("visual_extracted"):
            print(f"   ✓ Extracted visual frames for {case['case_id']}")

    # 5. Optional Visualization rendering
    if args.viz:
        print("\n [4/4] Rendering Score Curves & Annotated Visualizations...")
        viz_dir = out_dir / "visualizations"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "visualize_step_segment_results.py"),
            "--pred-file", str(pred_file),
            "--out-dir", str(viz_dir),
            "--top-n", str(min(3, len(pred_data))),
            "--tolerance", str(primary_window),
            "--render-video",
        ]
        subprocess.run(cmd, check=False)

    # 6. Save Reports
    eval_report_data = {
        "track": args.track,
        "iteration_id": args.iter_id,
        "exp_id": exp_id,
        "model_type": args.model_type,
        "output_dir": str(out_dir),
        "primary_window": primary_window,
        "tolerance_windows": thresholds,
        "primary_metrics": primary_metrics,
        "training_log_health": train_log_data,
        "per_video_performance": results_by_th[str(primary_window)]["per_video"],
        "severe_error_cases": severe_errors,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    json_report_file = report_dir / "eval_report.json"
    json_report_file.write_text(json.dumps(eval_report_data, indent=2), encoding="utf-8")
    print(f"\n ✓ Saved Structured JSON Report -> {json_report_file}")

    md_report_file = report_dir / "01_eval_report.md"
    generate_markdown_report(eval_report_data, train_log_data, severe_errors, md_report_file)
    print(f" ✓ Saved Markdown Evaluation Report -> {md_report_file}")

    print("\n" + "=" * 70)
    print(" EVALUATION & VISUAL ERROR INSPECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
