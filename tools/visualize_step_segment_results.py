#!/usr/bin/env python3
"""Unified Visualization Tool for Step Segmentation Models (DiffGEBD, EfficientGEBD, DDM-Net).

Generates:
  1. `score_curve.png` (or .svg fallback): Timeline curve showing GT vs Pred boundaries,
     tolerance matching zones, and highlighted False Positives / False Negatives.
  2. `annotated.mp4`: Video with HUD overlay showing current timestamp, progress bar,
     and alerts when approaching or crossing GT vs Predicted boundaries.

Usage:
  python tools/visualize_step_segment_results.py \\
      --pred-file outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/predictions.json \\
      --out-dir outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/visualizations \\
      --top-n 3 --render-video
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def is_unknown(name: str | None) -> bool:
    return not name or not name.strip() or name.strip().upper() in ["UNKNOWN", "NONE"]


def gt_boundaries_for_video(step_segments: dict) -> list[float]:
    segs = step_segments.get("segments", [])
    if not segs:
        return []
    start_idx, end_idx = 0, len(segs)
    while start_idx < len(segs) and is_unknown(segs[start_idx].get("operation_name", "")):
        start_idx += 1
    while end_idx > start_idx and is_unknown(segs[end_idx - 1].get("operation_name", "")):
        end_idx -= 1
    segs = segs[start_idx:end_idx]
    bounds = []
    for i in range(1, len(segs)):
        prev_end = float(segs[i - 1]["end_time_s"])
        cur_start = float(segs[i]["start_time_s"])
        bounds.append(round((prev_end + cur_start) / 2.0, 3))
    return bounds


def load_all_ground_truth(data_dir: Path) -> dict[str, dict]:
    """Returns mapping: video_id -> {'boundaries': [...], 'mp4_path': Path, 'duration': float, 'fps': float}"""
    gt_map = {}
    for f in sorted(data_dir.rglob("step_segments.json")):
        parent = f.parent
        cd_name = parent.parent.name
        chuyen_name = parent.name
        vid_id = f"{cd_name}_{chuyen_name}"
        
        mp4_files = list(parent.glob("*.mp4"))
        mp4_path = mp4_files[0] if mp4_files else None

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            bounds = gt_boundaries_for_video(data)
            fps = float(data.get("fps", 15.0))
            duration = 0.0
            if data.get("segments"):
                duration = float(data["segments"][-1]["end_time_s"])
            
            gt_map[vid_id] = {
                "boundaries": bounds,
                "mp4_path": mp4_path,
                "duration": duration,
                "fps": fps,
                "stem": mp4_path.stem if mp4_path else vid_id,
            }
            if mp4_path:
                gt_map[mp4_path.stem] = gt_map[vid_id]
        except Exception as e:
            continue
    return gt_map


def match_boundaries(gt_bounds: list[float], pred_bounds: list[float], tol: float):
    """Matches pred to gt, categorizing into TP, FP, FN."""
    rem_pred = list(pred_bounds)
    tp_pairs = []
    fn_list = []
    
    for g in gt_bounds:
        if not rem_pred:
            fn_list.append(g)
            continue
        closest = min(rem_pred, key=lambda p: abs(p - g))
        if abs(closest - g) <= tol:
            tp_pairs.append((g, closest))
            rem_pred.remove(closest)
        else:
            fn_list.append(g)
    
    fp_list = rem_pred
    return tp_pairs, fp_list, fn_list


def render_score_curve_plt(
    vid_id: str,
    gt_bounds: list[float],
    pred_bounds: list[float],
    out_path: Path,
    tol: float = 0.5,
    duration: float = 60.0,
    scores_series: Optional[Tuple[list[float], list[float]]] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tp_pairs, fp_list, fn_list = match_boundaries(gt_bounds, pred_bounds, tol)
    
    max_time = max([duration] + gt_bounds + pred_bounds + [10.0])
    fig, ax = plt.subplots(figsize=(14, 4), dpi=120)

    # If continuous scores exist
    if scores_series is not None:
        times, scores = scores_series
        ax.plot(times, scores, color="tab:blue", label="Boundary Probability", alpha=0.8, linewidth=1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Probability / Score")
    else:
        # Discrete timeline representation
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])

    # Plot Ground Truth tolerance zones (Green bands)
    for g in gt_bounds:
        ax.axvspan(max(0, g - tol), g + tol, color="green", alpha=0.15, label="Tolerance Zone" if g == gt_bounds[0] else "")
        ax.axvline(g, color="green", linestyle="-", linewidth=2.0, alpha=0.8, label="GT Boundary" if g == gt_bounds[0] else "")

    # Plot Predictions
    for p in pred_bounds:
        matched = any(abs(p - g) <= tol for g in gt_bounds)
        line_color = "tab:green" if matched else "tab:red"
        line_style = "--" if matched else "-."
        label = "Pred (Matched)" if matched and p == pred_bounds[0] else ("Pred (False Positive)" if not matched and p == pred_bounds[0] else "")
        ax.axvline(p, color=line_color, linestyle=line_style, linewidth=1.8, alpha=0.9, label=label)

    # Highlight Missed GTs (False Negatives)
    for fn in fn_list:
        ax.scatter([fn], [0.95], color="red", marker="v", s=80, zorder=5, label="Missed GT (FN)" if fn == fn_list[0] else "")

    # Highlight False Positives
    for fp in fp_list:
        ax.scatter([fp], [1.05], color="darkorange", marker="^", s=80, zorder=5, label="Spurious Pred (FP)" if fp == fp_list[0] else "")

    ax.set_xlim(0, max_time)
    ax.set_xlabel("Time (seconds)")
    rec = len(tp_pairs) / len(gt_bounds) if gt_bounds else 0.0
    prec = len(tp_pairs) / len(pred_bounds) if pred_bounds else 0.0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0

    ax.set_title(f"Boundary Alignment: {vid_id} | F1@{tol}s: {f1*100:.1f}% (Rec: {rec*100:.1f}%, Prec: {prec*100:.1f}%) | GT: {len(gt_bounds)}, Pred: {len(pred_bounds)}", fontsize=11, fontweight="bold")
    
    # Clean unique labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, framealpha=0.9)

    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def render_score_curve_svg(
    vid_id: str,
    gt_bounds: list[float],
    pred_bounds: list[float],
    out_path: Path,
    tol: float = 0.5,
    duration: float = 60.0,
):
    """Pure-python SVG generator if matplotlib is not installed."""
    max_time = max([duration] + gt_bounds + pred_bounds + [10.0])
    w, h = 900, 220
    pad_l, pad_r, pad_t, pad_b = 60, 40, 40, 40
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    def t2x(t):
        return pad_l + (t / max_time) * plot_w

    svg_lines = [
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{w}" height="{h}" fill="#ffffff" />',
        f'<text x="{pad_l}" y="25" font-family="sans-serif" font-size="14" font-weight="bold" fill="#222">Boundary Alignment: {vid_id} (GT: {len(gt_bounds)}, Pred: {len(pred_bounds)})</text>',
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#666" stroke-width="2"/>',
    ]

    # Grid & time ticks
    step = 5.0 if max_time < 60 else (10.0 if max_time < 180 else 30.0)
    cur = 0.0
    while cur <= max_time:
        x = t2x(cur)
        svg_lines.append(f'<line x1="{x}" y1="{pad_t}" x2="{x}" y2="{pad_t + plot_h}" stroke="#eee" stroke-width="1"/>')
        svg_lines.append(f'<text x="{x}" y="{pad_t + plot_h + 20}" font-family="sans-serif" font-size="11" fill="#666" text-anchor="middle">{cur:.0f}s</text>')
        cur += step

    # Tolerance zones & GT
    for g in gt_bounds:
        x_min = t2x(max(0, g - tol))
        x_max = t2x(g + tol)
        svg_lines.append(f'<rect x="{x_min}" y="{pad_t}" width="{x_max - x_min}" height="{plot_h}" fill="#4caf50" opacity="0.25"/>')
        x_gt = t2x(g)
        svg_lines.append(f'<line x1="{x_gt}" y1="{pad_t}" x2="{x_gt}" y2="{pad_t + plot_h}" stroke="#2e7d32" stroke-width="2.5"/>')

    # Pred lines
    for p in pred_bounds:
        x_p = t2x(p)
        matched = any(abs(p - g) <= tol for g in gt_bounds)
        color = "#388e3c" if matched else "#d32f2f"
        dash = 'stroke-dasharray="4,4"' if matched else 'stroke-dasharray="2,2"'
        svg_lines.append(f'<line x1="{x_p}" y1="{pad_t}" x2="{x_p}" y2="{pad_t + plot_h}" stroke="{color}" stroke-width="2" {dash}/>')

    svg_lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_lines), encoding="utf-8")


def render_annotated_video(
    video_path: Path,
    gt_bounds: list[float],
    pred_bounds: list[float],
    out_path: Path,
    tol: float = 0.5,
    max_duration: float = 120.0,
    pause_seconds: float = 1.0,
):
    """Renders annotated video using OpenCV or FFmpeg overlay, pausing on predicted boundaries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"   [WARN] Could not open video {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(min(total_frames, max_duration * fps))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # Precompute target frame for each predicted boundary
        pred_frame_map = {}
        for p in sorted(pred_bounds):
            f = int(round(p * fps))
            if f not in pred_frame_map:
                pred_frame_map[f] = p

        pause_frames = max(1, int(round(pause_seconds * fps))) if pause_seconds > 0 else 0

        for f_idx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            cur_time = f_idx / fps

            # Check status at current time
            near_gt = any(abs(cur_time - g) <= tol for g in gt_bounds)
            near_pred = any(abs(cur_time - p) <= tol for p in pred_bounds)

            # Draw HUD bar at top
            cv2.rectangle(frame, (0, 0), (width, 50), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"Time: {cur_time:6.2f}s | Frame: {f_idx:5d}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Banner if near boundary
            if near_gt and near_pred:
                cv2.rectangle(frame, (width - 340, 5), (width - 10, 45), (46, 139, 87), -1)
                cv2.putText(frame, "MATCHED BOUNDARY (TP)", (width - 330, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            elif near_pred and not near_gt:
                cv2.rectangle(frame, (width - 340, 5), (width - 10, 45), (0, 140, 255), -1)
                cv2.putText(frame, "FALSE POSITIVE (FP)", (width - 325, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            elif near_gt and not near_pred:
                cv2.rectangle(frame, (width - 340, 5), (width - 10, 45), (0, 0, 205), -1)
                cv2.putText(frame, "MISSED GT BOUNDARY (FN)", (width - 335, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            writer.write(frame)

            # If current frame is an exact predicted boundary -> FREEZE / PAUSE for pause_seconds
            if pause_frames > 0 and f_idx in pred_frame_map:
                p_time = pred_frame_map[f_idx]
                is_matched = any(abs(p_time - g) <= tol for g in gt_bounds)
                border_color = (46, 139, 87) if is_matched else (0, 140, 255)

                freeze_frame = frame.copy()
                # Frame border highlight
                cv2.rectangle(freeze_frame, (0, 0), (width - 1, height - 1), border_color, 8)

                # Center highlight badge
                badge_w, badge_h = 500, 80
                badge_x = (width - badge_w) // 2
                badge_y = 60
                cv2.rectangle(freeze_frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (20, 20, 20), -1)
                cv2.rectangle(freeze_frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), border_color, 3)

                status_text = "MATCHED BOUNDARY (TP)" if is_matched else "FALSE POSITIVE (FP)"
                cv2.putText(freeze_frame, f"|| PAUSED {pause_seconds:.1f}s | PRED: {p_time:.2f}s", (badge_x + 15, badge_y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(freeze_frame, f"Status: {status_text}", (badge_x + 15, badge_y + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_color, 2)

                for _ in range(pause_frames):
                    writer.write(freeze_frame)

        cap.release()
        writer.release()
        return True

    except ImportError:
        # Fallback to ffmpeg drawtext filter
        print("   [INFO] OpenCV not available, falling back to FFmpeg for video annotation...")
        filt = f"drawtext=text='Time\\: %{{pts\\:hms}}':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6"
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", filt,
            "-t", str(max_duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            str(out_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path.exists()


def main():
    parser = argparse.ArgumentParser(description="Unified Step Segmentation Visualizer for DiffGEBD, EfficientGEBD, and DDM-Net.")
    parser.add_argument("--pred-file", type=Path, required=True, help="Path to predictions.json or diffgebd_preds.json")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Path to dataset containing step_segments.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to save visualizations")
    parser.add_argument("--video-id", type=str, default=None, help="Specific video_id to visualize (comma-separated for multiple)")
    parser.add_argument("--top-n", type=int, default=3, help="Number of worst-performing videos to visualize (default: 3)")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Tolerance window in seconds (default: 0.5s)")
    parser.add_argument("--render-video", action="store_true", help="Render annotated.mp4 video")
    parser.add_argument("--render-curves", action="store_true", default=True, help="Render score_curve.png")
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Pause duration in seconds when reaching a predicted boundary (default: 1.0s, set 0 to disable)")
    args = parser.parse_args()

    pred_data = json.loads(args.pred_file.read_text(encoding="utf-8"))
    out_dir = args.out_dir or (args.pred_file.parent / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_map = load_all_ground_truth(args.data_dir)

    # Determine videos to visualize
    target_vids = []
    if args.video_id:
        target_vids = [v.strip() for v in args.video_id.split(",") if v.strip()]
    else:
        # Score each video to find worst N
        scored = []
        for vid, preds in pred_data.items():
            if vid in gt_map and gt_map[vid]["boundaries"]:
                gt_b = gt_map[vid]["boundaries"]
                tp, fp, fn = match_boundaries(gt_b, preds, args.tolerance)
                rec = len(tp) / len(gt_b) if gt_b else 0.0
                prec = len(tp) / len(preds) if preds else 0.0
                f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
                scored.append((f1, len(fp) + len(fn), vid))
        
        # Sort ascending by F1, then descending by error count
        scored.sort(key=lambda x: (x[0], -x[1]))
        target_vids = [s[2] for s in scored[:args.top_n]]

    print("=" * 65)
    print(f" UNIFIED VISUALIZATION FOR STEP SEGMENTATION")
    print(f" Target Videos ({len(target_vids)}): {', '.join(target_vids)}")
    print(f" Output Directory: {out_dir}")
    print(f" Boundary Pause:   {args.pause_seconds:.1f}s")
    print("=" * 65)

    has_plt = False
    try:
        import matplotlib
        has_plt = True
    except ImportError:
        pass

    for vid in target_vids:
        preds = pred_data.get(vid, [])
        gt_info = gt_map.get(vid, {})
        gt_b = gt_info.get("boundaries", [])
        mp4_path = gt_info.get("mp4_path")
        duration = gt_info.get("duration", 60.0)

        vid_out_dir = out_dir / vid
        vid_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n → Processing {vid}: {len(gt_b)} GT boundaries, {len(preds)} Pred boundaries")

        # 1. Render score curve
        if args.render_curves:
            if has_plt:
                curve_file = vid_out_dir / "score_curve.png"
                render_score_curve_plt(vid, gt_b, preds, curve_file, tol=args.tolerance, duration=duration)
                print(f"   ✓ Generated score curve: {curve_file.name}")
            else:
                curve_file = vid_out_dir / "score_curve.svg"
                render_score_curve_svg(vid, gt_b, preds, curve_file, tol=args.tolerance, duration=duration)
                print(f"   ✓ Generated score curve (SVG): {curve_file.name}")

        # 2. Render annotated video
        if args.render_video:
            if mp4_path and mp4_path.exists():
                vid_file = vid_out_dir / "annotated.mp4"
                ok = render_annotated_video(mp4_path, gt_b, preds, vid_file, tol=args.tolerance, pause_seconds=args.pause_seconds)
                if ok:
                    print(f"   ✓ Generated annotated video (with {args.pause_seconds:.1f}s pause on predictions): {vid_file.name}")
            else:
                print(f"   ✗ Source video not found for {vid}, skipping MP4 render.")

    print("\n" + "=" * 65)
    print(f" Visualization completed! Results stored in {out_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
