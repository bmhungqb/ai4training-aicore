#!/usr/bin/env python3
"""Visualize a sample from dataset_tas_instance as an annotated .mp4 video.

Features:
- Dual-track timeline at bottom:
    Track 1: Instance-Level GT (annotations/*.json) with explicit same_class vs class_change boundaries
    Track 2: Frame-Level Coarse GT (groundTruth/*.txt) showing the collapse of adjacent same-class instances
- Live playhead cursor & timestamp / frame progress
- Top HUD with current active instance, class badge, duration, and intra-instance progress
- Dynamic boundary alert card during instance transitions (highlighting same_class transitions)
- Color-coded action class legend
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Add python3.14 dist-packages only if running on python 3.14+
if sys.version_info >= (3, 14):
    snap_dist_pkg = Path("/home/hungbm/snap/antigravity-cli/common/local/lib/python3.14/dist-packages")
    if snap_dist_pkg.exists() and str(snap_dist_pkg) not in sys.path:
        sys.path.insert(0, str(snap_dist_pkg))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# Color Definitions (BGR for OpenCV, RGB for PIL)
# ---------------------------------------------------------
CLASS_COLORS_RGB = {
    0: (46, 204, 113),    # Sewing/Joining -> Emerald Green
    1: (52, 152, 219),    # Positioning/Handling -> Dodger Blue
    2: (243, 156, 18),    # Adjustment/Alignment/Preparation -> Amber / Orange
    3: (155, 89, 182),    # Inspection/Auxiliary -> Amethyst Purple
    -1: (127, 140, 141),  # Background / Unknown -> Gray
}

BOUNDARY_COLORS_RGB = {
    "same_class": (0, 240, 255),    # Bright Cyan for same-class boundary
    "class_change": (255, 220, 40), # Vivid Gold/Yellow for class-change
}

# TTF Font paths
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


def load_dataset_metadata(dataset_dir: Path, video_id: str):
    """Load JSON annotation, coarse groundTruth, frame boundaries, and classes."""
    ann_path = dataset_dir / "annotations" / f"{video_id}.json"
    gt_path = dataset_dir / "groundTruth" / f"{video_id}.txt"
    bnd_path = dataset_dir / "boundaries" / f"{video_id}.txt"
    classes_path = dataset_dir / "classes.json"
    video_path = dataset_dir / "videos" / f"{video_id}.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    classes_map = {}
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            classes_list = json.load(f)
            classes_map = {item["class_id"]: item["class_name"] for item in classes_list}

    gt_labels = []
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_labels = [line.strip() for line in f if line.strip()]

    frame_boundaries = []
    if bnd_path.exists():
        with open(bnd_path, "r", encoding="utf-8") as f:
            frame_boundaries = [int(line.strip()) for line in f if line.strip()]

    return {
        "video_path": video_path,
        "annotation": annotation,
        "classes_map": classes_map,
        "gt_labels": gt_labels,
        "frame_boundaries": frame_boundaries,
    }


def render_visualization(
    dataset_dir: Path,
    video_id: str,
    output_path: Path,
    target_width: int = 1280,
    target_height: int = 720,
    fps_override: float | None = None,
):
    print(f"Loading metadata for video: {video_id} ...")
    meta = load_dataset_metadata(dataset_dir, video_id)
    ann = meta["annotation"]
    instances = ann.get("instances", [])
    boundaries = ann.get("boundaries", [])
    gt_labels = meta["gt_labels"]
    classes_map = meta["classes_map"]

    cap = cv2.VideoCapture(str(meta["video_path"]))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {meta['video_path']}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = fps_override or (ann.get("fps") or cap_fps)

    # Determine total frames according to annotation or video
    total_frames = ann.get("num_frames") or cap_frame_count or 1
    total_duration = ann.get("duration", total_frames / video_fps)

    print(
        f"Video specs: {orig_w}x{orig_h} @ {cap_fps:.2f}fps ({cap_frame_count} frames decoded)."
    )
    print(
        f"Annotation: {total_frames} frames, duration={total_duration:.2f}s, {len(instances)} instances, {len(boundaries)} boundaries."
    )
    print(f"Target rendering size: {target_width}x{target_height} @ {video_fps:.2f} fps")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, (target_width, target_height))

    # Pre-calculate boundary lookup (boundary at frame F triggers alert for [F - alert_window, F + alert_window])
    alert_window = max(3, int(video_fps * 0.6))  # ~0.6 second window
    boundary_alerts = {}
    for b in boundaries:
        b_frame = b["frame"]
        for f in range(b_frame - alert_window, b_frame + alert_window + 1):
            if f not in boundary_alerts or abs(f - b_frame) < abs(f - boundary_alerts[f]["frame"]):
                boundary_alerts[f] = b

    # Map class name to ID
    name_to_cid = {v: k for k, v in classes_map.items()}

    # Timeline layout metrics
    tl_margin_x = 50
    tl_w = target_width - 2 * tl_margin_x
    tl_h_track = 26
    tl_gap = 22
    tl_bottom_margin = 20
    tl_y_track2 = target_height - tl_bottom_margin - tl_h_track
    tl_y_track1 = tl_y_track2 - tl_gap - tl_h_track
    tl_panel_y = tl_y_track1 - 38
    tl_panel_h = target_height - tl_panel_y - 8

    # Fonts
    font_title = get_font(18, bold=True)
    font_bold = get_font(14, bold=True)
    font_regular = get_font(13, bold=False)
    font_small = get_font(11, bold=False)
    font_badge = get_font(13, bold=True)
    font_alert = get_font(16, bold=True)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Resize video frame to target dimensions
        if frame_bgr.shape[1] != target_width or frame_bgr.shape[0] != target_height:
            frame_resized = cv2.resize(
                frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
        else:
            frame_resized = frame_bgr.copy()

        # Convert to RGB PIL Image for overlay rendering
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        overlay = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)

        # Current time
        cur_time = frame_idx / video_fps
        # Normalized position in total frames
        norm_pos = min(1.0, max(0.0, frame_idx / max(1, total_frames - 1)))

        # Find current active instance
        active_inst = None
        for inst in instances:
            if inst["start_frame"] <= frame_idx < inst["end_frame"]:
                active_inst = inst
                break
        if active_inst is None and instances:
            if frame_idx < instances[0]["start_frame"]:
                active_inst = instances[0]
            else:
                active_inst = instances[-1]

        # -------------------------------------------------------------
        # 1. TOP HEADER PANEL (Semi-transparent HUD)
        # -------------------------------------------------------------
        top_h = 100
        draw_ov.rectangle([0, 0, target_width, top_h], fill=(15, 23, 42, 220))  # Slate dark
        draw_ov.line([0, top_h, target_width, top_h], fill=(51, 65, 85, 255), width=2)

        # Title & Video ID
        draw_ov.text((25, 10), f"TAS INSTANCE DATASET — SAMPLE: {video_id}", fill=(255, 255, 255), font=font_title)
        time_str = f"Time: {cur_time:05.2f}s / {total_duration:05.2f}s   |   Frame: {frame_idx:03d} / {total_frames:03d}   |   FPS: {video_fps:.1f}"
        draw_ov.text((25, 34), time_str, fill=(148, 163, 184), font=font_regular)

        # Active Instance Card in Header
        if active_inst:
            cid = active_inst.get("class_id", -1)
            cname = active_inst.get("class_name", "Background")
            col = CLASS_COLORS_RGB.get(cid, CLASS_COLORS_RGB[-1])
            inst_id = active_inst.get("id", 0)
            inst_len_f = active_inst["end_frame"] - active_inst["start_frame"]
            inst_len_s = active_inst["end_time"] - active_inst["start_time"]
            inst_prog = (frame_idx - active_inst["start_frame"]) / max(1, inst_len_f)
            inst_prog = min(1.0, max(0.0, inst_prog))

            # Dynamic pill badge width
            badge_text = f"Instance #{inst_id}: {cname.upper()}"
            bbox = font_badge.getbbox(badge_text)
            text_w = bbox[2] - bbox[0]
            badge_w = text_w + 24
            badge_x = 25
            badge_y = 60
            badge_h = 28
            draw_ov.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], radius=6, fill=(*col, 240))
            draw_ov.text((badge_x + 12, badge_y + 5), badge_text, fill=(255, 255, 255), font=font_badge)

            # Instance time & progress
            sub_info = f"Frames [{active_inst['start_frame']} - {active_inst['end_frame']}] ({inst_len_s:.2f}s)   •   Progress: {int(inst_prog * 100)}%"
            draw_ov.text((badge_x + badge_w + 16, badge_y + 6), sub_info, fill=(226, 232, 240), font=font_regular)

        # -------------------------------------------------------------
        # 2. TOP RIGHT LEGEND
        # -------------------------------------------------------------
        leg_w = 330
        leg_x = target_width - leg_w - 20
        leg_y = 10
        leg_h = 80
        draw_ov.rounded_rectangle([leg_x, leg_y, leg_x + leg_w, leg_y + leg_h], radius=8, fill=(10, 15, 30, 220), outline=(51, 65, 85, 200), width=1)
        draw_ov.text((leg_x + 12, leg_y + 5), "CLASS LEGEND", fill=(203, 213, 225), font=font_small)

        # 2x2 grid for 4 classes
        classes_items = sorted(classes_map.items())[:4]
        for idx, (cid, cname) in enumerate(classes_items):
            row = idx // 2
            col = idx % 2
            item_x = leg_x + 12 + col * 160
            item_y = leg_y + 22 + row * 18
            ccol = CLASS_COLORS_RGB.get(cid, (200, 200, 200))
            draw_ov.rectangle([item_x, item_y + 2, item_x + 10, item_y + 12], fill=(*ccol, 255))
            short_name = cname.replace("Adjustment/Alignment/Preparation", "Adjustment/Prep").replace("Positioning/Handling", "Positioning").replace("Inspection/Auxiliary", "Inspection")
            draw_ov.text((item_x + 15, item_y), short_name, fill=(241, 245, 249), font=font_small)

        # Boundary indicators in legend
        bnd_y = leg_y + 58
        draw_ov.rectangle([leg_x + 12, bnd_y + 2, leg_x + 20, bnd_y + 10], fill=(*BOUNDARY_COLORS_RGB["same_class"], 255))
        draw_ov.text((leg_x + 24, bnd_y), "Same-Class Boundary", fill=BOUNDARY_COLORS_RGB["same_class"], font=font_small)

        draw_ov.rectangle([leg_x + 172, bnd_y + 2, leg_x + 180, bnd_y + 10], fill=(*BOUNDARY_COLORS_RGB["class_change"], 255))
        draw_ov.text((leg_x + 184, bnd_y), "Class Change", fill=BOUNDARY_COLORS_RGB["class_change"], font=font_small)

        # -------------------------------------------------------------
        # 3. BOUNDARY TRANSITION ALERT CARD (Floating Card)
        # -------------------------------------------------------------
        if frame_idx in boundary_alerts:
            alert = boundary_alerts[frame_idx]
            b_type = alert.get("type", "class_change")
            b_col = BOUNDARY_COLORS_RGB.get(b_type, (255, 255, 255))
            card_w = 660
            card_h = 58
            card_x = (target_width - card_w) // 2
            card_y = top_h + 15

            dist = abs(frame_idx - alert["frame"])
            pulse_alpha = int(245 - dist * 10)
            pulse_alpha = max(170, min(255, pulse_alpha))

            draw_ov.rounded_rectangle(
                [card_x, card_y, card_x + card_w, card_y + card_h],
                radius=10,
                fill=(15, 23, 42, pulse_alpha),
                outline=(*b_col, pulse_alpha),
                width=2,
            )
            draw_ov.rounded_rectangle([card_x + 4, card_y + 4, card_x + 12, card_y + card_h - 4], radius=3, fill=(*b_col, 255))

            left_inst = alert.get("left_instance", "?")
            right_inst = alert.get("right_instance", "?")
            if b_type == "same_class":
                title_txt = f"★ SAME-CLASS INSTANCE BOUNDARY (Frame {alert['frame']})"
                desc_txt = f"Instance #{left_inst} -> Instance #{right_inst} | Preserves continuous operations of identical class!"
                title_col = b_col
            else:
                title_txt = f"▶ CLASS-CHANGE BOUNDARY (Frame {alert['frame']})"
                desc_txt = f"Instance #{left_inst} -> Instance #{right_inst} | Transition between distinct action categories"
                title_col = b_col

            draw_ov.text((card_x + 22, card_y + 8), title_txt, fill=title_col, font=font_alert)
            draw_ov.text((card_x + 22, card_y + 32), desc_txt, fill=(241, 245, 249), font=font_regular)

        # -------------------------------------------------------------
        # 4. BOTTOM DUAL-TRACK TIMELINE PANEL
        # -------------------------------------------------------------
        draw_ov.rounded_rectangle(
            [tl_margin_x - 15, tl_panel_y, target_width - tl_margin_x + 15, target_height - 6],
            radius=12,
            fill=(10, 15, 30, 230),
            outline=(51, 65, 85, 220),
            width=1,
        )

        # Header of Timeline
        draw_ov.text((tl_margin_x, tl_panel_y + 6), "TEMPORAL TIMELINE COMPARISON", fill=(241, 245, 249), font=font_bold)
        draw_ov.text((tl_margin_x + 270, tl_panel_y + 7), "Track 1: Instance-Level GT (primary)  vs  Track 2: Frame-Wise Coarse GT (collapsed)", fill=(148, 163, 184), font=font_small)

        # Track 1 Label
        draw_ov.text((tl_margin_x, tl_y_track1 - 15), "Track 1: Instance GT (keeps same-class boundaries)", fill=(203, 213, 225), font=font_small)

        # Render all instance segments
        for inst in instances:
            f_start = inst["start_frame"]
            f_end = inst["end_frame"]
            x1 = tl_margin_x + int((f_start / total_frames) * tl_w)
            x2 = tl_margin_x + int((f_end / total_frames) * tl_w)
            x2 = max(x1 + 1, x2)
            cid = inst.get("class_id", -1)
            col = CLASS_COLORS_RGB.get(cid, CLASS_COLORS_RGB[-1])

            # Draw block
            draw_ov.rectangle([x1, tl_y_track1, x2, tl_y_track1 + tl_h_track], fill=(*col, 220), outline=(255, 255, 255, 100), width=1)

            # Draw instance text inside block if width permits
            block_w = x2 - x1
            if block_w > 40:
                tag = f"#{inst['id']}"
                if block_w > 80:
                    tag += f" {inst['class_name'][:6]}."
                draw_ov.text((x1 + 4, tl_y_track1 + 5), tag, fill=(255, 255, 255), font=font_small)

        # Render explicit boundary markers on Track 1
        for b in boundaries:
            b_f = b["frame"]
            bx = tl_margin_x + int((b_f / total_frames) * tl_w)
            b_type = b.get("type", "class_change")
            b_col = BOUNDARY_COLORS_RGB.get(b_type, (255, 255, 255))

            if b_type == "same_class":
                draw_ov.line([bx, tl_y_track1 - 4, bx, tl_y_track1 + tl_h_track + 4], fill=(*b_col, 255), width=3)
                draw_ov.polygon([(bx - 4, tl_y_track1 - 5), (bx + 4, tl_y_track1 - 5), (bx, tl_y_track1 - 1)], fill=(*b_col, 255))
            else:
                draw_ov.line([bx, tl_y_track1, bx, tl_y_track1 + tl_h_track], fill=(255, 255, 255, 200), width=1)

        # Track 2 Label
        draw_ov.text((tl_margin_x, tl_y_track2 - 15), "Track 2: Frame-Wise Coarse GT (collapsed: no same-class boundaries)", fill=(148, 163, 184), font=font_small)

        if gt_labels:
            coarse_segments = []
            curr_lbl = gt_labels[0]
            curr_start = 0
            for i, lbl in enumerate(gt_labels):
                if lbl != curr_lbl:
                    coarse_segments.append((curr_lbl, curr_start, i))
                    curr_lbl = lbl
                    curr_start = i
            coarse_segments.append((curr_lbl, curr_start, len(gt_labels)))

            for clbl, cstart, cend in coarse_segments:
                x1 = tl_margin_x + int((cstart / total_frames) * tl_w)
                x2 = tl_margin_x + int((cend / total_frames) * tl_w)
                x2 = max(x1 + 1, x2)
                cid = name_to_cid.get(clbl, -1)
                col = CLASS_COLORS_RGB.get(cid, CLASS_COLORS_RGB[-1])

                draw_ov.rectangle([x1, tl_y_track2, x2, tl_y_track2 + tl_h_track], fill=(*col, 160), outline=(200, 200, 200, 80), width=1)
                block_w = x2 - x1
                if block_w > 60:
                    draw_ov.text((x1 + 4, tl_y_track2 + 5), clbl[:12], fill=(241, 245, 249), font=font_small)

        # -------------------------------------------------------------
        # Playhead Cursor Across Both Tracks
        # -------------------------------------------------------------
        playhead_x = tl_margin_x + int(norm_pos * tl_w)
        # Vertical cursor line
        draw_ov.line(
            [playhead_x, tl_y_track1 - 6, playhead_x, tl_y_track2 + tl_h_track + 6],
            fill=(255, 255, 255, 255),
            width=2,
        )
        # Needle pointer top
        draw_ov.polygon(
            [(playhead_x - 5, tl_y_track1 - 10), (playhead_x + 5, tl_y_track1 - 10), (playhead_x, tl_y_track1 - 4)],
            fill=(255, 255, 255, 255),
        )
        # Needle pointer bottom
        draw_ov.polygon(
            [(playhead_x - 5, tl_y_track2 + tl_h_track + 10), (playhead_x + 5, tl_y_track2 + tl_h_track + 10), (playhead_x, tl_y_track2 + tl_h_track + 4)],
            fill=(255, 255, 255, 255),
        )

        # Composite overlay onto base frame
        img_out = Image.alpha_composite(img.convert("RGBA"), overlay)
        frame_final = cv2.cvtColor(np.array(img_out.convert("RGB")), cv2.COLOR_RGB2BGR)

        writer.write(frame_final)
        frame_idx += 1

        if frame_idx % 60 == 0 or frame_idx == total_frames:
            print(f"Rendered {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    cap.release()
    writer.release()
    print(f"Done rendering! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render a sample video from dataset_tas_instance with HUD & dual-track timeline.")
    parser.add_argument("--video-id", "-v", default="cd10_chuyen1", help="Sample video ID (default: cd10_chuyen1)")
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=Path,
        default=Path("/home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/dataset_tas_instance"),
        help="Path to dataset_tas_instance directory",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output video path (default: <dataset-dir>/visualizations/<video_id>_viz.mp4)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output video width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Output video height (default: 720)")
    parser.add_argument("--fps", type=float, default=None, help="FPS override")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = args.dataset_dir / "visualizations" / f"{args.video_id}_viz.mp4"

    render_visualization(
        dataset_dir=args.dataset_dir,
        video_id=args.video_id,
        output_path=out_path,
        target_width=args.width,
        target_height=args.height,
        fps_override=args.fps,
    )


if __name__ == "__main__":
    main()
