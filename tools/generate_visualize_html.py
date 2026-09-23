#!/usr/bin/env python3
"""Generate a standalone HTML viewer to inspect videos alongside merged step segments.

Scans `data/**/step_segments.json` and corresponding `.mp4` videos, and outputs
an interactive HTML file under `visualize_html/index.html`.

Usage:
    python tools/generate_visualize_html.py
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def is_unknown(name: str | None) -> bool:
    return not name or not name.strip() or name.strip().upper() in ["UNKNOWN", "NONE"]


def collect_data(data_dir: Path, html_dir: Path) -> list[dict]:
    items = []
    for step_file in sorted(data_dir.rglob("step_segments.json")):
        parent = step_file.parent
        mp4s = list(parent.glob("*.mp4"))
        if not mp4s:
            continue
        mp4 = mp4s[0]

        # Calculate relative path from html_dir to mp4
        try:
            rel_mp4 = mp4.relative_to(html_dir)
            video_src = str(rel_mp4)
        except ValueError:
            # Fallback to computing relative path via repo root
            rel_from_root = mp4.resolve().relative_to(Path(".").resolve())
            video_src = f"../{rel_from_root}"

        data = json.loads(step_file.read_text())
        segs = data.get("segments", [])
        cd = parent.parent.name
        chuyen = parent.name
        vid_id = f"{cd}_{chuyen}"

        # Mark trimmed edge unknown flags
        start_idx = 0
        while start_idx < len(segs) and is_unknown(segs[start_idx].get("operation_name", "")):
            start_idx += 1
        end_idx = len(segs)
        while end_idx > start_idx and is_unknown(segs[end_idx - 1].get("operation_name", "")):
            end_idx -= 1

        annotated_segs = []
        for i, s in enumerate(segs):
            is_edge_unk = (i < start_idx) or (i >= end_idx)
            annotated_segs.append(
                {
                    "operation_name": s.get("operation_name", "") or "UNKNOWN",
                    "start_time_s": s.get("start_time_s", 0.0),
                    "end_time_s": s.get("end_time_s", 0.0),
                    "n_merged": s.get("n_merged", 1),
                    "is_edge_unknown": is_edge_unk,
                }
            )

        items.append(
            {
                "id": vid_id,
                "label": f"{cd.upper()} - {chuyen}",
                "cd": cd,
                "chuyen": chuyen,
                "video_src": video_src,
                "fps": round(data.get("fps", 0), 2),
                "n_segments": len(annotated_segs),
                "segments": annotated_segs,
            }
        )
    return items


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Step Segment Visualizer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f1f5f9;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: #1e293b;
      padding: 12px 20px;
      display: flex;
      align-items: center;
      gap: 16px;
      border-bottom: 1px solid #334155;
    }
    header h1 { font-size: 18px; font-weight: 600; color: #38bdf8; }
    select {
      background: #0f172a;
      color: #f1f5f9;
      border: 1px solid #475569;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 14px;
      outline: none;
      cursor: pointer;
    }
    .badge {
      background: #334155;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      color: #94a3b8;
    }
    .main-container {
      display: flex;
      flex: 1;
      overflow: hidden;
    }
    .video-pane {
      flex: 1.3;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      align-items: center;
      justify-content: center;
      background: #090d16;
    }
    video {
      max-width: 100%;
      max-height: 70vh;
      border-radius: 8px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      background: #000;
    }
    .current-step-banner {
      width: 100%;
      background: #1e293b;
      border: 1px solid #3b82f6;
      border-radius: 8px;
      padding: 12px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .current-step-title {
      font-size: 16px;
      font-weight: 600;
      color: #60a5fa;
    }
    .current-step-time {
      font-size: 13px;
      color: #94a3b8;
      font-variant-numeric: tabular-nums;
    }
    .timeline-pane {
      flex: 1;
      border-left: 1px solid #334155;
      background: #111827;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .timeline-header {
      padding: 12px 16px;
      background: #1e293b;
      border-bottom: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .timeline-header h2 { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
    .segment-list {
      flex: 1;
      overflow-y: auto;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .segment-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      background: #1e293b;
      border-radius: 6px;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s ease;
    }
    .segment-item:hover {
      background: #273549;
      border-color: #475569;
    }
    .segment-item.active {
      background: #1e3a8a;
      border-color: #3b82f6;
    }
    .segment-item.edge-unk {
      opacity: 0.55;
      border: 1px dashed #64748b;
    }
    .segment-idx {
      font-size: 12px;
      font-weight: 700;
      color: #94a3b8;
      width: 28px;
    }
    .segment-name {
      flex: 1;
      font-size: 14px;
      font-weight: 500;
      color: #f8fafc;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .segment-badge {
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      background: #334155;
      color: #cbd5e1;
    }
    .badge-trimmed {
      background: #7f1d1d;
      color: #fca5a5;
    }
    .segment-time {
      font-size: 12px;
      color: #94a3b8;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
  </style>
</head>
<body>
  <header>
    <h1>Step Segment Visualizer</h1>
    <label for="videoSelect" style="font-size:13px; color:#94a3b8;">Chọn Video:</label>
    <select id="videoSelect"></select>
    <span class="badge" id="videoInfoBadge">0 segments</span>
  </header>

  <div class="main-container">
    <div class="video-pane">
      <video id="player" controls></video>
      <div class="current-step-banner">
        <div>
          <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">Thao tác hiện tại</div>
          <div class="current-step-title" id="currentStepName">--</div>
        </div>
        <div class="current-step-time" id="currentStepTime">--</div>
      </div>
    </div>

    <div class="timeline-pane">
      <div class="timeline-header">
        <h2>Merged Step Segments</h2>
        <span style="font-size: 12px; color: #64748b;">Click bước để seek video</span>
      </div>
      <div class="segment-list" id="segmentList"></div>
    </div>
  </div>

  <script>
    const dataset = %DATASET_JSON%;
    const videoSelect = document.getElementById("videoSelect");
    const player = document.getElementById("player");
    const segmentList = document.getElementById("segmentList");
    const videoInfoBadge = document.getElementById("videoInfoBadge");
    const currentStepName = document.getElementById("currentStepName");
    const currentStepTime = document.getElementById("currentStepTime");

    let currentVideo = null;
    let currentSegments = [];

    // Populate dropdown
    dataset.forEach((v, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${v.label} (${v.n_segments} steps)`;
      videoSelect.appendChild(opt);
    });

    function loadVideo(index) {
      currentVideo = dataset[index];
      currentSegments = currentVideo.segments;
      player.src = currentVideo.video_src;
      videoInfoBadge.textContent = `${currentSegments.length} step segments | FPS: ${currentVideo.fps}`;

      renderSegmentList();
      player.currentTime = 0;
      updateActiveStep(0);
    }

    function renderSegmentList() {
      segmentList.innerHTML = "";
      currentSegments.forEach((seg, idx) => {
        const item = document.createElement("div");
        item.className = "segment-item" + (seg.is_edge_unknown ? " edge-unk" : "");
        item.id = `seg-${idx}`;

        const dur = (seg.end_time_s - seg.start_time_s).toFixed(2);
        let tagHtml = `<span class="segment-badge">${seg.n_merged} raw clips</span>`;
        if (seg.is_edge_unknown) {
          tagHtml = `<span class="segment-badge badge-trimmed">Trimmed in DDM</span>` + tagHtml;
        }

        item.innerHTML = `
          <span class="segment-idx">#${idx + 1}</span>
          <span class="segment-name" title="${seg.operation_name}">${seg.operation_name}</span>
          ${tagHtml}
          <span class="segment-time">${seg.start_time_s.toFixed(2)}s - ${seg.end_time_s.toFixed(2)}s (${dur}s)</span>
        `;

        item.addEventListener("click", () => {
          player.currentTime = seg.start_time_s;
          player.play();
        });

        segmentList.appendChild(item);
      });
    }

    function updateActiveStep(time) {
      let activeIdx = -1;
      for (let i = 0; i < currentSegments.length; i++) {
        const s = currentSegments[i];
        if (time >= s.start_time_s && time <= s.end_time_s) {
          activeIdx = i;
          break;
        }
      }

      currentSegments.forEach((_, i) => {
        const el = document.getElementById(`seg-${i}`);
        if (el) el.classList.toggle("active", i === activeIdx);
      });

      if (activeIdx !== -1) {
        const activeSeg = currentSegments[activeIdx];
        currentStepName.textContent = activeSeg.operation_name;
        currentStepTime.textContent = `${activeSeg.start_time_s.toFixed(2)}s - ${activeSeg.end_time_s.toFixed(2)}s`;

        const activeEl = document.getElementById(`seg-${activeIdx}`);
        if (activeEl) {
          activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      } else {
        currentStepName.textContent = "--";
        currentStepTime.textContent = "--";
      }
    }

    videoSelect.addEventListener("change", (e) => loadVideo(parseInt(e.target.value)));
    player.addEventListener("timeupdate", () => updateActiveStep(player.currentTime));

    if (dataset.length > 0) {
      loadVideo(0);
    }
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("visualize_html"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = collect_data(args.data_dir, args.out_dir)

    dataset_json = json.dumps(items, ensure_ascii=False)
    html_content = HTML_TEMPLATE.replace("%DATASET_JSON%", dataset_json)

    out_file = args.out_dir / "index.html"
    out_file.write_text(html_content, encoding="utf-8")
    print(f"Generated visualizer with {len(items)} videos -> {out_file}")


if __name__ == "__main__":
    main()
