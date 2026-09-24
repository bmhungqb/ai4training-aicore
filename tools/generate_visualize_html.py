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

        # Detect mask path
        mask_src = None
        mask_candidates = list(parent.glob("*.mask.png"))
        if not mask_candidates:
            mask_candidates = list(parent.glob("*mask*.png"))
        if mask_candidates:
            mask_file = mask_candidates[0]
            try:
                rel_mask = mask_file.relative_to(html_dir)
                mask_src = str(rel_mask)
            except ValueError:
                rel_mask_from_root = mask_file.resolve().relative_to(Path(".").resolve())
                mask_src = f"../{rel_mask_from_root}"

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
                "mask_src": mask_src,
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
      flex-wrap: wrap;
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
    .header-mask-group {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-left: auto;
      background: #0f172a;
      padding: 4px 12px;
      border-radius: 6px;
      border: 1px solid #334155;
    }
    .mask-checkbox-label {
      font-size: 13px;
      color: #cbd5e1;
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
      font-weight: 500;
    }
    .mask-checkbox-label input {
      accent-color: #38bdf8;
      cursor: pointer;
      width: 15px;
      height: 15px;
    }
    .mask-mode-select {
      font-size: 12px;
      padding: 3px 8px;
    }
    .mask-opacity-container {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: #94a3b8;
    }
    .mask-opacity-container input {
      width: 70px;
      cursor: pointer;
      accent-color: #38bdf8;
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
      overflow: hidden;
    }
    .video-wrapper {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      max-width: 100%;
      max-height: 70vh;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      background: #000;
    }
    video {
      display: block;
      max-width: 100%;
      max-height: 70vh;
      border-radius: 8px;
      background: #000;
    }
    #maskCanvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      border-radius: 8px;
      display: none;
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
      gap: 16px;
      flex-wrap: wrap;
    }
    .current-step-title {
      font-size: 16px;
      font-weight: 600;
      color: #60a5fa;
    }
    .current-step-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .ctrl-btn {
      background: #334155;
      color: #f1f5f9;
      border: 1px solid #475569;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s ease;
      user-select: none;
    }
    .ctrl-btn:hover:not(:disabled) {
      background: #475569;
      border-color: #64748b;
    }
    .ctrl-btn.btn-primary {
      background: #2563eb;
      border-color: #3b82f6;
      color: #ffffff;
      font-weight: 500;
    }
    .ctrl-btn.btn-primary:hover:not(:disabled) {
      background: #1d4ed8;
    }
    .ctrl-btn:disabled {
      opacity: 0.35;
      cursor: not-allowed;
    }
    .current-step-meta {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 6px;
    }
    .current-step-time {
      font-size: 13px;
      color: #94a3b8;
      font-variant-numeric: tabular-nums;
    }
    .auto-stop-label {
      font-size: 12px;
      color: #cbd5e1;
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    .auto-stop-label input {
      accent-color: #3b82f6;
      cursor: pointer;
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

    <div class="header-mask-group">
      <label class="mask-checkbox-label" title="Bật/Tắt hiển thị mặt nạ ROI (Worker / Machine Mask)">
        <input type="checkbox" id="maskToggle">
        <span>🎭 Hiện Mask ROI</span>
      </label>
      <select id="maskModeSelect" class="mask-mode-select" title="Chế độ hiển thị mặt nạ">
        <option value="dim">Che viền nền mờ</option>
        <option value="contour">Đường viền viền đỏ</option>
        <option value="tint">Phủ màu vùng ROI</option>
      </select>
      <div class="mask-opacity-container" title="Độ đậm nhạt của mặt nạ">
        <span>Độ mờ:</span>
        <input type="range" id="maskOpacity" min="10" max="100" value="70">
      </div>
    </div>
  </header>

  <div class="main-container">
    <div class="video-pane">
      <div class="video-wrapper" id="videoWrapper">
        <video id="player" controls></video>
        <canvas id="maskCanvas"></canvas>
      </div>
      <div class="current-step-banner">
        <div class="current-step-info">
          <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">Thao tác hiện tại</div>
          <div class="current-step-title" id="currentStepName">--</div>
        </div>
        <div class="current-step-controls">
          <button id="btnPrevStep" class="ctrl-btn" title="Bước trước">⏮ Bước trước</button>
          <button id="btnReplayStep" class="ctrl-btn btn-primary" title="Phát lại bước">🔄 Phát lại bước</button>
          <button id="btnNextStep" class="ctrl-btn" title="Bước sau">Bước sau ⏭</button>
        </div>
        <div class="current-step-meta">
          <div class="current-step-time" id="currentStepTime">--</div>
          <label class="auto-stop-label" title="Dừng video khi phát hết bước hiện tại">
            <input type="checkbox" id="autoStopToggle" checked> Dừng khi hết bước
          </label>
        </div>
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
    const btnPrevStep = document.getElementById("btnPrevStep");
    const btnReplayStep = document.getElementById("btnReplayStep");
    const btnNextStep = document.getElementById("btnNextStep");
    const autoStopToggle = document.getElementById("autoStopToggle");

    // Mask controls
    const maskToggle = document.getElementById("maskToggle");
    const maskModeSelect = document.getElementById("maskModeSelect");
    const maskOpacity = document.getElementById("maskOpacity");
    const maskCanvas = document.getElementById("maskCanvas");
    const maskCtx = maskCanvas.getContext("2d");

    let currentVideo = null;
    let currentSegments = [];
    let currentStepIdx = 0;
    let activeTargetEndTime = null;

    let maskImg = null;
    let maskLoaded = false;
    let maskCache = {}; // Cache rendered mask canvases per video mode

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

      currentStepIdx = 0;
      activeTargetEndTime = null;
      renderSegmentList();
      player.currentTime = 0;
      updateActiveStep(0);
      updateNavButtons();

      loadMask(currentVideo.mask_src);
    }

    function loadMask(src) {
      maskLoaded = false;
      maskImg = null;
      maskCanvas.style.display = "none";

      if (!src) {
        maskToggle.disabled = true;
        return;
      }

      maskToggle.disabled = false;
      const img = new Image();
      img.onload = () => {
        maskImg = img;
        maskLoaded = true;
        renderMask();
      };
      img.onerror = () => {
        maskLoaded = false;
        maskImg = null;
        renderMask();
      };
      img.src = src;
    }

    function renderMask() {
      if (!maskToggle.checked || !maskLoaded || !maskImg) {
        maskCanvas.style.display = "none";
        return;
      }

      const w = player.videoWidth || maskImg.naturalWidth || 1280;
      const h = player.videoHeight || maskImg.naturalHeight || 720;

      if (maskCanvas.width !== w || maskCanvas.height !== h) {
        maskCanvas.width = w;
        maskCanvas.height = h;
      }

      const mode = maskModeSelect.value;
      const opacity = parseFloat(maskOpacity.value) / 100.0;

      // Draw mask to an offscreen canvas to process pixels
      const offCanvas = document.createElement("canvas");
      offCanvas.width = w;
      offCanvas.height = h;
      const offCtx = offCanvas.getContext("2d");
      offCtx.drawImage(maskImg, 0, 0, w, h);

      const imgData = offCtx.getImageData(0, 0, w, h);
      const data = imgData.data;

      // Prepare output on maskCanvas
      const outData = maskCtx.createImageData(w, h);
      const out = outData.data;

      if (mode === "dim") {
        // Dim outside mask: mask pixel == 0 is dimmed black with alpha = opacity, mask pixel > 127 is clear (alpha 0)
        const alphaVal = Math.round(opacity * 255);
        for (let i = 0; i < data.length; i += 4) {
          const isFg = data[i] > 127;
          if (!isFg) {
            out[i] = 0;
            out[i + 1] = 0;
            out[i + 2] = 0;
            out[i + 3] = alphaVal;
          } else {
            out[i + 3] = 0;
          }
        }
      } else if (mode === "tint") {
        // Highlight ROI: mask pixel > 127 is tinted with cyan/blue, outside is clear
        const alphaVal = Math.round(opacity * 160);
        for (let i = 0; i < data.length; i += 4) {
          const isFg = data[i] > 127;
          if (isFg) {
            out[i] = 56;      // R
            out[i + 1] = 189;  // G (#38bdf8 sky blue)
            out[i + 2] = 248;  // B
            out[i + 3] = alphaVal;
          } else {
            out[i + 3] = 0;
          }
        }
      } else if (mode === "contour") {
        // Find boundary edges between 0 and 255
        const alphaVal = Math.round(opacity * 255);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * 4;
            const isFg = data[idx] > 127;
            let isEdge = false;
            if (isFg) {
              // Check 4 neighbors
              if (x === 0 || x === w - 1 || y === 0 || y === h - 1) {
                isEdge = true;
              } else {
                const left = data[idx - 4] > 127;
                const right = data[idx + 4] > 127;
                const top = data[idx - w * 4] > 127;
                const bottom = data[idx + w * 4] > 127;
                if (!left || !right || !top || !bottom) {
                  isEdge = true;
                }
              }
            }
            if (isEdge) {
              out[idx] = 239;     // #ef4444 Red
              out[idx + 1] = 68;
              out[idx + 2] = 68;
              out[idx + 3] = alphaVal;
            } else {
              out[idx + 3] = 0;
            }
          }
        }
      }

      maskCtx.putImageData(outData, 0, 0);
      maskCanvas.style.display = "block";
    }

    maskToggle.addEventListener("change", renderMask);
    maskModeSelect.addEventListener("change", renderMask);
    maskOpacity.addEventListener("input", renderMask);

    player.addEventListener("loadedmetadata", () => {
      if (maskLoaded) renderMask();
    });

    function playStep(idx) {
      if (!currentSegments || idx < 0 || idx >= currentSegments.length) return;
      currentStepIdx = idx;
      const seg = currentSegments[idx];
      player.currentTime = seg.start_time_s;
      if (autoStopToggle.checked) {
        activeTargetEndTime = seg.end_time_s;
      } else {
        activeTargetEndTime = null;
      }
      player.play();
      updateActiveStep(seg.start_time_s);
      updateNavButtons();
    }

    function updateNavButtons() {
      const hasSegments = currentSegments && currentSegments.length > 0;
      btnPrevStep.disabled = !hasSegments || currentStepIdx <= 0;
      btnNextStep.disabled = !hasSegments || currentStepIdx >= currentSegments.length - 1;
      btnReplayStep.disabled = !hasSegments || currentStepIdx < 0 || currentStepIdx >= currentSegments.length;
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
          playStep(idx);
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
        currentStepIdx = activeIdx;
        const activeSeg = currentSegments[activeIdx];
        currentStepName.textContent = activeSeg.operation_name;
        currentStepTime.textContent = `${activeSeg.start_time_s.toFixed(2)}s - ${activeSeg.end_time_s.toFixed(2)}s`;

        const activeEl = document.getElementById(`seg-${activeIdx}`);
        if (activeEl) {
          activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      } else {
        if (currentStepIdx >= 0 && currentStepIdx < currentSegments.length) {
          const seg = currentSegments[currentStepIdx];
          if (Math.abs(time - seg.end_time_s) < 0.1) {
            const el = document.getElementById(`seg-${currentStepIdx}`);
            if (el) el.classList.add("active");
            currentStepName.textContent = seg.operation_name;
            currentStepTime.textContent = `${seg.start_time_s.toFixed(2)}s - ${seg.end_time_s.toFixed(2)}s`;
          } else {
            currentStepName.textContent = "--";
            currentStepTime.textContent = "--";
          }
        } else {
          currentStepName.textContent = "--";
          currentStepTime.textContent = "--";
        }
      }
      updateNavButtons();
    }

    btnPrevStep.addEventListener("click", () => {
      if (currentStepIdx > 0) {
        playStep(currentStepIdx - 1);
      }
    });

    btnReplayStep.addEventListener("click", () => {
      if (currentStepIdx >= 0 && currentStepIdx < currentSegments.length) {
        playStep(currentStepIdx);
      }
    });

    btnNextStep.addEventListener("click", () => {
      if (currentStepIdx < currentSegments.length - 1) {
        playStep(currentStepIdx + 1);
      }
    });

    autoStopToggle.addEventListener("change", () => {
      if (!autoStopToggle.checked) {
        activeTargetEndTime = null;
      } else if (currentStepIdx >= 0 && currentStepIdx < currentSegments.length && !player.paused) {
        activeTargetEndTime = currentSegments[currentStepIdx].end_time_s;
      }
    });

    player.addEventListener("seeked", () => {
      if (activeTargetEndTime !== null && currentStepIdx >= 0 && currentStepIdx < currentSegments.length) {
        const seg = currentSegments[currentStepIdx];
        if (player.currentTime < seg.start_time_s || player.currentTime >= seg.end_time_s) {
          activeTargetEndTime = null;
        }
      }
    });

    videoSelect.addEventListener("change", (e) => loadVideo(parseInt(e.target.value)));

    player.addEventListener("timeupdate", () => {
      const time = player.currentTime;
      updateActiveStep(time);

      if (autoStopToggle.checked && activeTargetEndTime !== null) {
        if (time >= activeTargetEndTime) {
          player.pause();
          player.currentTime = activeTargetEndTime;
          activeTargetEndTime = null;
        }
      }
    });

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
