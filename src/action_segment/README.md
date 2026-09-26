# Action Segmentation & VLM Analysis Pipeline

Two-phase pipeline for industrial sewing-skill evaluation: learns the standard from an expert video, segments worker footage into physical actions via computer vision, classifies actions against the expert, and diagnoses where and why the worker is off-standard or slower than the expert.

---

## Architecture Overview

| Phase | Sub-step | Module | Input | Output | VLM |
|---|---|---|---|---|---|
| **1. Segmentation** | kinematic | `segmentation/kinematic.py` | `worker.mp4` | `action_segments.json` | ❌ (Zero API cost) |
| **2. Analysis** | expert | `analysis/expert_analysis.py` | `expert.json` + `expert.mp4` | `selected_frames.json`, `process_knowledge.json` | ✅ |
| | classify | `analysis/segment_classify.py` | `action_segments.json` + `selected_frames.json` + `worker.mp4` | `worker_segments.json` (+ cuts) | ✅ |
| | macro | `analysis/macro_eval.py` | classify output + `selected_frames.json` | `macro_eval.json` | ❌ (Pure code) |
| | micro | `analysis/micro_eval.py` | macro's slow segments | `micro_eval.json` | ✅ |

### Key Characteristics

- **Physical pre-segmentation without VLM**: Kinematic segmentation (SAM3 hand tracking + SEA-RAFT optical flow) identifies motion boundaries (speed valleys, directional changes) before calling any VLM.
- **ROI Masking**: Ignores other workers or bystanders in frame. An optional `<video_stem>.mask.png` (created via `tools/mask_editor`) restricts motion analysis to the active station.
- **Video-driven standards**: Automatically selects sharp reference frames from `expert.mp4` and prompts the VLM to synthesize standard operating procedure (SOP) rules.
- **Two-tier evaluation**:
  - *Macro*: Deterministic calculation of duration ratios, missing steps, and extra actions (zero API cost).
  - *Micro*: Targeted VLM analysis on slow or deviating segments to pinpoint root cause and suggest improvements.
- **Cost tracking**: Exact token usage and USD expenditure logged per segment and across the whole run.

---

## Directory Structure

```text
src/action_segment/
├── pipeline.py                 # CLI entry point: segment | analyze | all
├── requirements.txt            # Dependencies for Action Segmentation
│
├── segmentation/               # Phase 1: Kinematic action segmentation
│   └── kinematic.py            # KinematicSegmenter (invokes kinematic_pipeline/)
│
├── analysis/                   # Phase 2: VLM-based skill analysis
│   ├── expert_analysis.py      # expert sub-step: frame selection & SOP guideline
│   ├── segment_classify.py     # classify sub-step: match segments to expert scenes
│   ├── macro_eval.py           # macro sub-step: duration ratios & timing checks
│   └── micro_eval.py           # micro sub-step: slow-segment root-cause feedback
│
├── vlm_client.py               # OpenRouterClient (chat_json, retries, cost tracking)
├── manifest.py                 # Scene ordering, guideline formatting, duration math
│
├── config/                     # Sub-step knobs and file paths
│   ├── common.py               # Crop boxes, resize dimensions, default data paths
│   ├── phase1_segmentation.py  # Kinematic thresholds, angle tolerance, debug settings
│   ├── phase2_expert.py        # Expert model, frame sampling pool
│   ├── phase2_classify.py      # Classifier model, FPS downsampling
│   ├── phase2_macro.py         # Timing thresholds (slow/fast ratios)
│   └── phase2_micro.py         # Micro model, frame caps
│
├── utils/                      # Shared helper functions
│   ├── frames.py               # Frame encoding, cropping, Laplacian sharpness
│   ├── video.py                # OpenCV frame extraction, ffmpeg clipping
│   ├── message_content.py      # VLM multi-modal payload formatting
│   └── env.py                  # Environment file loader
│
├── prompts/                    # VLM prompt templates
│   ├── expert_analysis_prompts.py
│   ├── kinematic_classify_prompts.py
│   └── evaluation_prompts.py
│
└── kinematic_pipeline/         # Vendored SAM3 + SEA-RAFT engine
```

---

## Setup & Requirements

1. **Install dependencies**:
   ```bash
   pip install -r src/action_segment/requirements.txt
   ```
   *System requirements: `ffmpeg` installed on system PATH.*

2. **Configure API key**:
   ```bash
   cp .env.example .env
   # Add your OpenRouter API key in .env:
   # OPENROUTER_API_KEY=sk-or-v1-...
   ```

---

## Usage

### 1. Run Complete Pipeline (End-to-End)
Runs Phase 1 (kinematic segmentation) and Phase 2 (VLM analysis) sequentially in one process:
```bash
python -m src.action_segment.pipeline all
```

### 2. Run Individual Phases
```bash
# Phase 1: Physical action segmentation (zero VLM cost)
python -m src.action_segment.pipeline segment

# Phase 2: Expert SOP -> Classify -> Macro -> Micro
python -m src.action_segment.pipeline analyze
```

### 3. Run Specific Sub-steps (Cache-Aware)
```bash
python -m src.action_segment.pipeline segment --step kinematic
python -m src.action_segment.pipeline analyze --step expert
python -m src.action_segment.pipeline analyze --step classify
python -m src.action_segment.pipeline analyze --step macro
python -m src.action_segment.pipeline analyze --step micro
```

### 4. Batch Execution across Multiple Operations
```bash
# Run Phase 1 across all operations in data/ with visualization
python -m src.action_segment.pipeline segment --all-data --visualize

# Run for a specific operation (e.g., Công đoạn 1)
python -m src.action_segment.pipeline segment --cong-doan 1 --visualize

# Low VRAM mode (frame-by-frame SAM3 + downscaled resolution)
python -m src.action_segment.pipeline segment --all-data --resize-scale 0.25 --frame-step 2 --frame-by-frame
```

---

## CLI Flags Reference

| Flag | Description | Default |
|---|---|---|
| `--step` | Run only one sub-step of the phase | `None` (all sub-steps) |
| `--vlm-model` | Model for the expert SOP sub-step | `google/gemini-2.5-flash` |
| `--model` | Model for classify, macro, and micro sub-steps | `qwen/qwen3.7-plus` |
| `--cut` | Cut worker video into physical MP4 clips per segment | `False` |
| `--save-crop-frames` | Save cropped worker frames sent to the VLM | `False` |
| `--visualize` | Render boundary videos, motion plots, and timelines | `False` |
| `--mask` | Path to ROI mask PNG (overrides default naming) | `None` |
| `--force-segment` | Force re-running Phase 1 even if cached report exists | `False` |
| `--force-kinematic-expert` | Force re-running expert kinematic segmentation | `False` |
| `--action-segments` | Override input path for `action_segments.json` | `None` |

---

## Output Layout

Outputs are persisted under `data/`:
```text
data/
├── kinematic/
│   ├── action_segments.json        # Phase 1 raw physical boundaries
│   ├── pipe1_report.json           # Kinematic metrics report
│   └── debug/                      # --visualize: boundary plots & annotated video
├── expert_scenes/
│   ├── frames/                     # Auto-selected sharp expert frames
│   ├── selected_frames.json        # Scene guidelines & frame manifests
│   └── process_knowledge.json      # Full synthesized SOP
├── worker_segments/
│   ├── worker_segments.json        # Classified segments with off_standard flags
│   ├── timeline_debug.json         # Human-scannable timeline
│   └── cuts/                       # Individual video clips (when --cut is passed)
├── macro_eval.json                 # Missing/extra/slow/fast operational summary
└── micro_eval.json                 # Detailed root-cause feedback for slow actions
```

---

## Tuning Configuration

All parameters live in isolated files under `src/action_segment/config/`:
- `phase1_segmentation.py`: `ANGLE_TOLERANCE`, `MAGNITUDE_THRESHOLD`, `KERNEL_SIZE`.
- `phase2_macro.py`: `TIMING_SLOW_RATIO` (default `1.5`), `TIMING_FAST_RATIO` (default `0.7`).
- `phase2_expert.py`: `FRAMES_PER_ACTION_SEGMENT`, `SHARPNESS_POOL_FACTOR`.
- `common.py`: `EXPERT_CROP_BOX`, `WORKER_CROP_BOX`. Adjust these coordinates when working with new camera angles.
