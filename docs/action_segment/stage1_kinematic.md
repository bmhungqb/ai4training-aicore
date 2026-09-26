# Stage 1: Kinematic Action Segmentation Tracker

> **Single Source of Truth** for method architecture, issue tracking, and experiment metrics for **Stage 1 (Kinematic Action Pre-Segmentation)** in the AI4Training pipeline.

---

## 1. Method Info

### 1.1. Overview & Principles
- **Goal**: Automatically extract physical micro-action boundaries from raw worker videos with **zero VLM dependencies** and **zero API costs**.
- **Core Technology Stack**:
  - **SAM 3 (Segment Anything Model 3)**: Frame-by-frame tracking of worker hands and arms (`masks.npz`).
  - **SEA-RAFT**: High-precision dense optical flow estimation (`flow.npz`).
  - **Kinematic Fusion**: Fuses speed magnitude, directional velocity, and motion turbulence to detect physical transition valleys (pauses, direction reversals, repositioning).
- **Primary Outputs**:
  - `action_segments.json`: Micro-action segments with `start_time_s`, `end_time_s`, and `duration_s`.
  - `decomposed_motion.npz`: Per-frame velocity, directional turbulence, and transition likelihood.
  - `pipe1_report.json`: Quantitative segmentation report.
- **Repository Location**:
  - Orchestration: `pipeline.py segment`
  - Kinematic sub-pipeline: `src/action_segment/segmentation/` & `src/action_segment/kinematic_pipeline/`

### 1.2. Dataset & Directory Layout
```text
ai4training-aicore-poc/
├── data/
│   ├── {cd_id}/chuyen{chuyen_id}/
│   │   ├── *.mp4, *.mask.png
│   │   └── chuyen1_segment.json                # Human ground-truth annotations
│   └── {cd_id}/kinematic/{video_stem}/         # Stage 1 generated outputs
│       ├── action_segments.json
│       ├── decomposed_motion.npz
│       └── pipe1_report.json
├── tools/
│   ├── eval_boundary_recall.py                 # 9-CD boundary recall evaluation
│   └── export_eval_to_excel.py                 # Generates Excel audit spreadsheet
└── experiments/stage1_boundary_recall_9cd/
    ├── evaluation_result_9cd.xlsx              # 349-step audit sheet
    └── eval_report.json                        # JSON evaluation report
```

### 1.3. Quick Setup & Execution
```bash
# Dependencies
sudo apt update && sudo apt install -y ffmpeg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r requirements-kinematic.txt
huggingface-cli login                           # Required for SAM 3 access

# Run Stage 1 (Batch across all operations)
python pipeline.py segment --all-data --visualize

# Run single operation (e.g. CD 1)
python pipeline.py segment --cong-doan 1 --visualize

# Run on low VRAM GPUs
python pipeline.py segment --all-data --resize-scale 0.25 --frame-step 2 --frame-by-frame
```

---

## 2. Status & Issues

### 2.1. Core Algorithmic Fixes Implemented (Stage 1 v2)
Implemented in `src/action_segment/kinematic_pipeline/calculate_direction_magnitude.py`:

1. **Mask Border Erosion (`cv2.erode` $3 \times 3$ kernel)**:
   - Eliminates 1–3 pixel boundary bleeding from SAM3 hand masks onto the sewing table, suppressing false optical flow gradients and virtual turbulence noise.
2. **Resting-Wrist Energy Fusion (Translation + RMS Energy)**:
   - Replaced raw `np.median(hand_flow)` (which drops to zero when wrists rest on the table) with hybrid kinetic energy:
     $$\text{speed} = 0.5 \times v_{\text{median}} + 0.5 \times v_{\text{RMS}}$$
   - Retains finger push/fold motions and prevents false boundary cuts during active sewing passes.
3. **Temporal Linear Interpolation for Mask Dropouts**:
   - Replaces artificial zero-velocity drops during brief SAM3 tracking loss (5%–11% of frames during fast motions) with linear interpolation (`np.interp`), bridging transient tracking gaps.
4. **Sewing-Domain Hyperparameter Tuning**:
   - Lowered `min_distance` from 1.5s $\to$ **0.5s** to capture rapid micro-actions (reverse stitch 0.4s, corner pivot 0.5s).
   - Adaptive dynamic threshold: $\text{Threshold} = \text{local\_mean} + 0.7 \times \text{local\_std}$.
   - Balanced weights: `Speed=0.40, Direction=0.40, Turbulence=0.20`, angular threshold $25.0^\circ$.

---

### 2.2. Current Challenges Transitioning to Stage 2 (VLM Classification)

| Challenge | Root Cause | Impact |
|---|---|---|
| **Temporal Blindness** | Stage 2 samples 2–4 static frames per segment | Motion vectors lost; VLM cannot tell push vs pull vs pause $\to$ Hallucination |
| **Latency & API Costs** | Stage 1 outputs 60–134 segments/video | Sequential VLM calls take 3–5 min/video; vulnerable to API timeouts |
| **Granularity Mismatch** | Micro-actions (0.6s) vs Macro-steps (2–6s) | VLM output flickers; simple consecutive string merge fails to group steps |
| **Chat API Limitations** | Text-chat completion endpoint with base64 images | Not designed for continuous industrial motion sequence reasoning |

---

## 3. Experiments & Results

### 3.1. Benchmark: Operation 1 (CĐ 1) In-Depth Evaluation
Comparison on Operation 1 (`chuyen1_segment.json`, 24 Ground Truth steps) before vs after v2 improvements:

| Evaluation Metric | Baseline | Stage 1 v2 | Net Improvement |
|:---|:---:|:---:|:---:|
| **Boundary Recall (@0.5s)** | 44.0% | **88.0%** (22/25 boundaries) | **+44.0% (2x recall)** 🚀 |
| **Boundary Recall (@1.0s)** | 88.0% | **100.0%** (25/25 boundaries) | **Full boundary capture** |
| **Both-Ends Match (Start & End)** | 12.5% (3/24) | **79.2% (19/24 steps)** | **+66.7% coverage jump** |
| **At Least One End Match** | 66.7% | **100.0% (24/24 steps)** | **100% partial alignment** |
| **Boundary Precision (@0.5s)** | 48.0% | **45.9%** (28/61 cuts) | Stable (reflects 2.4x over-segmentation) |
| **F1-Score (@0.5s)** | 45.9% | **60.3%** | **+14.4%** |
| **Zero-Speed Frame Drops** | 21 frames | **0 frames** | 100% mask dropout elimination |

---

### 3.2. Benchmark: 9 Independent Operations (Chuyền 1 Full Line)
- **Scope**: 9 independent sewing operations (CĐ 1, 2, 3, 4, 5, 6, 8, 9, 10).
- **Scale**: 9 videos, **349 GT steps**, **363 GT boundaries**, **1,505 generated segments** (1,514 transition boundaries).

#### Per-Operation Breakdown ($\pm 0.5\text{s}$ tolerance):

| CĐ | Operation Name | GT Steps | GT Bounds | Machine Bounds | Over-seg | Boundary Recall | MAE (s) | Both-Ends Match | At Least One Match |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Flap hem topstitch x2 | 24 | 25 | 65 | 2.7x | **100.0%** | 0.135s | **24/24 (100.0%)** | 24/24 (100.0%) |
| **2** | Gusset lock + piping | 37 | 38 | 146 | 3.9x | **92.1%** | 0.203s | **31/37 (83.8%)** | 37/37 (100.0%) |
| **3** | Sleeve lining attach | 14 | 15 | 38 | 2.7x | **86.7%** | 0.231s | **10/14 (71.4%)** | 14/14 (100.0%) |
| **4** | Pocket patch stitch | 11 | 12 | 36 | 3.3x | 75.0% | 0.198s | 5/11 (45.5%) | 11/11 (100.0%) |
| **5** | Yoke template basting | 25 | 26 | 143 | 5.7x | **80.8%** | 0.210s | **17/25 (68.0%)** | 24/25 (96.0%) |
| **6** | Back yoke join | 24 | 25 | 97 | 4.0x | **80.0%** | 0.238s | **15/24 (62.5%)** | 24/24 (100.0%) |
| **8** | Sleeve lining set | 63 | 66 | 344 | 5.5x | **77.3%** | 0.233s | **38/63 (60.3%)** | 61/63 (96.8%) |
| **9** | Main collar set | 40 | 41 | 143 | 3.6x | **92.7%** | 0.192s | **35/40 (87.5%)** | 40/40 (100.0%) |
| **10** | Main sleeve set | 111 | 115 | 502 | 4.5x | **91.3%** | 0.238s | **92/111 (82.9%)** | 110/111 (99.1%) |
| **All** | **9 Operations** | **349** | **363** | **1,514** | **4.1x** | **Macro: 86.2%<br/>Micro: 87.3%** | **0.213s** | **267/349 (76.5%)** | **343/349 (98.3%)** |

#### Temporal Tolerance Sweep:

| Window | Macro Recall | Micro Recall | Boundaries Hit | Both Ends Match | At Least One Match | MAE (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\pm 0.25$s** | 50.5% | 52.1% | 189 / 363 | 28.1% (98/349) | 80.2% (280/349) | 0.126s |
| **$\pm 0.50$s** | **86.2%** | **87.3%** | **317 / 363** | **76.5% (267/349)** | **98.3% (343/349)** | **0.213s** |
| **$\pm 0.75$s** | **95.4%** | **95.6%** | **347 / 363** | **91.4% (319/349)** | **100.0% (349/349)** | **0.255s** |
| **$\pm 1.00$s** | **98.4%** | **98.6%** | **358 / 363** | **97.4% (340/349)** | **100.0% (349/349)** | **0.278s** |
| **$\pm 1.50$s** | **100.0%** | **100.0%** | **363 / 363** | **100.0% (349/349)** | **100.0% (349/349)** | **0.297s** |

### 3.3. Evaluation Artifacts & Reproduction
- Excel audit report: [`experiments/stage1_boundary_recall_9cd/evaluation_result_9cd.xlsx`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/stage1_boundary_recall_9cd/evaluation_result_9cd.xlsx)
- JSON metrics file: [`experiments/stage1_boundary_recall_9cd/eval_report.json`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/stage1_boundary_recall_9cd/eval_report.json)
- Reproduction commands:
  ```bash
  python -m tools.eval_boundary_recall --no-tune --exclude-cd 11 --out experiments/stage1_boundary_recall_9cd/eval_report.json
  python -m tools.export_eval_to_excel
  ```
