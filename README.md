# AI4Training AI Core

A unified AI system for industrial sewing skill evaluation and generic event boundary detection. It provides end-to-end capabilities to learn standard procedures from expert demonstrations, segment worker videos into physical operations, benchmark performance against standards, and diagnose technique or timing deviations.

---

## 1. Two Core Research & Engineering Tracks

The codebase is organized into two primary tracks:

### 🧵 Track 1: Action Segmentation & VLM Analysis ([`src/action_segment/`](src/action_segment/))
- **Paradigm**: Kinematic computer vision (SAM3 hand tracking + SEA-RAFT dense optical flow) + Multimodal Large Language Model (VLM).
- **Core Capabilities**:
  - Zero-VLM physical action pre-segmentation (`action_segments.json`).
  - Automatic expert reference frame selection (sharpness-weighted).
  - VLM-guided standard synthesis, worker segment classification, and off-standard technique detection.
  - Micro root-cause feedback on slow actions.
- 📖 **Detailed Guide**: [`src/action_segment/README.md`](src/action_segment/README.md)

### ⚡ Track 2: Step Segmentation (Deep Learning GEBD) ([`src/step_segment/`](src/step_segment/))
- **Paradigm**: Supervised deep learning Generic Event Boundary Detection (GEBD) across 3 architectures:
  - **DDM-Net**: Dual-stream spatial RGB + dense difference motion with Co-Transformer decoder.
  - **DiffGEBD**: Denoising diffusion generative model (DDIM) conditioned on visual similarity.
  - **EfficientGEBD**: Temporal sliding-window Feature Pyramid Network + DiffFormer/DiffMixer dissimilarity.
- **Master Comparison Tracker**: [`experiments/step_segment/step_segment_overview.md`](experiments/step_segment/step_segment_overview.md)
- 📖 **Detailed Guide**: [`src/step_segment/README.md`](src/step_segment/README.md)

---

## 2. Directory Layout

```text
.
├── src/
│   ├── action_segment/       # Track 1: Kinematic + VLM Action Analysis
│   │   ├── pipeline.py       # Main CLI entry point (segment | analyze | all)
│   │   ├── requirements.txt  # Action segmentation dependencies
│   │   └── README.md         # Full track documentation
│   └── step_segment/         # Track 2: Deep Learning GEBD Models
│       ├── DDM-Net/          # Dual-stream spatial + motion model
│       ├── DiffGEBD/         # Diffusion generative model
│       ├── EfficientGEBD/    # Temporal FPN + DiffFormer model
│       └── README.md         # Full track documentation
│
├── experiments/              # Structured AI Research Loop experiments
│   ├── action_segment/       # Experiments & TRACKER for action segmentation
│   ├── step_segment/         # Experiments & step_segment_overview.md
│   └── templates/            # Standardized runners and config schemas
│
├── outputs/                  # [Git-ignored] Checkpoints, logs, predictions, metrics
├── tools/                    # Video download, dataset prep, evaluation, and mask editor
├── docs/                     # Architectural specs & AI Research Loop protocol
└── data/                     # Videos, annotations, and generated datasets
```

---

## 3. Quick Start

### 3.1. Environment Setup

```bash
# Clone repository
git clone git@github.com:bmhungqb/ai4training-aicore.git
cd ai4training-aicore

# Track 1 (Action Segmentation):
pip install -r src/action_segment/requirements.txt
cp .env.example .env          # Set OPENROUTER_API_KEY

# Track 2 (Step Segmentation):
# See src/step_segment/README.md for per-model requirements
```

### 3.2. Downloading Datasets

Use the bundled video synchronization tool to fetch factory video datasets:
```bash
# Preview changes for Chuyền 2:
python -m tools.download_videos --chuyen 2 --dry-run

# Download all new/missing videos for Chuyền 2:
python -m tools.download_videos --chuyen 2
```

### 3.3. Running Pipelines

* **Action Segmentation Pipeline**:
  ```bash
  # Run both phases (kinematic segmentation + VLM analysis):
  python -m src.action_segment.pipeline all
  ```

* **Step Segmentation Benchmark**:
  ```bash
  # Run baseline training for all 3 models:
  bash experiments/step_segment/run_all_baselines.sh
  ```

---

## 4. Human-in-the-Loop AI Research Loop

All research experiments strictly follow the Human-in-the-loop lifecycle:
> **Evaluate** ➔ **Diagnose** ➔ **Research** ➔ **Debate** ➔ **Implement** ➔ **Human Executes**

- Detailed Protocol & Rules: [`docs/AI_RESEARCH_LOOP.md`](docs/AI_RESEARCH_LOOP.md)
- Agent Orchestration Skills: [`.agents/skills/`](.agents/skills/)
