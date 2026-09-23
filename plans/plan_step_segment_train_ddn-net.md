# Plan: Step Segmentation Fine-Tuning with DDM-Net (NVIDIA SOP Blueprint Architecture)

## 1. Overview & Architectural Decisions
- **Selected Implementation**: NVIDIA SOP Monitoring Blueprints (`https://github.com/NVIDIA/sop-monitoring-blueprints/tree/main/microservices/sop-training-bp/microservices/ddm-training-ms/ddm`).
  - *Rationale*: Modern PyTorch Lightning 2.x codebase, clean YAML configuration system, direct video stream decoding via PyAV/TorchCodec (no cumbersome manual frame extraction to disk).
- **Event Boundary Definition**:
  - Consecutive segments with identical `operation_name` are merged together.
  - Empty (`""`) or `"UNKNOWN"` segments are retained as distinct segments (representing transition/idle intervals).
  - Merged segment boundaries define the ground-truth step transition points.
- **Project Structure**:
  - Data processing scripts: `tools/`
  - Model code & training engine: `src/step_segment/DDM-Net/`
  - Training guidance & documentation: `docs/step_segment_training_guidance.md`

---

## 2. Directory Layout
```text
ai4training-aicore-poc/
├── data/
│   ├── cd1/chuyen1/
│   │   ├── cam-03_....mp4
│   │   ├── action_segments_annotated.json
│   │   └── step_segments.json                  <-- Generated per-video
│   └── ...
├── tools/
│   ├── process_step_segments.py                <-- Merges consecutive actions -> step_segments.json
│   └── prepare_ddm_dataset.py                  <-- Formats dataset & generates train/val splits
├── src/
│   └── step_segment/
│       └── DDM-Net/                            <-- Vendor NVIDIA DDM-Net code
│           ├── config/
│           │   ├── ddm_train_config.yaml       <-- Customized for local/server dataset
│           │   └── config.py
│           ├── datasets/
│           │   ├── ddm_dataset.py
│           │   └── ddm_val_dataset.py
│           ├── modeling/
│           │   ├── resnetGEBD.py
│           │   └── transformer.py
│           ├── pl_ddm_datamodule.py
│           ├── train_sop_lightning.py          <-- Main training entry script
│           ├── requirements.txt
│           └── tools/
│               └── run_train.sh                <-- Server launch script
└── docs/
    └── step_segment_training_guidance.md       <-- Detailed server execution instructions
```

---

## 3. Implementation Steps

### Step 1: Data Processing Script (`tools/process_step_segments.py`)
- Traverse `data/` finding all `action_segments_annotated.json` files.
- For each file:
  - Merge consecutive segments that have the exact same `operation_name`.
  - Retain empty/UNKNOWN segments as valid transitional segments.
  - Compute merged start and end timestamps.
  - Save output to `step_segments.json` in the same directory alongside `action_segments_annotated.json`.

### Step 2: Dataset Preparation & Train/Val Split Script (`tools/prepare_ddm_dataset.py`)
- Parse generated `step_segments.json` across all 33 videos.
- Support two user-selectable splitting options:
  1. **Random split by video**: `--split-mode random --val-ratio 0.2 --seed 42`
  2. **Split by operation/CD folder**: `--split-mode by_folder --val-folders cd18,cd19,cd20` (or automatic ratio)
- Convert segments into NVIDIA DDM-Net annotation format:
  ```json
  {
    "video_unique_id": [
      {
        "event": "operation_name",
        "description": "operation_name",
        "start_timestamp": 0.0,
        "end_timestamp": 2.26
      },
      ...
    ]
  }
  ```
- Output generated annotation files and symlinked/referenced video directory structure under `data/ddm_dataset/`:
  - `train_annotation.json`
  - `val_annotation.json`
  - `videos/` (symlinks or relative mapping to avoid duplicating large MP4 files).

### Step 3: Populate Model Architecture (`src/step_segment/DDM-Net/`)
- Fetch and place the core DDM-Net PyTorch Lightning modules from NVIDIA's repository:
  - Dataset & DataLoader: `ddm_dataset.py`, `ddm_val_dataset.py`, `pl_ddm_datamodule.py`, `augmentation.py`.
  - Modeling: `resnetGEBD.py`, `transformer.py`, `co_transformer.py`, `attn.py`, `position_embedding.py`.
  - Training loop & utils: `train_sop_lightning.py`, `optim_factory.py`, `scheduler.py`, `metric.py`, `visualize.py`, `model_ema.py`.
- Create default configuration `ddm_train_config.yaml` tuned for sewing videos (input resolution 224, ResNet50 backbone, batch size, epochs, AdamW).

### Step 4: Server Training Guidance & Launch Scripts
- Create `src/step_segment/DDM-Net/tools/run_train.sh` supporting single-GPU or multi-GPU training via CLI arguments.
- Write `docs/step_segment_training_guidance.md`:
  - Step-by-step setup in a clean Python/Conda environment (Python 3.10+, PyTorch with CUDA, PyAV, Lightning).
  - Data processing commands (`python tools/process_step_segments.py`, `python tools/prepare_ddm_dataset.py`).
  - Launching training on server (background nohup/tmux, monitoring loss & F1 score, checkpoint saving).
  - Validation & inference scoring commands.
