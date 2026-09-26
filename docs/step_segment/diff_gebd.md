# DiffGEBD: Step Segmentation Tracker

> **Single Source of Truth** for method architecture, issue tracking, and experiment metrics for **DiffGEBD** on industrial sewing step segmentation.

---

## 1. Method Info

### 1.1. Overview & Architecture
- **Framework**: [DiffGEBD](https://github.com/JaejunHwang/DiffGEBD) (*Generic Event Boundary Detection via Denoising Diffusion*, ICCV 2025, [arXiv:2508.12084](https://arxiv.org/pdf/2508.12084)).
- **Core Mechanism**:
  - Treats 1D temporal boundary detection as a **generative denoising diffusion process** (DDPM/DDIM-style) rather than deterministic binary classification.
  - Boundary prediction is conditioned on ResNet-50 visual features and a temporal self-similarity matrix, processed through a `DiffFormer` diffusion head.
  - Employs **Classifier-Free Guidance (CFG)** during inference to balance boundary diversity against prediction fidelity.
- **Repository Implementation**: `src/step_segment/DiffGEBD/`
  - Visual Backbone: ResNet-50 (`modeling/resnet.py`).
  - Diffusion Core: `modeling/diffusion_model.py`, `modeling/diff_former.py`, `modeling/time_transformer.py`.
  - Dataset: `datasets/dataset.py` with `SEWING` domain support.
  - Runner: `train.py` via PyTorch `torchrun` DDP.
  - Configs: `config/sewing_diffgebd_resnet50.yaml`, `config/sewing_diffgebd_resnet50_chunked.yaml`.

### 1.2. Dataset & Directory Layout
```text
ai4training-aicore-poc/
├── data/
│   └── diff_gebd_dataset/
│       ├── images/{train,val}/<video_id>/frame%d.jpg
│       ├── train_annotation.pkl            # Full-video Kinetics-GEBD schema
│       ├── val_annotation.pkl
│       ├── train_annotation_chunked.pkl    # Sliced 10-15s chunk schema
│       └── val_annotation_chunked.pkl
├── tools/
│   ├── prepare_diff_gebd_dataset.py        # Frame extraction + annotation pickling
│   ├── chunk_diff_gebd_dataset.py          # Video slicing for sampling fix
│   ├── infer_diffgebd.py                   # Standalone inference runner
│   ├── export_diffgebd_predictions.py      # Exports pkl -> step_segments_pred.json
│   └── benchmark_step_segment_models.py    # Cross-model evaluation
└── src/step_segment/DiffGEBD/
    ├── config/
    └── train.py
```

### 1.3. Quick Setup & Execution
```bash
# Environment
conda create -n diffgebd python=3.10 -y && conda activate diffgebd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r src/step_segment/DiffGEBD/requirements.txt

# Data Preparation
python tools/process_step_segments.py
python tools/prepare_diff_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42
python tools/chunk_diff_gebd_dataset.py --chunk-seconds 12 --overlap-seconds 1.5

# Training (Chunked Config)
cd src/step_segment/DiffGEBD
torchrun --nproc_per_node=1 train.py \
  --config-file config/sewing_diffgebd_resnet50_chunked.yaml

# Inference & Benchmarking
python tools/export_diffgebd_predictions.py \
  --pred-pkl src/step_segment/DiffGEBD/output/.../model_pred_dict_ep-1.pkl \
  --split val --out-dir experiments/diffgebd_preds
```

### 1.4. Key Hyperparameters
- `DIFFUSION.CFG_SCALE`: Guidance scale (default 7.0; higher = sharper, less noisy boundaries).
- `DIFFUSION.SAMPLING_TIMESTEPS`: DDIM denoising steps at inference (default 16).
- `INPUT.SEQUENCE_LENGTH`: Sampled frames per chunk (set to 150 matching ~12s @ 15fps).
- `MODEL.POS_LOSS_WEIGHT`: Up-weights MSE loss on boundary frames (set to 5.0).
- `MODEL.SYNC_BN`: Must remain `false` for single-GPU training.

---

## 2. Status & Issues

### 2.1. Status Matrix

| Issue | Severity | Status | Solution Summary |
|---|---|---|---|
| **Sampling Imbalance: Precision 0.9+ vs Recall 0.19** | 🔴 CRITICAL | ✅ FIXED | Whole-video `np.linspace` skips 40 frames on long videos; resolved via 12s overlapping video chunking |
| **Inference Latency** | 🟡 MEDIUM | Monitored | 16 DDIM diffusion steps require 16 passes vs 1 pass in classifiers; balanced at `SAMPLING_TIMESTEPS=16` |
| **VRAM Limits in End-to-End Mode** | 🟡 MEDIUM | ✅ OK | 150 frames $\times$ 224px fits consumer GPU using `BATCH_SIZE=1-2` and `AMPE: True` |
| **Single-GPU SyncBatchNorm Error** | 🟢 MINOR | ✅ FIXED | Process group error when `nproc=1`; keep `MODEL.SYNC_BN: false` on single GPU |

---

### 2.2. Root Cause Analysis: Long-Video Sampling Imbalance
- **Root Cause**:
  In `datasets/dataset.py`:
  ```python
  if cfg.INPUT.END_TO_END:
      selected_indices = np.linspace(1, vlen, cfg.INPUT.SEQUENCE_LENGTH, dtype=int)
  ```
  DiffGEBD was engineered for Kinetics-GEBD, where all videos are pre-trimmed to **10 seconds** (~300 frames), making `np.linspace(1, 300, 150)` step by 2 frames.
  Sewing videos are multi-minute (up to 6,000 frames). Running `np.linspace(1, 6000, 150)` introduces a **40-frame stride**. Since a short sewing step lasts only 5–10 frames, sampling skips almost all ground-truth boundaries. The model observes almost zero positive frames and trivially learns to predict all zeros.
- **Solution (`tools/chunk_diff_gebd_dataset.py`)**:
  1. Chunks long videos into overlapping 10–15s clips (1.5s overlap) so boundaries are not sliced at borders.
  2. Re-maps boundary timestamps into each chunk's local reference frame, producing `train_annotation_chunked.pkl` and `val_annotation_chunked.pkl`.
  3. Aligns `INPUT.SEQUENCE_LENGTH: 150` with ~12s @ 15fps, restoring effective 1:1 sampling stride.

---

## 3. Experiments & Results

### 3.1. Baseline Metrics (Pre-Chunking vs Expected)

| State | Rec@0.05 | Prec@0.05 | F1@0.05 | Analysis |
|---|:---:|:---:|:---:|---|
| **Before Chunking** | **0.191** | **0.912** | **0.316** | Model misses ~81% of boundaries due to 40-frame sampling stride |
| **Target (Chunked)** | > 0.45 | > 0.60 | > 0.50 | 100% boundary retention in training input tensor |

### 3.2. Cross-Model Comparison Table

| Metric / Property | DDM-Net | EfficientGEBD | DiffGEBD |
|---|---|---|---|
| **Formulation** | Per-frame binary classifier | Multi-scale FPN + DiffFormer | Denoising diffusion generative model + CFG |
| **Backbone** | ResNet-50 / DINOv2 | CSN (R50/R152) / ResNet-50 | ResNet-50 |
| **Temporal Input** | Sliding window ($\pm 8$ frames) | 10s slice (100 frames) | 12s chunk (150 frames) |
| **Inference Cost** | 1 forward pass | 1 forward pass | 16 DDIM passes |
| **Key Tunables** | `frames_per_side`, `aux_loss_weight` | `POS_WEIGHT`, `SIGMA`, threshold | `CFG_SCALE`, `POS_LOSS_WEIGHT`, steps |
| **Primary Config** | `config/ddm_train_config.yaml` | `config-files/sewing_resnet50.yaml` | `config/sewing_diffgebd_resnet50_chunked.yaml` |
