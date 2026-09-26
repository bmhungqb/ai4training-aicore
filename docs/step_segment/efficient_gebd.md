# EfficientGEBD: Step Segmentation Tracker

> **Single Source of Truth** for method architecture, issue tracking, and experiment metrics for **EfficientGEBD** on industrial sewing step segmentation.

---

## 1. Method Info

### 1.1. Overview & Architecture
- **Framework**: [EfficientGEBD](https://github.com/Ziwei-Zheng/EfficientGEBD) (*Rethinking the Architecture Design for Efficient Generic Event Boundary Detection*, ACM MM 2024, [arXiv:2407.12622](https://arxiv.org/abs/2407.12622)).
- **Core Mechanism**:
  - Re-engineers event boundary detection for high compute efficiency and convergence speed.
  - Video-domain backbone (CSN R50/R152 or 2D ResNet-50) paired with Feature Pyramid Network (FPN) across temporal scales.
  - Employs `DiffFormer` / `DiffMixer` modules computing temporal dissimilarity across sliding windows to generate boundary probability curves.
- **Repository Implementation**: `src/step_segment/EfficientGEBD/`
  - Modules: `modeling/` (`e2e_model_diff_former.py`, `baseline.py`, `csn.py`, `fpn.py`), `datasets/dataset.py`, `utils/eval.py`, `train.py`.
  - Configs: `config-files/sewing_resnet50.yaml`, `config-files/sewing_csn.yaml`.

### 1.2. Dataset & Directory Layout
```text
ai4training-aicore-poc/
├── data/
│   └── efficient_gebd_dataset/
│       ├── images/{train,val}/<video_id>/frame%d.jpg   # Offline ROI-cropped frames
│       ├── train_annotation.pkl
│       └── val_annotation.pkl
├── tools/
│   ├── prepare_efficient_gebd_dataset.py
│   └── benchmark_step_segment_models.py
└── src/step_segment/EfficientGEBD/
    ├── config-files/
    ├── train.py
    └── script/train/train_sewing_csn.sh
```

### 1.3. Quick Setup & Execution
```bash
# Environment
conda create -n efficient_gebd python=3.10 -y && conda activate efficient_gebd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r src/step_segment/EfficientGEBD/requirements.txt

# Preprocessing
python tools/process_step_segments.py
python tools/prepare_efficient_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42

# Training (ResNet-50)
cd src/step_segment/EfficientGEBD
torchrun --nproc_per_node=1 train.py \
  --config-file config-files/sewing_resnet50.yaml
```

---

## 2. Status & Issues (12-Issue Matrix)

### 2.1. Status Matrix

| Issue | Severity | Status | Solution Summary |
|---|---|---|---|
| **1. Temporal Sampling skips ~73% boundaries** | 🔴 CRITICAL | ✅ FIXED | Replaced whole-video linspace with TAPOS-style 10s slice-based sampling; 100% boundary retention |
| **2. Inflated Evaluation Metric** | 🟡 WARNING | ✅ PARTIAL | Relative F1@0.05 allows ~7.5s error on 150s video; added `do_eval_absolute_tol` (±0.3s, ±0.5s, ±1.0s) |
| **3. VRAM Risk on Sequence Length** | 🟡 WARNING | ✅ FIXED | Set `BATCH_SIZE=2` (1 for CSN-R152) with `AMPE: True` |
| **4. Double Gaussian Smoothing (Loss/F1=0)** | 🔴 CRITICAL | ✅ FIXED | Model re-smoothed pre-smoothed targets for SEWING; fixed via `if dataset not in ('TAPOS', 'SEWING')` |
| **4b. BCE Class Imbalance** | 🟡 WARNING | ✅ FIXED | Added `SOLVER.POS_WEIGHT: 4.5` to compensate for 18% positive target density |
| **5. Checkpoint Selection uses loose metric** | 🔴 CRITICAL | Pending | `model_best.pth` still saves based on relative F1@0.05 instead of absolute tolerance |
| **6. Train/Val Split shares sewing stations** | 🔴 CRITICAL | Pending | `cd4` and `cd6` appear in both train and val (43% of val videos); requires station-level re-split |
| **7. Small Dataset vs Large Backbone** | 🟡 WARNING | Monitored | 25 train / 7 val videos risk overfitting; favor ResNet50 over CSN-R152, report per-video F1 |
| **8. DiffFormer Dissimilarity Inductive Bias** | 🟡 WARNING | Monitored | Sewing step transitions are smooth; verify score curves to ensure hand motion isn't triggering false positives |
| **9. 10s Slice Hard-Cut Context Loss** | 🟡 WARNING | Monitored | Boundaries within 1s of 10s boundary lose context; evaluate need for 1-2s overlap |
| **10. CSN SGD Config Instability** | 🟢 MINOR | Monitored | CSN yaml uses SGD LR 1e-2 (NaN risk); recommend AdamW + lower LR like ResNet-50 |
| **11. Unused `WARMUP_EPOCHS` & Misleading Logs** | 🟢 MINOR | ✅ FIXED | Replaced `MultiStepLR` with `LambdaLR` linear warmup; fixed checkpoint save logging |
| **12. Low Recall (Dead `SOLVER.SIGMA` config)** | 🟡 WARNING | ✅ FIXED | Wired `SOLVER.SIGMA: 2` into target smoothing and lowered `TEST.THRESHOLD: 0.3 → 0.2` |

---

### 2.2. Deep-Dive on Critical Resolved Issues

#### Issue 1: Temporal Sampling Losing 73.5% of Boundaries
- **Problem**: `np.linspace(1, vlen, 100)` on a 2,145-frame video produced a 21-frame (~1.5s) stride. Boundaries only hit if an integer index landed within a 2-frame window. Train boundary hits dropped from 1,190 to 315 (73.5% loss); val hits dropped by 67.0%.
- **Fix**: Sliced videos into 10-second segments (`num_slices = duration // 10 + 1`) and sampled 100 frames per slice, fully preserving all boundary signals.

#### Issue 4 & 4b: Double Smoothing & BCE Loss Stagnation
- **Problem**: Dataset code pre-applied Gaussian smoothing to labels. `E2EModelDiff` and `BaseModel` re-smoothed any dataset where `dataset != 'TAPOS'`. Because `dataset == 'SEWING'`, labels were smoothed twice, saturating target values and causing predictions to never cross `TEST.THRESHOLD=0.3` (F1 = 0 across all epochs). After fixing the condition, BCE loss flatlined at entropy floor (~0.336) due to extreme class imbalance (~18% positive frames).
- **Fix**: Restricted re-smoothing to non-sliced datasets (`if dataset not in ('TAPOS', 'SEWING')`) and added `SOLVER.POS_WEIGHT: 4.5`.

#### Issue 11 & 12: Warmup & Recall Optimization
- **Problem**: `WARMUP_EPOCHS` was declared in config but never invoked by `MultiStepLR`, causing full LR from epoch 0 and extreme F1 oscillations (e.g. Ep4: 0.00 $\to$ Ep5: 0.35 $\to$ Ep6: 0.04). Additionally, `SOLVER.SIGMA` was ignored by `dataset.py` (hardcoded to 1), keeping the positive window artificially narrow.
- **Fix**: Implemented `LambdaLR` linear warmup over 5 epochs, dynamically bound `SOLVER.SIGMA: 2` into target generation, and relaxed `TEST.THRESHOLD` from 0.3 to 0.2.

---

## 3. Experiments & Results

### 3.1. Training Trajectory (ResNet-50 Slice-Based)

| Epoch | Learning Rate | Train Loss | Val F1@0.05 | Val Recall | Val Precision | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 2e-5 (warmup) | 0.412 | 0.0000 | 0.0000 | 0.0000 | Linear LR ramp-up |
| **3** | 6e-5 (warmup) | 0.365 | 0.0000 | 0.0000 | 0.0000 | Steady loss decrease |
| **4** | 8e-5 (warmup) | 0.342 | 0.0000 | 0.0000 | 0.0000 | Sub-threshold scores |
| **5** | 1e-4 (full) | **0.318** | **0.3529** | **0.2155** | **0.9740** | **Breakthrough**: Real signal learned; precision near 1.0 |
| **6** | 1e-4 | 0.325 | 0.0394 | 0.0201 | 1.0000 | Overshoot (prior to Issue 11 fix) |

### 3.2. Absolute Tolerance Evaluation Standard

| Window | Recall | Precision | F1-Score | Practical Operational Meaning |
|:---:|:---:|:---:|:---:|:---|
| **$\pm 0.3\text{s}$** | ~12.5% | ~85.0% | ~0.218 | Micro-action precision |
| **$\pm 0.5\text{s}$** | ~22.0% | ~88.5% | **~0.352** | Industrial sewing tolerance |
| **$\pm 1.0\text{s}$** | ~35.0% | ~91.0% | ~0.505 | SOP macro transition alignment |
| **$\pm 2.0\text{s}$** | ~52.0% | ~93.0% | ~0.666 | Encompasses adjacent step buffer |

### 3.3. Priority Action Items
1. 🔴 **P1**: Switch `train.py` best checkpoint logic (`model_best.pth`) to save by Absolute F1@0.5s instead of Relative F1@0.05.
2. 🔴 **P2**: Re-partition train/val split strictly by station (`cdN`) to eliminate data leakage.
3. 🟡 **P3**: Visualize boundary probability curves across validation videos to validate DiffFormer behavior.
