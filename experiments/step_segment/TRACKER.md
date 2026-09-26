# Track: Step Segmentation & GEBD Models (TRACKER)

> **Single Source of Truth** for deep learning generic event boundary detection (GEBD) experiments: DDM-Net, DiffGEBD, and EfficientGEBD.

---

## 1. Track Overview & Architecture

- **Scope**: Detects generic event boundaries (operation transitions) using deep learning temporal models on industrial sewing footage.
- **Model Candidates**:
  1. **DDM-Net**: NVIDIA SOP Blueprint Architecture + PyTorch Lightning (ResNet-50 / DINOv2 dual-stream + Co-Transformer).
  2. **DiffGEBD**: Denoising Diffusion Generative Model with CFG (ResNet-50 + DiffFormer temporal diffusion head).
  3. **EfficientGEBD**: ACM MM 2024 Architecture Redesign (Video-domain CSN / ResNet-50 + FPN + DiffFormer / DiffMixer).
- **Core Documentation**:
  - [`docs/step_segment/ddm_net.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/ddm_net.md)
  - [`docs/step_segment/diff_gebd.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/diff_gebd.md)
  - [`docs/step_segment/efficient_gebd.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/efficient_gebd.md)
- **Standardized Evaluation Tool**: [`tools/eval_step_segment_predictions.py`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/tools/eval_step_segment_predictions.py)
- **Unified Experiment Runner**: [`experiments/templates/step_segment_runner.py`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/templates/step_segment_runner.py)

---

## 2. Standardized Output Protocol

Every experiment in this track must output artifacts to:
`outputs/step_segment/<iteration_id>/<experiment_id>/`

Required artifacts:
- `run.log`: Terminal execution logs.
- `predictions.json`: Mapping `{video_id: [t1, t2, ...]}` with predicted boundary timestamps in seconds.
- `metrics.json`: Standardized evaluation report containing:
  - `primary_metrics.macro_f1` (at $\pm 0.5$s window)
  - `primary_metrics.macro_recall`
  - `primary_metrics.macro_precision`
  - `primary_metrics.f1_at_0_25s`, `f1_at_0_5s`, `f1_at_1_0s`
  - `per_video_performance`
  - `error_cases`
- `checkpoints/` or `model_best.pth` / `best_model.ckpt`.

---

## 3. Metrics Leaderboard

| Rank | Model Architecture | Iteration | Experiment ID | Macro F1 (0.5s) | Recall (0.5s) | Precision (0.5s) | F1 @ 0.25s | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| - | EfficientGEBD (ResNet-50) | Pending | `exp_001` | - | - | - | - | Slicing & BCE fixes applied |
| - | DiffGEBD (Chunked 12s) | Pending | `exp_002` | - | - | - | - | Chunking dataset prepared |
| - | DDM-Net (ResNet-50) | Pending | `exp_003` | - | - | - | - | Aux loss re-weighting applied |

*Primary Target Metric*: Macro F1 $\ge 65.0\%$ at $\pm 0.5$s window tolerance.

---

## 4. Model Execution Standards

### A. DDM-Net (`src/step_segment/DDM-Net/`)
- **Training**:
  ```bash
  python src/step_segment/DDM-Net/train_sop_lightning.py \
    --config src/step_segment/DDM-Net/config/ddm_train_config.yaml \
    --output-dir outputs/step_segment/<iter>/<exp_id> \
    --backbone resnet50 --epochs 30 --learning-rate 0.0001
  ```
- **Inference**:
  ```bash
  python tools/infer_ddm_net.py \
    --checkpoint outputs/step_segment/<iter>/<exp_id>/best_model.ckpt \
    --output-dir outputs/step_segment/<iter>/<exp_id>
  ```

### B. DiffGEBD (`src/step_segment/DiffGEBD/`)
- **Training**:
  ```bash
  cd src/step_segment/DiffGEBD
  torchrun --nproc_per_node=1 train.py \
    --config-file config/sewing_diffgebd_resnet50_chunked.yaml \
    OUTPUT_DIR ../../../outputs/step_segment/<iter>/<exp_id>
  ```
- **Inference & Export**:
  ```bash
  python tools/infer_diffgebd.py \
    --config-file src/step_segment/DiffGEBD/config/sewing_diffgebd_resnet50_chunked.yaml \
    --weights outputs/step_segment/<iter>/<exp_id>/model_best.pth \
    --out-dir outputs/step_segment/<iter>/<exp_id>
  ```

### C. EfficientGEBD (`src/step_segment/EfficientGEBD/`)
- **Training**:
  ```bash
  cd src/step_segment/EfficientGEBD
  torchrun --nproc_per_node=1 train.py \
    --config-file config-files/sewing_resnet50.yaml \
    OUTPUT_DIR ../../../outputs/step_segment/<iter>/<exp_id>
  ```
- **Inference & Export**:
  ```bash
  python src/step_segment/EfficientGEBD/infer.py \
    --config-file src/step_segment/EfficientGEBD/config-files/sewing_resnet50.yaml \
    --checkpoint outputs/step_segment/<iter>/<exp_id>/model_best.pth \
    --input data/efficient_gebd_dataset/images/val \
    --gt data/efficient_gebd_dataset/val_annotation.pkl \
    --output-dir outputs/step_segment/<iter>/<exp_id>
  ```

---

## 5. Iteration History

| Iteration | Date | Focus / Goal | Experiments in Batch | Best Exp | Decision / Next Steps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `iter_01` (Planned) | TBD | Cross-model baseline comparison on Chuyền 1 | `exp_001_effgebd`, `exp_002_diffgebd`, `exp_003_ddm` | TBD | Measure baseline absolute-tolerance F1 across all 3 architectures. |

---

## 6. Hypotheses Log

| ID | Hypothesis | Proposed In | Tested In | Evidence / Verdict | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **H-GEBD-001** | Whole-video `np.linspace` causes severe recall drop in DiffGEBD; 10-15s chunking recovers recall. | `docs/step_segment/diff_gebd.md` | Pending | Tool `chunk_diff_gebd_dataset.py` ready. | ⏳ **Queued for iter_01** |
| **H-GEBD-002** | CSN video backbone outperforms 2D ResNet-50 by capturing 3D spatiotemporal hand flow. | `docs/step_segment/efficient_gebd.md` | Pending | Architecture implemented in `EfficientGEBD`. | ⏳ **Queued for iter_01** |
| **H-GEBD-003** | Weighted auxiliary heads (`main + 0.3 * aux`) breaks the 18-head loss plateau in DDM-Net. | `docs/step_segment/ddm_net.md` | Pending | Implemented in `train_sop_lightning.py`. | ⏳ **Queued for iter_01** |
