# Track: Step Segmentation & GEBD Models (TRACKER)

> **Single Source of Truth** for deep learning generic event boundary detection (GEBD) experiments: DDM-Net, DiffGEBD, and EfficientGEBD.

---

## 1. Track Overview & Architecture

- **Scope**: Detects generic event boundaries (operation transitions) using deep learning temporal models.
- **Model Candidates**:
  1. **DDM-Net**: NVIDIA SOP Blueprint Architecture + PyTorch Lightning (ResNet-50 / DINOv2).
  2. **DiffGEBD**: Denoising Diffusion Generative Model with CFG (ResNet-50 + DiffFormer).
  3. **EfficientGEBD**: ACM MM 2024 Architecture Redesign (Video-domain CSN / ResNet-50 + FPN + DiffFormer).
- **Core Documentation**:
  - [`docs/step_segment/ddm_net.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/ddm_net.md)
  - [`docs/step_segment/diff_gebd.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/diff_gebd.md)
  - [`docs/step_segment/efficient_gebd.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/efficient_gebd.md)

---

## 2. Metrics Leaderboard

| Rank | Model Architecture | Iteration | Experiment ID | Precision | Recall | F1 Score | Validation Loss | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| - | EfficientGEBD | Pending | - | - | - | - | - | Ready for Benchmark |
| - | DiffGEBD (Chunked) | Pending | - | - | - | - | - | Data Prepared |
| - | DDM-Net | Pending | - | - | - | - | - | Investigating flat loss |

---

## 3. Iteration History

| Iteration | Date | Focus / Goal | Experiments in Batch | Best Exp | Decision / Next Steps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `iter_01` (Upcoming) | TBD | Cross-model GEBD Benchmark on Chuyền 1 | `exp_001_diffgebd`, `exp_002_efficientgebd` | TBD | Compare F1 at tolerance windows [0.25s, 0.5s, 1.0s]. |

---

## 4. Hypotheses Log

| ID | Hypothesis | Proposed In | Tested In | Evidence / Verdict | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **H-GEBD-001** | Whole-video `np.linspace` causes severe recall drop in DiffGEBD; 10-15s chunking recovers recall. | `docs/step_segment/diff_gebd.md` | Pending | Tool `chunk_diff_gebd_dataset.py` ready. | ⏳ **Queued for iter_01** |
| **H-GEBD-002** | CSN video backbone outperforms 2D ResNet-50 by capturing 3D spatiotemporal hand flow. | `docs/step_segment/efficient_gebd.md` | Pending | Architecture implemented in `EfficientGEBD`. | ⏳ **Queued for iter_01** |
