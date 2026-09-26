# AI4Training Documentation & Model Trackers

Project documentation is structured into **single-source-of-truth trackers** per method/model. Each tracker consolidates:
1. **Method Info**: Architecture, environment setup, training commands, and core configs.
2. **Status & Issues**: Known issues, root cause analyses, applied fixes, and pending mitigations.
3. **Experiments & Results**: Training logs, quantitative benchmarks, cross-model comparisons, and ablation notes.

---

## Directory Index

### 1. Step Segmentation (`docs/step_segment/`)
Detects generic event boundaries (operation transitions) using deep learning models:

- [`ddm_net.md`](ddm_net.md):
  - **Model**: DDM-Net (NVIDIA SOP Blueprint Architecture + PyTorch Lightning).
  - **Backbone**: ResNet-50 / DINOv2 (`dinov2_vitb14`).
  - **Key Issues**: Flat loss ~12.4 (18 unweighted heads), validation RAM leak (`decord` C++ buffer), temporal window expansion (0.37s $\to$ 1.1s), ROI bbox crop.
- [`diff_gebd.md`](diff_gebd.md):
  - **Model**: DiffGEBD (ICCV 2025 - Denoising Diffusion Generative Model + CFG).
  - **Backbone**: ResNet-50 + DiffFormer diffusion head.
  - **Key Issues**: Sampling imbalance (precision 0.9+ but recall 0.19 from whole-video `np.linspace`), solved via 10-15s video chunking (`tools/chunk_diff_gebd_dataset.py`).
- [`efficient_gebd.md`](efficient_gebd.md):
  - **Model**: EfficientGEBD (ACM MM 2024 - Architecture Redesign for Efficient GEBD).
  - **Backbone**: Video-domain CSN (R50/R152) / ResNet-50 + FPN + DiffFormer.
  - **Key Issues**: 12 tracked issues (slice-based sampling, double Gaussian smoothing fix, BCE class imbalance with `POS_WEIGHT: 4.5`, absolute tolerance eval, station-level data splitting).

---

### 2. Action Segmentation & Evaluation (`docs/action_segment/`)
Kinematic physical action pre-segmentation and VLM-based process evaluation:

- [`stage1_kinematic.md`](stage1_kinematic.md):
  - **Pipeline**: SAM 3 (hand tracking) + SEA-RAFT (dense optical flow) + Magnitude/Direction Fusion (0 VLM, 0 API cost).
  - **v2 Fixes**: Mask erosion ($3\times 3$), wrist-resting translation + RMS energy fusion, linear interpolation for mask drops.
  - **Benchmarks**: 9 independent operations (Chuyền 1: 349 GT steps, 1,505 machine segments, Macro Recall 86.2%, Micro Recall 87.3%, MAE 0.213s).
- [`stage2_vlm_analysis.md`](stage2_vlm_analysis.md):
  - **4-Step Pipeline**: Step 1 Expert Analysis (`expert_analysis.py`), Step 2 Worker Classification (`segment_classify.py`), Step 3 Macro Evaluation (`macro_eval.py`), Step 4 Micro Diagnosis (`micro_eval.py`).
  - **Key Issues**: Resolves 4 static-frame bottlenecks (downsampling blur, visual noise, motion ambiguity, token explosion) via Dynamic Action ROI Crop, Dual-View Composite, and Motion History Image (MHI) vector overlay.
