# Step Segmentation Models: Master Comparison Tracker

> **Single Source of Truth** tracking and comparing all experiments across the 3 deep learning generic event boundary detection models: **DDM-Net**, **DiffGEBD**, and **EfficientGEBD**.

---

## 1. Cross-Model Leaderboard

*Ranked by primary metric: **Macro F1 at $\pm 0.5$s window tolerance** on Chuyền 1 sewing operations.*

| Rank | Model Architecture | Experiment ID | Backbone | Macro F1 (0.5s) | Recall (0.5s) | Precision (0.5s) | F1 @ 0.25s | F1 @ 1.0s | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| - | **EfficientGEBD** | [`exp_001_baseline`](efficient_gebd/iter_01/exp_001_baseline/) | ResNet-50 | - | - | - | - | - | Ready to benchmark |
| - | **DiffGEBD** | [`exp_001_baseline`](diff_gebd/iter_01/exp_001_baseline/) | ResNet-50 | - | - | - | - | - | Chunked baseline ready |
| - | **DDM-Net** | [`exp_001_baseline`](ddm_net/iter_01/exp_001_baseline/) | ResNet-50 | - | - | - | - | - | Aux re-weighted ready |

*Target Criteria for Champion Model*: Macro F1 $\ge 65.0\%$ at $\pm 0.5$s, Inference speed $\ge 25$ fps, VRAM $\le 8$GB.

---

## 2. Chronological Experiment Results Log (All Models)

*Append every completed experiment run here to maintain a complete historical ledger.*

| Timestamp | Model | Experiment ID | Path / Config | Key Tested Variable | Macro F1 (0.5s) | Recall | Precision | Verdict / Notes |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| Planned | `EfficientGEBD` | `exp_001_baseline` | [config.yaml](efficient_gebd/iter_01/exp_001_baseline/config.yaml) | 10s slice sampling + `POS_WEIGHT: 4.5` | - | - | - | Pending Human execution |
| Planned | `DiffGEBD` | `exp_001_baseline` | [config.yaml](diff_gebd/iter_01/exp_001_baseline/config.yaml) | 12s overlapping chunks + CFG 7.0 | - | - | - | Pending Human execution |
| Planned | `DDM-Net` | `exp_001_baseline` | [config.yaml](ddm_net/iter_01/exp_001_baseline/config.yaml) | Weighted aux heads (`main + 0.3*aux`) | - | - | - | Pending Human execution |

---

## 3. Model-Specific Experiment Tracks

Each model maintains its own isolated experiment hierarchy, configs, and output folders:

### 3.1. DDM-Net Track ([`experiments/step_segment/ddm_net/`](ddm_net/))
- **Paradigm**: Dual-stream spatial RGB + dense difference motion (DDM) with Co-Transformer decoder.
- **Reference Doc**: [`docs/step_segment/ddm_net.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/ddm_net.md)
- **Active Iteration**: [`iter_01`](ddm_net/iter_01/exp_001_baseline/)
- **Outputs**: `outputs/step_segment/ddm_net/iter_XX/exp_YY/`

### 3.2. DiffGEBD Track ([`experiments/step_segment/diff_gebd/`](diff_gebd/))
- **Paradigm**: Denoising diffusion generative model (DDPM/DDIM) conditioned on visual similarity + CFG.
- **Reference Doc**: [`docs/step_segment/diff_gebd.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/diff_gebd.md)
- **Active Iteration**: [`iter_01`](diff_gebd/iter_01/exp_001_baseline/)
- **Outputs**: `outputs/step_segment/diff_gebd/iter_XX/exp_YY/`

### 3.3. EfficientGEBD Track ([`experiments/step_segment/efficient_gebd/`](efficient_gebd/))
- **Paradigm**: Temporal sliding-window Feature Pyramid Network + DiffFormer / DiffMixer dissimilarity.
- **Reference Doc**: [`docs/step_segment/efficient_gebd.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/efficient_gebd.md)
- **Active Iteration**: [`iter_01`](efficient_gebd/iter_01/exp_001_baseline/)
- **Outputs**: `outputs/step_segment/efficient_gebd/iter_XX/exp_YY/`

---

## 4. Architectural & Resource Profile

| Feature | EfficientGEBD | DiffGEBD | DDM-Net |
| :--- | :--- | :--- | :--- |
| **Model Type** | Deterministic Temporal Detector | Generative Diffusion (DDIM) | Deterministic Dual-Stream Classifier |
| **Backbone** | ResNet-50 / Video CSN | ResNet-50 | ResNet-50 / DINOv2 (`dinov2_vitb14`) |
| **Sampling Mechanism** | 10s slice-based (100 frames/slice) | 12s overlapping chunks (150 frames) | Dense streaming validation (`temporal_stride=1`) |
| **Loss Formulation** | Weighted BCE (`POS_WEIGHT: 4.5`) | Boundary MSE (unweighted) | Re-weighted multi-layer CrossEntropy |
| **Inference Mechanism** | 1 forward pass + peak thresholding | 16 DDIM reverse-diffusion steps + CFG | 1 forward pass + 1D NMS filter |
| **Inference Latency** | Ultra fast (~1,200 fps features) | Moderate (16 diffusion passes) | Fast (~340 fps) |
| **VRAM Consumption** | Low (~4.2 GB) | Medium (~6-8 GB in e2e mode) | Medium (~6.1 GB) |
| **Primary Strength** | Fast convergence, high compute efficiency | Strong temporal boundary modeling | Dual-stream spatial + motion synergy |
| **Key Limitation** | 10s slice boundary context loss | Slower inference due to diffusion steps | Flat loss if aux heads unweighted |

---

## 5. Standard Evaluation Tool

All 3 models are benchmarked using the shared evaluation tool:
```bash
python tools/eval_step_segment_predictions.py \
  --pred outputs/step_segment/<model>/<iter>/<exp_id>/predictions.json \
  --data-dir data \
  --out outputs/step_segment/<model>/<iter>/<exp_id>/metrics.json \
  --thresholds 0.25 0.5 1.0
```
This produces the exact schema ingested by the `research-evaluator` skill to update this overview table.
