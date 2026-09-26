# Step Segmentation: Deep Learning Event Boundary Detection

Framework for training, evaluating, and benchmarking deep learning Generic Event Boundary Detection (GEBD) models on industrial garment manufacturing operations (Chuyền 1, Chuyền 2, Chuyền 3).

Three competitive architectures are integrated, standardized, and benchmarked: **DDM-Net**, **DiffGEBD**, and **EfficientGEBD**.

---

## Model Architectures & Profiles

| Feature | EfficientGEBD | DiffGEBD | DDM-Net |
| :--- | :--- | :--- | :--- |
| **Model Paradigm** | Deterministic Temporal Detector | Generative Diffusion (DDIM) | Deterministic Dual-Stream Classifier |
| **Backbone** | ResNet-50 / Video CSN | ResNet-50 | ResNet-50 / DINOv2 (`dinov2_vitb14`) |
| **Sampling Mechanism** | 10s slice-based (100 frames/slice) | 12s overlapping chunks (150 frames) | Dense streaming validation (`temporal_stride=1`) |
| **Loss Formulation** | Weighted BCE (`POS_WEIGHT: 4.5`) | Boundary MSE (`POS_LOSS_WEIGHT: 5.0`) | Re-weighted multi-layer CrossEntropy |
| **Inference Mechanism** | 1 forward pass + peak thresholding | 16 DDIM reverse-diffusion steps + CFG | 1 forward pass + 1D NMS filter |
| **Inference Latency** | Ultra fast (~1,200 fps features) | Moderate (16 diffusion passes) | Fast (~340 fps) |
| **VRAM Consumption** | Low (~4.2 GB) | Medium (~6-8 GB in e2e mode) | Medium (~6.1 GB) |
| **Primary Strength** | Fast convergence, high compute efficiency | Strong temporal boundary modeling | Dual-stream spatial + motion synergy |
| **Key Limitation** | 10s slice boundary context loss | Slower inference due to diffusion steps | Flat loss if aux heads unweighted |

---

## Directory Structure

```text
src/step_segment/
├── DDM-Net/                    # Dual-stream spatial RGB + dense difference motion
│   ├── train_sop_lightning.py  # PyTorch Lightning trainer (re-weighted loss & leak-free)
│   ├── requirements.txt        # DDM-Net dependencies
│   └── ...
│
├── DiffGEBD/                   # Denoising diffusion generative model
│   ├── train.py                # Distributed / torchrun trainer
│   ├── requirements.txt        # DiffGEBD dependencies
│   └── ...
│
└── EfficientGEBD/              # Temporal FPN + DiffFormer / DiffMixer
    ├── train.py                # torchrun trainer with single best checkpointing
    ├── infer.py                # Standalone inference & prediction export
    ├── requirements.txt        # EfficientGEBD dependencies
    └── ...
```

---

## Dataset Formats & Preparation

Each model operates on a pre-processed representation derived from `data/`:

| Model | Dataset Path | Preparation Command |
|---|---|---|
| **DDM-Net** | `data/ddm_dataset/` | `python tools/process_step_segments.py && python tools/prepare_ddm_dataset.py --split-mode random --val-ratio 0.2 --seed 42` |
| **DiffGEBD** | `data/diff_gebd_dataset/` | `python tools/prepare_diff_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42 && python tools/chunk_diff_gebd_dataset.py --chunk-seconds 12 --overlap-seconds 1.5` |
| **EfficientGEBD** | `data/efficient_gebd_dataset/` | `python tools/prepare_efficient_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42` |

---

## Standardized Experiment Protocol

All three models follow the **Human-in-the-Loop AI Research Loop**:
1. **Isolated Experiments**: Each model maintains its own experiment tree under `experiments/step_segment/<model>/iter_XX/exp_YY/`.
2. **Standard Artifacts**:
   - Checkpoint: Only saves the single best epoch checkpoint (`model_best.pth` / `best_model.ckpt`), preventing disk overflow.
   - Logs: Real-time logging to `train.log`, `infer.log`, and `run.log`.
   - Predictions: Standardized `{video_id: [timestamp_seconds]}` format in `predictions.json`.
   - Metrics: Evaluated via `tools/eval_step_segment_predictions.py` into `metrics.json`.
3. **Master Comparison Tracker**:
   - Results from all 3 models are automatically logged and compared in:
     [`experiments/step_segment/step_segment_overview.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/step_segment/step_segment_overview.md)

---

## Running Experiments

### Option A: Run All 3 Models Sequentially (Master Batch)
```bash
bash experiments/step_segment/run_all_baselines.sh
```

### Option B: Run Individual Models

```bash
# 1. EfficientGEBD Baseline
python experiments/step_segment/efficient_gebd/iter_01/exp_001_baseline/run.py --mode train

# 2. DiffGEBD Baseline
python experiments/step_segment/diff_gebd/iter_01/exp_001_baseline/run.py --mode train

# 3. DDM-Net Baseline
python experiments/step_segment/ddm_net/iter_01/exp_001_baseline/run.py --mode train
```

### Option C: Inference Only (Existing Checkpoint)
```bash
python experiments/step_segment/<model>/iter_01/exp_001_baseline/run.py --mode infer
```

---

## Shared Evaluation Tool

To benchmark any prediction file against ground truth:
```bash
python tools/eval_step_segment_predictions.py \
  --pred outputs/step_segment/<model>/<iter>/<exp_id>/predictions.json \
  --data-dir data \
  --out outputs/step_segment/<model>/<iter>/<exp_id>/metrics.json \
  --thresholds 0.25 0.5 1.0
```
Key primary metric: **Macro F1 at $\pm 0.5$s tolerance window**.
