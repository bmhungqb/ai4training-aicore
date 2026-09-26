# Experiment: exp_001_baseline

- **Track**: `step_segment`
- **Model**: `DiffGEBD` (ResNet-50 + Denoising Diffusion Generative Model)
- **Iteration**: `iter_01`
- **Dataset**: `data/diff_gebd_dataset/` (Chunked clips: ~12s @ 15fps)

## Objective
Benchmark DiffGEBD under the chunked dataset regime to establish baseline performance on sewing operations and solve sampling imbalance with effective stride $\approx 1:1$.

## Prerequisites
Ensure chunked dataset is prepared:
```bash
python tools/prepare_diff_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42
python tools/chunk_diff_gebd_dataset.py --chunk-seconds 12 --overlap-seconds 1.5
```

## How to Run

```bash
# 1. Full Training & Inference
python experiments/step_segment/diff_gebd/iter_01/exp_001_baseline/run.py --mode train

# 2. Inference Only
python experiments/step_segment/diff_gebd/iter_01/exp_001_baseline/run.py --mode infer
```

## Expected Outputs
Outputs stored in `outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/`:
- `run.log`, `train.log`, `infer.log`: Real-time training & inference logs.
- `model_best.pth`: Best checkpoint (single best saved).
- `predictions.json`: Boundary predictions ({video_id: [timestamps]}).
- `metrics.json`: Standardized evaluation report (Macro F1@0.5s, Recall, Precision).
