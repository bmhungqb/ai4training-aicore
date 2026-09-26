# Experiment: exp_002_diffgebd_chunked

- **Track**: `step_segment`
- **Iteration**: `iter_01`
- **Model**: `DiffGEBD` (ResNet-50 + Denoising Diffusion Generative Model)
- **Dataset**: `data/diff_gebd_dataset/` (Chunked clips: ~12s @ 15fps)

## Objective
Benchmark DiffGEBD under the chunked dataset regime to determine if the sampling imbalance (precision ~0.9 vs recall ~0.19) is solved by keeping effective stride $\approx 1:1$.

## Prerequisites
Ensure chunked dataset is prepared:
```bash
python tools/prepare_diff_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42
python tools/chunk_diff_gebd_dataset.py --chunk-seconds 12 --overlap-seconds 1.5
```

## How to Run

```bash
# 1. Full Training & Inference
python experiments/step_segment/iter_01/exp_002_diffgebd_chunked/run.py --mode train

# 2. Inference Only
python experiments/step_segment/iter_01/exp_002_diffgebd_chunked/run.py --mode infer
```

## Expected Outputs
Outputs stored in `outputs/step_segment/iter_01/exp_002_diffgebd_chunked/`:
- `run.log`: Training & inference logs.
- `model_best.pth`: Best EMA checkpoint.
- `predictions.json`: Boundary predictions.
- `metrics.json`: Standardized evaluation report.
