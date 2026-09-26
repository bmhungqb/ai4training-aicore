# Experiment: exp_001_efficientgebd_baseline

- **Track**: `step_segment`
- **Iteration**: `iter_01`
- **Model**: `EfficientGEBD` (ResNet-50 backbone + DiffFormer temporal head)
- **Dataset**: `data/efficient_gebd_dataset/`

## Objective
Establish the baseline event boundary detection metrics for EfficientGEBD on Chuyền 1 sewing operations, testing the effectiveness of:
1. 10-second slice-based temporal sampling (resolving the 73.5% boundary loss of whole-video linspace).
2. BCE class imbalance weighting (`POS_WEIGHT: 4.5`).
3. Single Gaussian smoothing without re-smoothing.

## How to Run

```bash
# 1. Full Training & Inference
python experiments/step_segment/iter_01/exp_001_efficientgebd_baseline/run.py --mode train

# 2. Inference Only (with existing checkpoint)
python experiments/step_segment/iter_01/exp_001_efficientgebd_baseline/run.py --mode infer
```

## Expected Outputs
All outputs are saved to `outputs/step_segment/iter_01/exp_001_efficientgebd_baseline/`:
- `run.log`: Console logging.
- `model_best.pth`: Best checkpoint.
- `predictions.json`: Predicted boundary timestamps per video.
- `metrics.json`: Standardized metrics report (F1@0.25s, F1@0.5s, F1@1.0s, per-video breakdown).
