# Experiment: exp_001_baseline

- **Track**: `step_segment`
- **Model**: `EfficientGEBD` (ResNet-50 backbone + DiffFormer temporal head)
- **Iteration**: `iter_01`
- **Dataset**: `data/efficient_gebd_dataset/`

## Objective
Establish the baseline event boundary detection metrics for EfficientGEBD on Chuyền 1 sewing operations, testing the effectiveness of:
1. 10-second slice-based temporal sampling (resolving the 73.5% boundary loss of whole-video linspace).
2. BCE class imbalance weighting (`POS_WEIGHT: 4.5`).
3. Single Gaussian smoothing without re-smoothing.
4. Real-time logging (`train.log`, `infer.log`) and single best checkpoint retention (`model_best.pth`).

## How to Run

```bash
# 1. Full Training & Inference
python experiments/step_segment/efficient_gebd/iter_01/exp_001_baseline/run.py --mode train

# 2. Inference Only (with existing checkpoint)
python experiments/step_segment/efficient_gebd/iter_01/exp_001_baseline/run.py --mode infer
```

## Expected Outputs
All outputs are saved to `outputs/step_segment/efficient_gebd/iter_01/exp_001_baseline/`:
- `run.log`, `train.log`, `infer.log`: Real-time training & inference logs.
- `model_best.pth`: Best checkpoint (single best saved).
- `predictions.json`: Predicted boundary timestamps per video.
- `metrics.json`: Standardized metrics report (F1@0.25s, F1@0.5s, F1@1.0s, per-video breakdown).
