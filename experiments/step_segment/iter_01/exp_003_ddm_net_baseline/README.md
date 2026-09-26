# Experiment: exp_003_ddm_net_baseline

- **Track**: `step_segment`
- **Iteration**: `iter_01`
- **Model**: `DDM-Net` (ResNet-50 + Dense Difference Module + Co-Transformer)
- **Dataset**: `data/ddm_dataset/`

## Objective
Benchmark DDM-Net on the Chuyền 1 industrial dataset after applying:
1. Re-weighted loss ($L_{\text{main}} + 0.3 \times \frac{1}{N} \sum L_{\text{aux}}$) to escape the $18 \times \ln(2) \approx 12.47$ loss plateau.
2. Leak-free validation streaming configuration (`workers: 1`, periodic reader release).

## Prerequisites
Ensure DDM dataset is built:
```bash
python tools/process_step_segments.py
python tools/prepare_ddm_dataset.py --split-mode random --val-ratio 0.2 --seed 42
```

## How to Run

```bash
# 1. Full Training & Validation
python experiments/step_segment/iter_01/exp_003_ddm_net_baseline/run.py --mode train

# 2. Inference Only (with existing checkpoint)
python experiments/step_segment/iter_01/exp_003_ddm_net_baseline/run.py --mode infer
```

## Expected Outputs
Outputs stored in `outputs/step_segment/iter_01/exp_003_ddm_net_baseline/`:
- `run.log`: Lightning trainer logs.
- `best_model.ckpt`: Best checkpoint.
- `predictions.json`: Boundary predictions.
- `metrics.json`: Standardized evaluation report.
