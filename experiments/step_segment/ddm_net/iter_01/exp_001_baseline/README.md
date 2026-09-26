# Experiment: exp_001_baseline

- **Track**: `step_segment`
- **Model**: `DDM-Net` (ResNet-50 + Dense Difference Module + Co-Transformer)
- **Iteration**: `iter_01`
- **Dataset**: `data/ddm_dataset/`

## Objective
Benchmark DDM-Net on the Chuyền 1 industrial dataset after applying:
1. Re-weighted loss ($L_{\text{main}} + 0.3 \times \frac{1}{N} \sum L_{\text{aux}}$) to escape the $18 \times \ln(2) \approx 12.47$ loss plateau.
2. Leak-free validation streaming configuration (`workers: 1`, periodic reader release).
3. Real-time logging (`train.log`, `run.log`) and single best checkpoint retention (`best_model.ckpt`).

## Prerequisites
Ensure DDM dataset is built:
```bash
python tools/process_step_segments.py
python tools/prepare_ddm_dataset.py --split-mode random --val-ratio 0.2 --seed 42
```

## How to Run

```bash
# 1. Full Training & Validation
python experiments/step_segment/ddm_net/iter_01/exp_001_baseline/run.py --mode train

# 2. Inference Only (with existing checkpoint)
python experiments/step_segment/ddm_net/iter_01/exp_001_baseline/run.py --mode infer
```

## Expected Outputs
Outputs stored in `outputs/step_segment/ddm_net/iter_01/exp_001_baseline/`:
- `run.log`, `train.log`: Real-time trainer logs.
- `best_model.ckpt`: Best checkpoint (single best saved).
- `predictions.json`: Boundary predictions ({video_id: [timestamps]}).
- `metrics.json`: Standardized evaluation report (Macro F1@0.5s, Recall, Precision).
