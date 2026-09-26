# Iteration 01: Multi-Model Step Segmentation Benchmark

- **Track**: `step_segment`
- **Iteration**: `iter_01`
- **Objective**: Establish the official baseline cross-model comparison on Chuyền 1 across all 3 deep learning boundary detection models.
- **Evaluation Criteria**: Absolute tolerance F1 at $\pm 0.5$s window, plus strict ($\pm 0.25$s) and loose ($\pm 1.0$s) tolerances.

## Batch Experiments

| ID | Model Architecture | Focus / Tested Mitigation | Runner Command |
| :--- | :--- | :--- | :--- |
| [`exp_001`](exp_001_efficientgebd_baseline/) | `EfficientGEBD` (ResNet-50) | 10s slice-based temporal sampling + `POS_WEIGHT: 4.5` | `python experiments/step_segment/iter_01/exp_001_efficientgebd_baseline/run.py` |
| [`exp_002`](exp_002_diffgebd_chunked/) | `DiffGEBD` (ResNet-50) | 12s overlapping chunked dataset + CFG scale 7.0 | `python experiments/step_segment/iter_01/exp_002_diffgebd_chunked/run.py` |
| [`exp_003`](exp_003_ddm_net_baseline/) | `DDM-Net` (ResNet-50) | Re-weighted auxiliary loss heads (`main + 0.3 * aux`) | `python experiments/step_segment/iter_01/exp_003_ddm_net_baseline/run.py` |

## Batch Execution

To run all 3 experiments in batch:
```bash
# Run training:
bash experiments/step_segment/iter_01/run_all.sh train

# Run inference only (if checkpoints already exist):
bash experiments/step_segment/iter_01/run_all.sh infer
```

Outputs will be stored in:
`outputs/step_segment/iter_01/<exp_id>/`
- `predictions.json`
- `metrics.json`
- `checkpoints/`
- `run.log`
