# Iteration 00: Baseline Benchmark (Stage 1 Kinematic Boundary Recall)

- **Track**: `action_segment`
- **Iteration**: `iter_00_baseline`
- **Method**: Kinematic Motion Segmentation (SAM 3 + SEA-RAFT + Magnitude/Direction Fusion)
- **Evaluation Dataset**: Chuyền 1 (9 Công đoạn: CĐ1 -> CĐ9, 9 videos, 349 GT steps, 363 GT boundaries)

## Summary Metrics

| Metric | Baseline | Tuned Final | Unit / Notes |
| :--- | :--- | :--- | :--- |
| **Macro Recall** | **86.20%** | 75.20% | Primary target metric |
| **Micro Recall** | **87.33%** | 72.18% | Boundary-level recall |
| **Step Both Match** | **76.50%** | 52.72% | Both start & end within window |
| **Step Either Match** | **98.28%** | 92.26% | At least one boundary matched |
| **Macro F1** | 46.56% | **48.13%** | Precision-Recall harmonic mean |
| **Tolerance Window** | 0.5s | 0.5s | Temporal window tolerance |

## Artifacts

- [`eval_report.json`](file:///home/hungbm/ai4training/ai4training-aicore/experiments/action_segment/iter_00_baseline/eval_report.json): Raw benchmark outputs across window thresholds (0.25s -> 2.0s) and per-video parameter options.
- `evaluation_result_9cd.xlsx`: Excel export containing detailed per-operation breakdown.

## Observations for Next Iterations

1. Baseline achieves high recall (86.2%) but lower precision (over-segmentation ~1,505 segments vs 349 GT steps).
2. Tuning distance/thresholds increased F1 marginally (48.13%) at the expense of recall dropping significantly (-11% Macro Recall).
3. Primary research challenge: Improving boundary precision without sacrificing recall, especially handling subtle hand pauses and wrist resting.
