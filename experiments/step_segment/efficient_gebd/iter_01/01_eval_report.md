# Structured Evaluation Report: iter_01

- **Track**: `step_segment`
- **Model**: `EfficientGEBD`
- **Iteration / Baseline**: `iter_01` (`exp_001_baseline`)
- **Evaluated Artifacts Source**: `outputs/step_segment/efficient_gebd/iter_01/exp_001_baseline/`
- **Timestamp**: 2026-09-28T12:46:10+07:00

---

## 1. Executive Performance Fact Sheet

| Tolerance Window | Macro F1 | Recall | Precision | Status |
| :--- | :--- | :--- | :--- | :--- |
| **+/- 0.25s** | **34.83%** | 53.26% | 25.87% | ⚠️ Tight alignment |
| **+/- 0.50s (Primary)** | **57.60%** | **88.09%** | **42.79%** | 🏆 **Current Champion Baseline** |
| **+/- 1.00s** | **63.34%** | 96.85% | 47.05% | ✅ Near-complete boundary coverage |

### Quantitative Overview:
- **Total Ground-Truth Boundaries**: 445
- **Total Predicted Boundaries**: 916
- **True Positives (@0.5s)**: 392 / 445 (**88.09%**)
- **False Positives (@0.5s)**: 524 / 916 (**57.21%**)
- **False Negatives (@0.5s)**: 53 / 445 (**11.91%**)
- **Tolerance Scaling**: F1 increases from 34.83% (@0.25s) to 57.60% (@0.50s) and 63.34% (@1.00s). Recall reaches 96.85% at 1.00s tolerance, establishing that the model successfully locates virtually all boundary events with strong temporal proximity.

---

## 2. Training Log Health & Dynamics Analysis

- **Log File**: `outputs/step_segment/efficient_gebd/iter_01/exp_001_baseline/train.log`
- **Recorded Epochs**: 16 epochs (Epoch 00 through Epoch 15). Early stopping triggered with patience 5.
- **Loss / Convergence Trajectory**:
  - Training loss (`total_loss` / `head_x2`) began at 1.1770 and stabilized around 1.0671.
  - Learning rate maintained at constant `1e-4` with warmup.
  - No gradient explosion, NaN values, or numerical instability observed.
- **Validation F1 Dynamics**:
  - Peak validation performance achieved at **Epoch 00** (`F1@0.05: 0.6583`, Recall: 1.0000, Precision: 0.4906) and preserved as `model_best.pth`.
  - Later epochs hovered between F1 0.4850 and 0.6578, triggering early stopping after epoch 15.

---

## 3. Top Severe Error Cases (Quantitative Discrepancies)

### Case 1: `cd12_chuyen1` Over-segmentation (False Positives)
- **Video ID**: `cd12_chuyen1`
- **Discrepancy**: Model emitted 169 predicted boundaries for 52 ground-truth boundaries (ratio: 3.25x).
- **Factual Observation**: While recall is high at 90.38% (47/52 TP), precision drops to 27.81%, resulting in the lowest per-video F1 on the benchmark (42.53%).
- **Evidence Reference**: `outputs/step_segment/efficient_gebd/iter_01/exp_001_baseline/predictions.json`

### Case 2: `cd4_chuyen1` Over-segmentation (False Positives)
- **Video ID**: `cd4_chuyen1`
- **Discrepancy**: Model produced 30 predicted boundaries for 9 ground-truth boundaries (ratio: 3.33x).
- **Factual Observation**: Perfect recall (100.00%, 9/9 boundaries found), but precision is constrained to 30.00% due to 21 spurious detections.
- **Evidence Reference**: `outputs/step_segment/efficient_gebd/iter_01/exp_001_baseline/predictions.json`

### Case 3: `cd8_chuyen2` Multi-Peak Clustering (False Positives)
- **Video ID**: `cd8_chuyen2`
- **Discrepancy**: Model produced 255 predicted boundaries for 121 ground-truth boundaries (134 excess predictions).
- **Factual Observation**: Captures 116 / 121 true boundaries (95.87% recall), but precision remains at 45.49%.
- **Evidence Reference**: `outputs/step_segment/efficient_gebd/iter_01/exp_001_baseline/predictions.json`

### Case 4: `cd6_chuyen2` Boundary Misses (False Negatives)
- **Video ID**: `cd6_chuyen2`
- **Discrepancy**: 13 missed boundaries out of 52 ground-truth boundaries.
- **Factual Observation**: This video recorded the lowest recall (75.00%) across the entire validation split, indicating specific sewing sub-operations were not triggered above the 0.2 score threshold.
- **Evidence Reference**: `outputs/step_segment/efficient_gebd/iter_01/exp_001_baseline/predictions.json`

---

## 4. Per-Video Breakdown Ranking (Worst to Best)

| Rank | Video ID | F1 @ 0.5s | Recall | Precision | GT Count | Pred Count | TP Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `cd12_chuyen1` | **42.53%** | 90.38% | 27.81% | 52 | 169 | 47 |
| 2 | `cd4_chuyen1` | **46.15%** | 100.00% | 30.00% | 9 | 30 | 9 |
| 3 | `cd10_chuyen1` | **48.48%** | 88.89% | 33.33% | 9 | 24 | 8 |
| 4 | `cd5_chuyen3` | **53.21%** | 87.88% | 38.16% | 33 | 76 | 29 |
| 5 | `cd19_chuyen3` | **57.60%** | 81.82% | 44.44% | 88 | 162 | 72 |
| 6 | `cd6_chuyen2` | **58.65%** | 75.00% | 48.15% | 52 | 81 | 39 |
| 7 | `cd8_chuyen2` | **61.70%** | 95.87% | 45.49% | 121 | 255 | 116 |
| 8 | `cd10_chuyen2` | **70.00%** | 96.55% | 54.90% | 29 | 51 | 28 |
| 9 | `cd12_chuyen2` | **73.33%** | 84.62% | 64.71% | 52 | 68 | 44 |

---

## 5. What Does The Model Do Correctly?
- **High Sensitivity & Coverage**: Captured 392 of 445 boundaries (88.09% recall at 0.5s), scaling to 96.85% at 1.0s. It completely eliminated the severe FN blind-spots seen in DDM-Net (where recall was only 17.08%).
- **Strong Top-Performer Consistency**: Achieved $\ge 70.0\%$ F1 on `cd10_chuyen2` (70.00%) and `cd12_chuyen2` (73.33%) with precision above 54-64%.
- **Zero Loss Divergence**: Dataloading, multi-slice temporal aggregation, and early stopping completed stably in ~30 minutes on a single GPU.

---

> [!CAUTION]
> **Evaluator Rule**: No root causes or solutions are allowed in this document. Any speculative why or how to fix belongs in `02_diagnosis.md` and `03_research_plan.md`.
