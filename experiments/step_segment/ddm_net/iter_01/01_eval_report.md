# Structured Evaluation Report: iter_01

- **Track**: `step_segment`
- **Model**: `DDM-Net`
- **Iteration / Baseline**: `iter_01` (`exp_001_baseline`)
- **Evaluated Artifacts Source**: `outputs/step_segment/ddm_net/iter_01/exp_001_baseline/`
- **Timestamp**: 2026-09-28T11:45:51+07:00

---

## 1. Executive Performance Fact Sheet

| Tolerance Window | Macro F1 | Recall | Precision | Status |
| :--- | :--- | :--- | :--- | :--- |
| **+/- 0.25s** | **14.68%** | 10.11% | 26.79% | ❌ Severe temporal offset |
| **+/- 0.50s (Primary)** | **24.80%** | **17.08%** | **45.24%** | ⚠️ Baseline established (FN-dominated) |
| **+/- 1.00s** | **38.17%** | 26.29% | 69.64% | ℹ️ Moderate precision scaling |

### Quantitative Overview:
- **Total Ground-Truth Boundaries**: 445
- **Total Predicted Boundaries**: 168
- **True Positives (@0.5s)**: 76 / 445 (**17.08%**)
- **False Positives (@0.5s)**: 92 / 168 (**54.76%**)
- **False Negatives (@0.5s)**: 369 / 445 (**82.92%**)
- **Tolerance Scaling**: F1 scales from 14.68% (@0.25s) to 24.80% (@0.50s) and 38.17% (@1.00s). Precision reaches 69.64% at 1.00s tolerance, whereas recall remains capped at 26.29%, confirming that DDM-Net severely under-predicts event boundaries on this sewing dataset.

---

## 2. Training Log Health & Dynamics Analysis

- **Log File**: `outputs/step_segment/ddm_net/iter_01/exp_001_baseline/train.log`
- **Recorded Epochs**: 30 epochs (completed 30 full training epochs with Lightning trainer).
- **Loss / Convergence Trajectory**:
  - Main loss decreased steadily with auxiliary loss weighting `training_config.aux_loss_weight=0.3`.
  - No gradient explosion or NaN values observed.
- **Validation Metrics Dynamics**:
  - Validation metrics stabilized with high precision relative to recall (Precision: 45.24% vs Recall: 17.08% at 0.5s).
  - Validation boundary score predictions are conservative, emitting roughly ~19 predictions per video regardless of video duration or actual GT boundary count.

---

## 3. Top Severe Error Cases (Quantitative Discrepancies)

### Case 1: `cd6_chuyen2` Severe Boundary Dropout (False Negatives)
- **Video ID**: `cd6_chuyen2`
- **Discrepancy**: Model emitted only 8 predictions for 52 ground-truth boundaries (recall: 3.85%, 2/52 TP).
- **Factual Observation**: Lowest F1 across the entire benchmark (6.67%). 50 of 52 ground-truth boundaries went undetected.
- **Evidence Reference**: `outputs/step_segment/ddm_net/iter_01/exp_001_baseline/predictions.json`

### Case 2: `cd8_chuyen2` Massive Under-Prediction (False Negatives)
- **Video ID**: `cd8_chuyen2`
- **Discrepancy**: Model emitted only 23 predictions for 121 ground-truth boundaries (recall: 9.92%, 12/121 TP).
- **Factual Observation**: 109 missed boundaries across a long multi-cycle recording. Precision on the few emitted predictions was 52.17%.
- **Evidence Reference**: `outputs/step_segment/ddm_net/iter_01/exp_001_baseline/predictions.json`

### Case 3: `cd19_chuyen3` Boundary Sparsity (False Negatives)
- **Video ID**: `cd19_chuyen3`
- **Discrepancy**: Model emitted only 19 predictions for 88 ground-truth boundaries (recall: 14.77%, 13/88 TP).
- **Factual Observation**: Despite high precision (68.42%), F1 remains capped at 24.30% due to 75 missed boundaries.
- **Evidence Reference**: `outputs/step_segment/ddm_net/iter_01/exp_001_baseline/predictions.json`

### Case 4: `cd4_chuyen1` Low Precision on Short Sequence (False Positives)
- **Video ID**: `cd4_chuyen1`
- **Discrepancy**: Model generated 17 predictions for 9 ground-truth boundaries with only 4 TP.
- **Factual Observation**: Precision dropped to 23.53% with 13 false positives.
- **Evidence Reference**: `outputs/step_segment/ddm_net/iter_01/exp_001_baseline/predictions.json`

---

## 4. Per-Video Breakdown Ranking (Worst to Best)

| Rank | Video ID | F1 @ 0.5s | Recall | Precision | GT Count | Pred Count | TP Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `cd6_chuyen2` | **6.67%** | 3.85% | 25.00% | 52 | 8 | 2 |
| 2 | `cd5_chuyen3` | **14.29%** | 12.12% | 17.39% | 33 | 23 | 4 |
| 3 | `cd8_chuyen2` | **16.67%** | 9.92% | 52.17% | 121 | 23 | 12 |
| 4 | `cd12_chuyen1` | **19.72%** | 13.46% | 36.84% | 52 | 19 | 7 |
| 5 | `cd19_chuyen3` | **24.30%** | 14.77% | 68.42% | 88 | 19 | 13 |
| 6 | `cd4_chuyen1` | **30.77%** | 44.44% | 23.53% | 9 | 17 | 4 |
| 7 | `cd12_chuyen2` | **33.80%** | 23.08% | 63.16% | 52 | 19 | 12 |
| 8 | `cd10_chuyen2` | **56.00%** | 48.28% | 66.67% | 29 | 21 | 14 |
| 9 | `cd10_chuyen1` | **57.14%** | 88.89% | 42.11% | 9 | 19 | 8 |

---

## 5. What Does The Model Do Correctly?
- **High Detection Precision on Confident Events**: On videos like `cd19_chuyen3` (68.42%), `cd10_chuyen2` (66.67%), and `cd12_chuyen2` (63.16%), emitted boundary predictions reliably correspond to real physical step transitions.
- **Top Performer on Short Operations**: On `cd10_chuyen1`, achieved 57.14% F1 and 88.89% recall (8/9 boundaries detected).
- **Execution & Training Stability**: Completed 30 full training epochs with dual-stream RGB + dense difference motion with zero CUDA OOMs or divergence.

---

> [!CAUTION]
> **Evaluator Rule**: No root causes or solutions are allowed in this document. Any speculative why or how to fix belongs in `02_diagnosis.md` and `03_research_plan.md`.
