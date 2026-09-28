# Structured Evaluation Report: iter_01

- **Track**: `step_segment`
- **Iteration / Baseline**: `iter_01`
- **Evaluated Model**: `DiffGEBD` (`exp_001_baseline`)
- **Evaluated Artifacts Source**: `outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/`
- **Timestamp**: 2026-09-27T23:30:00+07:00

---

## 1. Executive Performance Fact Sheet

| Tolerance Window | Macro F1 | Recall | Precision | Status |
| :--- | :--- | :--- | :--- | :--- |
| **+/- 0.25s** | **22.75%** | 22.70% | 22.80% | ❌ Severe temporal offset |
| **+/- 0.50s (Primary)** | **40.77%** | 40.67% | 40.86% | ⚠️ Baseline established |
| **+/- 1.00s** | **58.56%** | 58.43% | 58.69% | ℹ️ Moderate temporal alignment |

### Quantitative Overview:
- **Total Ground-Truth Boundaries**: 445
- **Total Predicted Boundaries**: 443
- **True Positives (@0.5s)**: 181 / 445 (40.67%)
- **False Positives (@0.5s)**: 262 / 443 (59.14%)
- **False Negatives (@0.5s)**: 264 / 445 (59.33%)
- **Tolerance Scaling**: F1 increases from 22.75% (@0.25s) to 40.77% (@0.50s) and 58.56% (@1.00s), demonstrating that predictions are temporally proximate to ground-truth events but exhibit temporal displacement between 0.25s and 1.00s.

---

## 2. Training Log Health & Dynamics Analysis

- **Log File**: `outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/train.log`
- **Loss / Convergence Trajectory**:
  - Denoising MSE loss decreased steadily across training steps without exploding or NaN values.
  - No loss divergence or NaN gradients detected.
- **Validation F1 Dynamics**:
  - Training metrics stabilized with balanced precision and recall (`~40.7%` recall, `~40.9%` precision at 0.5s tolerance).
  - Validation metrics did not exhibit severe post-peak degradation (>8% drop).

---

## 3. Top Severe Error Cases (Visual Evidence)

Prioritizing the top 4 most severe failure modes with extracted video frame strips:

### Case 1: `cd5_chuyen3_FN_t3.05s` (False Negative)
- **Video ID**: `cd5_chuyen3` | **Timestamp**: `3.05s`
- **Discrepancy**: Ground-truth boundary at 3.05s completely missed; closest model prediction is at 7.74s (temporal offset: 4.69s).
- **Factual Visual Observation**: The worker in red fabric station continuously feeds fabric under the presser foot across timestamps `2.70s -> 3.05s -> 3.40s`. No physical pause, hand release, or fabric rotation is visible in the frame strip.
- **Visual Frame Strip**: [`cd5_chuyen3_FN_t3.05s.jpg`](file:///home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/error_cases/cd5_chuyen3_FN_t3.05s.jpg)

### Case 2: `cd12_chuyen1_FP_t94.29s` (False Positive)
- **Video ID**: `cd12_chuyen1` | **Timestamp**: `94.29s`
- **Discrepancy**: Spurious boundary predicted at 94.29s where no ground-truth boundary exists; closest GT boundary is at 85.56s (temporal offset: 8.73s).
- **Factual Visual Observation**: Worker in blue shirt steadily guides fabric into the sewing machine. Head posture, arm angle, and needle interaction remain continuous with no transition or pause.
- **Visual Frame Strip**: [`cd12_chuyen1_FP_t94.29s.jpg`](file:///home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/error_cases/cd12_chuyen1_FP_t94.29s.jpg)

### Case 3: `cd4_chuyen1_FN_t3.43s` (False Negative)
- **Video ID**: `cd4_chuyen1` | **Timestamp**: `3.43s`
- **Discrepancy**: Ground-truth boundary at 3.43s missed; closest model prediction is at 10.80s (temporal offset: 7.37s).
- **Factual Visual Observation**: Continuous sewing on dark fabric. Minor head tilt observed; background person moves arm, but primary operator maintains unbroken sewing feed.
- **Visual Frame Strip**: [`cd4_chuyen1_FN_t3.43s.jpg`](file:///home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/error_cases/cd4_chuyen1_FN_t3.43s.jpg)

### Case 4: `cd12_chuyen1_FP_t66.25s` (False Positive)
- **Video ID**: `cd12_chuyen1` | **Timestamp**: `66.25s`
- **Discrepancy**: Spurious boundary predicted at 66.25s; closest ground-truth boundary is at 73.36s (temporal offset: 7.11s).
- **Factual Visual Observation**: Continuous needle feeding; hand and head posture remain steady across `65.90s -> 66.25s -> 66.60s`.
- **Visual Frame Strip**: [`cd12_chuyen1_FP_t66.25s.jpg`](file:///home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/diff_gebd/iter_01/exp_001_baseline/error_cases/cd12_chuyen1_FP_t66.25s.jpg)

---

## 4. Per-Video Breakdown Ranking (Worst to Best)

*Evaluated on the 9 full validation videos under primary window $\pm 0.5$s:*

| Rank | Video ID | F1 @ 0.5s | Recall | Precision | GT Count | Pred Count | TP Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `cd4_chuyen1` | **21.05%** | 22.22% | 20.00% | 9 | 10 | 2 |
| 2 | `cd12_chuyen1` | **26.02%** | 30.77% | 22.54% | 52 | 71 | 16 |
| 3 | `cd10_chuyen1` | **33.33%** | 33.33% | 33.33% | 9 | 9 | 3 |
| 4 | `cd8_chuyen2` | **39.73%** | 38.67% | 40.85% | 75 | 71 | 29 |
| 5 | `cd5_chuyen3` | **40.00%** | 41.56% | 38.55% | 77 | 83 | 32 |
| 6 | `cd10_chuyen2` | **43.64%** | 42.86% | 44.44% | 28 | 27 | 12 |
| 7 | `cd6_chuyen2` | **44.00%** | 42.31% | 45.83% | 52 | 48 | 22 |
| 8 | `cd19_chuyen3` | **47.17%** | 51.02% | 43.86% | 49 | 57 | 25 |
| 9 | `cd12_chuyen2` | **52.17%** | 53.73% | 50.70% | 94 | 67 | 40 |

---

## 5. What Does The Model Do Correctly?
- **Balanced Recall & Precision**: Recall (40.67%) and Precision (40.86%) are balanced at 0.5s tolerance, showing that the model does not suffer from extreme under-triggering or over-triggering.
- **Boundary Count Calibration**: Total predicted boundaries (443) closely match the total ground-truth boundaries (445) across the dataset.
- **Consistent Operations**: Performance reaches over 52% F1 on longer multi-cycle videos such as `cd12_chuyen2` (52.17% F1).

---

> [!CAUTION]
> **Evaluator Rule**: No root causes or solutions are allowed in this document. Any speculative why or how to fix belongs in `02_diagnosis.md` and `03_research_plan.md`.
