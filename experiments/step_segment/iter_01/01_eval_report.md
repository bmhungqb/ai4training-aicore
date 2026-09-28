# Structured Evaluation Report: iter_01

- **Track**: `step_segment`
- **Experiment ID**: `sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75`
- **Model Type**: `unspecified`
- **Evaluated Output Directory**: `/home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75`
- **Timestamp**: 2026-09-27 23:22:59

---

## 1. Executive Performance Fact Sheet

| Tolerance Window | Macro F1 | Recall | Precision | Status |
| :--- | :--- | :--- | :--- | :--- |
| **+/- 0.25s** | **22.75%** | 22.70% | 22.80% | ❌ |
| **+/- 0.50s** **(Primary)** | **40.77%** | 40.67% | 40.86% | ❌ |
| **+/- 1.00s** | **58.56%** | 58.43% | 58.69% | ⚠️ |

---

## 2. Training Log Health & Dynamics Analysis

- **Recorded Epochs**: 0
- **Loss Trajectory**: Initial `N/A` -> Final `N/A` (Minimum: `N/A`)
- **Peak Validation F1**: `N/A`

### Detected Training Anomalies:
- ✓ No critical training anomalies detected (smooth loss convergence, no NaN/divergence).

---

## 3. Top Severe Error Cases (Visual Evidence)

These cases represent the most severe failure modes (missed transitions or false triggers) prioritized for root-cause diagnosis:

### Case 1: `cd5_chuyen3_FN_t3.05s` (🔴 FALSE NEGATIVE (MISSED GT))
- **Video ID**: `cd5_chuyen3`
- **Event Timestamp**: `3.05s`
- **Discrepancy**: Nearest prediction is 7.742s away (Severity Score: `7.742`)
- **Factual Observation**: Ground-truth boundary at 3.05s completely missed by model (closest prediction was 7.74s away).
- **Visual Frame Sequence**: [View Strip Image](file:////home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd5_chuyen3_FA_t3.05s_strip.jpg)
  ![Visual Frame Strip](/home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd5_chuyen3_FA_t3.05s_strip.jpg)

### Case 2: `cd12_chuyen1_FP_t94.29s` (🟠 FALSE POSITIVE (SPURIOUS))
- **Video ID**: `cd12_chuyen1`
- **Event Timestamp**: `94.29s`
- **Discrepancy**: Nearest GT is 8.728s away (Severity Score: `8.728`)
- **Factual Observation**: Spurious boundary predicted at 94.29s where no ground-truth boundary exists (closest GT is 8.73s away).
- **Visual Frame Sequence**: [View Strip Image](file:////home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd12_chuyen1_FA_t94.29s_strip.jpg)
  ![Visual Frame Strip](/home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd12_chuyen1_FA_t94.29s_strip.jpg)

### Case 3: `cd4_chuyen1_FN_t3.43s` (🔴 FALSE NEGATIVE (MISSED GT))
- **Video ID**: `cd4_chuyen1`
- **Event Timestamp**: `3.43s`
- **Discrepancy**: Nearest prediction is 7.365s away (Severity Score: `7.365`)
- **Factual Observation**: Ground-truth boundary at 3.43s completely missed by model (closest prediction was 7.37s away).
- **Visual Frame Sequence**: [View Strip Image](file:////home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd4_chuyen1_FA_t3.43s_strip.jpg)
  ![Visual Frame Strip](/home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd4_chuyen1_FA_t3.43s_strip.jpg)

### Case 4: `cd12_chuyen1_FP_t66.25s` (🟠 FALSE POSITIVE (SPURIOUS))
- **Video ID**: `cd12_chuyen1`
- **Event Timestamp**: `66.25s`
- **Discrepancy**: Nearest GT is 7.111s away (Severity Score: `7.111`)
- **Factual Observation**: Spurious boundary predicted at 66.25s where no ground-truth boundary exists (closest GT is 7.11s away).
- **Visual Frame Sequence**: [View Strip Image](file:////home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd12_chuyen1_FA_t66.25s_strip.jpg)
  ![Visual Frame Strip](/home/hungbm/ai4training/ai4training-aicore/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd12_chuyen1_FA_t66.25s_strip.jpg)

---

## 4. Per-Video Breakdown Ranking (Worst to Best)

| Rank | Video ID | F1 @ 0.5s | Recall | Precision | GT Count | Pred Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `cd4_chuyen1` | **21.1%** | 22.2% | 20.0% | 9 | 10 |
| 2 | `cd12_chuyen1` | **26.0%** | 30.8% | 22.5% | 52 | 71 |
| 3 | `cd10_chuyen1` | **33.3%** | 33.3% | 33.3% | 9 | 9 |
| 4 | `cd8_chuyen2` | **39.7%** | 40.5% | 38.9% | 121 | 126 |
| 5 | `cd5_chuyen3` | **40.0%** | 39.4% | 40.6% | 33 | 32 |
| 6 | `cd10_chuyen2` | **43.6%** | 41.4% | 46.2% | 29 | 26 |
| 7 | `cd6_chuyen2` | **44.0%** | 38.5% | 51.3% | 52 | 39 |
| 8 | `cd19_chuyen3` | **47.2%** | 47.7% | 46.7% | 88 | 90 |
| 9 | `cd12_chuyen2` | **52.2%** | 46.2% | 60.0% | 52 | 40 |

---

> [!CAUTION]
> **Evaluator Protocol Rule**: This report contains strictly observable metrics and extracted visual evidence. Speculation on root causes (why it happened) is deferred to `@research-diagnoser`, and solutions belong to `@research-planner`.
