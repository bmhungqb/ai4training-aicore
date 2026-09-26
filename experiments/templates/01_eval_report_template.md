# Structured Evaluation Report: {{ITERATION_ID}}

- **Track**: `{{TRACK}}`
- **Iteration / Baseline**: `{{ITERATION_ID}}`
- **Evaluated Artifacts Source**: `outputs/{{TRACK}}/{{SOURCE_EXP_ID}}/`
- **Timestamp**: {{TIMESTAMP}}

---

## 1. Executive Performance Fact Sheet

| Tolerance Window | Macro F1 | Recall | Precision | Status |
| :--- | :--- | :--- | :--- | :--- |
| **+/- 0.25s** | **{{F1_025}}%** | {{REC_025}}% | {{PREC_025}}% | {{ICON_025}} |
| **+/- 0.50s (Primary)** | **{{F1_050}}%** | {{REC_050}}% | {{PREC_050}}% | {{ICON_050}} |
| **+/- 1.00s** | **{{F1_100}}%** | {{REC_100}}% | {{PREC_100}}% | {{ICON_100}} |

---

## 2. Training Log Health & Dynamics Analysis

- **Recorded Epochs**: {{RECORDED_EPOCHS}}
- **Loss Trajectory**: Initial `{{INIT_LOSS}}` -> Final `{{FINAL_LOSS}}` (Minimum: `{{MIN_LOSS}}`)
- **Peak Validation F1**: `{{PEAK_F1}}%`

### Detected Training Pathologies & Anomalies:
- {{ANOMALY_1}}
- {{ANOMALY_2}}

---

## 3. Top Severe Error Cases (Visual Evidence)

Prioritizing the top 3-4 most severe failure modes with extracted video frames:

### Case 1: `{{CASE_ID_1}}` ({{CATEGORY_BADGE_1}})
- **Video ID**: `{{VIDEO_ID_1}}` | **Timestamp**: `{{TIMESTAMP_1}}s`
- **Discrepancy**: {{DISCREPANCY_1}}
- **Factual Observation**: {{OBSERVATION_1}}
- **Visual Frame Strip**: [View Strip Image](file://{{STRIP_PATH_1}})
  ![Visual Frame Strip]({{STRIP_PATH_1}})

### Case 2: `{{CASE_ID_2}}` ({{CATEGORY_BADGE_2}})
- **Video ID**: `{{VIDEO_ID_2}}` | **Timestamp**: `{{TIMESTAMP_2}}s`
- **Discrepancy**: {{DISCREPANCY_2}}
- **Factual Observation**: {{OBSERVATION_2}}
- **Visual Frame Strip**: [View Strip Image](file://{{STRIP_PATH_2}})
  ![Visual Frame Strip]({{STRIP_PATH_2}})

---

## 4. Per-Video Breakdown Ranking (Worst to Best)

| Rank | Video ID | F1 @ 0.5s | Recall | Precision | GT Count | Pred Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `{{WORST_VID_1}}` | **{{WORST_F1_1}}%** | {{WORST_REC_1}}% | {{WORST_PREC_1}}% | {{GT_1}} | {{PRED_1}} |
| 2 | `{{WORST_VID_2}}` | **{{WORST_F1_2}}%** | {{WORST_REC_2}}% | {{WORST_PREC_2}}% | {{GT_2}} | {{PRED_2}} |

---

## 5. What Does The Model Do Correctly?
- {{FACT_SUCCESS_1}}
- {{FACT_SUCCESS_2}}

---

> [!CAUTION]
> **Evaluator Rule**: No root causes or solutions are allowed in this document. Any speculative why or how to fix belongs in `02_diagnosis.md` and `03_research_plan.md`.
