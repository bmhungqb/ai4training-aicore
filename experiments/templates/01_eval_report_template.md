# Structured Evaluation Report: {{ITERATION_ID}}

- **Track**: `{{TRACK}}`
- **Iteration / Baseline**: `{{ITERATION_ID}}`
- **Evaluated Artifacts Source**: `outputs/{{TRACK}}/{{SOURCE_EXP_ID}}/`
- **Timestamp**: {{TIMESTAMP}}

---

## 1. Executive Fact Sheet (Metrics Only)

| Metric | Target / Baseline | Observed Value | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Metric (e.g. Macro Recall)** | {{BASELINE_VAL}}% | **{{OBSERVED_VAL}}%** | {{DELTA}}% | {{STATUS_ICON}} |
| **Secondary Metric (e.g. Macro F1)** | {{BASELINE_VAL}}% | **{{OBSERVED_VAL}}%** | {{DELTA}}% | {{STATUS_ICON}} |
| **Micro Metric** | {{BASELINE_VAL}}% | **{{OBSERVED_VAL}}%** | {{DELTA}}% | {{STATUS_ICON}} |

---

## 2. Where Does The Model Fail? (Per-Category / Per-Class Breakdown)

| Category / Operation ID | Name | Sample Count | Metric Observed | Comparison vs Average |
| :--- | :--- | :--- | :--- | :--- |
| `case_01` | {{NAME_1}} | {{COUNT_1}} | {{METRIC_1}} | {{DIFF_1}} |
| `case_02` | {{NAME_2}} | {{COUNT_2}} | {{METRIC_2}} | {{DIFF_2}} |

---

## 3. How Does The Model Fail? (Error Categorization)

### False Positives (Over-segmentation / Spurious Triggers)
- **Count**: {{FP_COUNT}}
- **Typical occurrence pattern**: {{FP_DESCRIPTION_FACTUAL_ONLY}}
- **Evidence**: `{{FP_SAMPLE_PATH_OR_LOG_LINE}}`

### False Negatives (Missed Boundaries / Actions)
- **Count**: {{FN_COUNT}}
- **Typical occurrence pattern**: {{FN_DESCRIPTION_FACTUAL_ONLY}}
- **Evidence**: `{{FN_SAMPLE_PATH_OR_LOG_LINE}}`

---

## 4. What Does The Model Do Correctly?
- {{FACT_SUCCESS_1}}: supported by logs `{{LOG_REF_1}}`.
- {{FACT_SUCCESS_2}}: supported by metrics `{{METRIC_REF_2}}`.

---

> [!CAUTION]
> **Evaluator Rule**: No root causes or solutions are allowed in this document. Any speculative why or how to fix belongs in `02_diagnosis.md` and `03_research_plan.md`.
