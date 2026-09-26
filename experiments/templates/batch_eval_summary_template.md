# Batch Evaluation Summary: {{ITERATION_ID}}

- **Track**: `{{TRACK}}`
- **Iteration**: `{{ITERATION_ID}}`
- **Baseline Compared**: `{{BASELINE_EXP_ID}}`
- **Timestamp**: {{TIMESTAMP}}

---

## 1. Batch Comparative Metrics Table

| Experiment | Isolated Variable | Primary Metric (Delta) | Secondary Metric (Delta) | Key Trade-off | Verdict |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Baseline** | Default setup | {{BASE_PRIMARY}}% | {{BASE_SEC}}% | - | Benchmark |
| `exp_001` | {{VAR_1}} | {{EXP1_PRIMARY}}% ({{DELTA_1}}%) | {{EXP1_SEC}}% ({{DELTA_1_SEC}}%) | {{TRADEOFF_1}} | `[Improved / Regressed / Inconclusive]` |
| `exp_002` | {{VAR_2}} | {{EXP2_PRIMARY}}% ({{DELTA_2}}%) | {{EXP2_SEC}}% ({{DELTA_2_SEC}}%) | {{TRADEOFF_2}} | `[Improved / Regressed / Inconclusive]` |

---

## 2. Hypothesis Validation Verdicts

- **Hypothesis 1 (`{{H1_ID}}`)**: `[Validated | Refuted | Inconclusive]`
  - *Evidence*: {{FACTUAL_EVIDENCE}}
- **Hypothesis 2 (`{{H2_ID}}`)**: `[Validated | Refuted | Inconclusive]`
  - *Evidence*: {{FACTUAL_EVIDENCE}}

---

## 3. Promotion Decision
- **Promoted to New Baseline?**: `[YES (exp_XXX) | NO]`
- **Reason**: {{RATIONALE}}
- **TRACKER.md Status**: `[Updated]`
- **Next Loop Action**: `[Start Iteration XX+1 | Consolidate and deploy]`
