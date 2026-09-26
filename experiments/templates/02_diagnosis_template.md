# Root-Cause Diagnosis: {{ITERATION_ID}}

- **Track**: `{{TRACK}}`
- **Input Report**: `01_eval_report.md` (`eval_report.json`)
- **Reference**: `TRACKER.md` (past iterations & tested hypotheses)
- **Timestamp**: {{TIMESTAMP}}

---

## Failure Pattern Synthesis

Brief summary of the primary failure modes identified by Evaluator:
1. **Primary failure mode 1**: {{SUMMARY_MODE_1}}
2. **Primary failure mode 2**: {{SUMMARY_MODE_2}}

---

## Root-Cause Hypotheses (2 - 3 Hypotheses)

### Hypothesis 1: {{HYPOTHESIS_1_TITLE}}

- **Hypothesis Statement**: {{EXPLICIT_ROOT_CAUSE_CLAIM}}
- **Domain**: `[Data Quality / Class Imbalance / Model Capacity / Optimization / Label Noise / Domain Shift / Feature Representation]`
- **Confidence Level**: `[High | Medium | Low]` ({{CONFIDENCE_SCORE}}%)
- **Supporting Evidence**:
  - Evidence A: {{SUPPORTING_EVIDENCE_A}} (Ref: `01_eval_report.md#section`)
  - Evidence B: {{SUPPORTING_EVIDENCE_B}}
- **Contradicting Evidence**:
  - Contradiction A: {{CONTRADICTING_EVIDENCE_A}} (or "None observed")
- **What Evidence Is Still Missing**:
  - Missing probe: {{WHAT_WOULD_PROVE_OR_DISPROVE_THIS}}

---

### Hypothesis 2: {{HYPOTHESIS_2_TITLE}}

- **Hypothesis Statement**: {{EXPLICIT_ROOT_CAUSE_CLAIM}}
- **Domain**: `[Data Quality / Class Imbalance / Model Capacity / Optimization / Label Noise / Domain Shift / Feature Representation]`
- **Confidence Level**: `[High | Medium | Low]` ({{CONFIDENCE_SCORE}}%)
- **Supporting Evidence**:
  - Evidence A: {{SUPPORTING_EVIDENCE_A}}
- **Contradicting Evidence**:
  - Contradiction A: {{CONTRADICTING_EVIDENCE_A}}
- **What Evidence Is Still Missing**:
  - Missing probe: {{WHAT_WOULD_PROVE_OR_DISPROVE_THIS}}

---

### Hypothesis 3: {{HYPOTHESIS_3_TITLE}} (Optional)

- **Hypothesis Statement**: {{EXPLICIT_ROOT_CAUSE_CLAIM}}
- **Domain**: `[Data Quality / Class Imbalance / Model Capacity / Optimization / Label Noise / Domain Shift / Feature Representation]`
- **Confidence Level**: `[High | Medium | Low]` ({{CONFIDENCE_SCORE}}%)
- **Supporting Evidence**:
  - Evidence A: {{SUPPORTING_EVIDENCE_A}}
- **Contradicting Evidence**:
  - Contradiction A: {{CONTRADICTING_EVIDENCE_A}}
- **What Evidence Is Still Missing**:
  - Missing probe: {{WHAT_WOULD_PROVE_OR_DISPROVE_THIS}}

---

> [!CAUTION]
> **Diagnoser Rule**: Diagnoser isolates root cause hypotheses and missing evidence. It does not propose concrete code solutions or experiment implementations. That responsibility belongs to `03_research_plan.md`.
