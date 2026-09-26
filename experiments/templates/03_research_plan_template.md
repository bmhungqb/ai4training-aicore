# Research & Experiment Plan: {{ITERATION_ID}}

- **Track**: `{{TRACK}}`
- **Input Diagnosis**: `02_diagnosis.md`
- **Target Hypotheses Addressed**: `{{HYPOTHESIS_IDS}}`
- **Timestamp**: {{TIMESTAMP}}

---

## 1. Literature & Technical Survey

| Solution Idea | Dimension (Data / Training / Arch / Inference) | References / Prior Art | Key Mechanism |
| :--- | :--- | :--- | :--- |
| Idea 1: {{IDEA_1}} | `[Data | Training | Arch | Inference]` | {{CITATION_OR_PAPER_LINK}} | {{HOW_IT_WORKS}} |
| Idea 2: {{IDEA_2}} | `[Data | Training | Arch | Inference]` | {{CITATION_OR_PAPER_LINK}} | {{HOW_IT_WORKS}} |

---

## 2. Proposed Solutions Breakdown

### Solution Candidate A: {{SOLUTION_A_NAME}}
- **Target Hypothesis**: Addresses `Hypothesis X: {{NAME}}`
- **Component Affected**: `{{SRC_FILE_OR_CONFIG_MODULE}}`
- **Pros / Cons**:
  - *Pros*: {{PROS}}
  - *Risks / Cons*: {{CONS}}

### Solution Candidate B: {{SOLUTION_B_NAME}}
- **Target Hypothesis**: Addresses `Hypothesis Y: {{NAME}}`
- **Component Affected**: `{{SRC_FILE_OR_CONFIG_MODULE}}`
- **Pros / Cons**:
  - *Pros*: {{PROS}}
  - *Risks / Cons*: {{CONS}}

---

## 3. Batch Experiment Plan (Candidates for Debate)

Proposed experiments to run concurrently or sequentially in this iteration:

### Experiment 1: `{{EXP_001_DIR_NAME}}`
- **Variable to Isolate**: {{EXACTLY_ONE_VARIABLE_CHANGED}}
- **Baseline to Compare Against**: `{{BASELINE_EXP_ID}}`
- **Implementation Changes**:
  - File: `{{TARGET_FILE}}`
  - Modification: {{EXPLICIT_CHANGE_DESCRIPTION}}
- **Expected Outcome**:
  - Primary metric expectation: {{METRIC_CHANGE_EXPECTED}}
  - Verification test: {{WHAT_TO_CHECK}}

### Experiment 2: `{{EXP_002_DIR_NAME}}`
- **Variable to Isolate**: {{EXACTLY_ONE_VARIABLE_CHANGED}}
- **Baseline to Compare Against**: `{{BASELINE_EXP_ID}}`
- **Implementation Changes**:
  - File: `{{TARGET_FILE}}`
  - Modification: {{EXPLICIT_CHANGE_DESCRIPTION}}
- **Expected Outcome**:
  - Primary metric expectation: {{METRIC_CHANGE_EXPECTED}}
  - Verification test: {{WHAT_TO_CHECK}}

---

## 4. Resource & Feasibility Estimation
- Estimated compute time per experiment: ~{{TIME_MINUTES}} mins.
- GPU / RAM requirements: {{HARDWARE_REQUIREMENTS}}.

---

## 5. Rebuttal & Plan Revision History (Debate Feedback Loop)

*Use this section when revising the plan based on critiques from `@research-debater` or Human feedback.*

### Revision Round: `{{REVISION_ROUND_ID}}` (e.g. Rev 2)
- **Critique Addressed**: {{SUMMARY_OF_DEBATE_CHALLENGE}}
- **Planner Defense / Concession**:
  - *Defense*: {{THEORETICAL_OR_EMPIRICAL_JUSTIFICATION_IF_DEFENDING}}
  - *Concession / Fix*: {{WHAT_WAS_CHANGED_IN_RESPONSE}}
- **Modified Experiments**:
  - Updated `{{MODIFIED_EXP_ID}}`: {{DESCRIPTION_OF_DELTA}}

