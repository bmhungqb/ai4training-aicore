---
name: research-loop
description: >-
  Master Orchestrator for the Human-in-the-loop AI Research Loop. Coordinates transitions
  between Evaluator, Diagnoser, Researcher, Debate, and Implementor skills, enforcing
  human review gates and tracker updates.
---

# Research Loop Orchestrator (`research-loop`)

The **Research Loop Orchestrator** manages the lifecycle of an iteration cycle in the AI Research Loop.

```
       ┌────────────────────────┐
       │   HUMAN RUNS BATCH     │
       └───────────┬────────────┘
                   │ outputs/<track>/<iter>/
                   ▼
       ┌────────────────────────┐
       │   research-evaluator   │ ──► 01_eval_report.md + eval_report.json
       └───────────┬────────────┘
                   ▼
       ┌────────────────────────┐
       │   research-diagnoser   │ ──► 02_diagnosis.md (2-3 hypotheses)
       └───────────┬────────────┘
                   ▼
       ┌────────────────────────┐
       │   research-planner     │ ──► 03_research_plan.md (batch exp proposals)
       └───────────┬────────────┘
                   ▼
       ┌────────────────────────┐
       │   research-debater     │ ◄──► [HUMAN DEBATE & SIGN-OFF]
       └───────────┬────────────┘
                   │ Approved
                   ▼
       ┌────────────────────────┐
       │  research-implementor  │ ──► exp_001/, exp_002/, run_all.sh
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   HUMAN RUNS BATCH     │
       └────────────────────────┘
```

## How to Run an Iteration Cycle

When the user asks to start a new iteration or process an existing run:

1. **Step 1: Evaluation**
   - Check if raw outputs exist in `outputs/<track>/<iter>/`.
   - Invoke `research-evaluator` to generate `01_eval_report.md` and `eval_report.json`.

2. **Step 2: Diagnosis**
   - Invoke `research-diagnoser` to analyze failure patterns and output `02_diagnosis.md`.

3. **Step 3: Planning**
   - Invoke `research-planner` to investigate prior art and draft `03_research_plan.md`.

4. **Step 4: Debate & Human Gate (STOP POINT 1 - Multi-Turn Feedback Loop)**
   - Invoke `research-debater` to present critical challenges.
   - Stop and interview the Human researcher.
   - **Feedback Sub-loop**: If critiques reveal flaws or Human requests adjustments, call `@research-planner` in Revision Mode to update `03_research_plan.md` (Revision 2, 3...) until aligned.
   - Once approved by Human: Record verdict in `04_debate_verdict.md` with status `APPROVED FOR IMPLEMENTATION`.
   - **Do NOT proceed to Step 5 until Human explicitly confirms approval.**

5. **Step 5: Implementation**
   - Invoke `research-implementor` to scaffold `exp_001/`, `exp_002/`, `run_all.sh`.

6. **Step 6: Human Execution (STOP POINT 2)**
   - Provide exact run commands to the Human.
   - Halt execution and wait for the Human to report that the runs have concluded.

7. **Step 7: Batch Wrap-up**
   - Once runs are done, run `research-evaluator` in batch mode to generate `batch_eval_summary.md` and update `TRACKER.md`.
