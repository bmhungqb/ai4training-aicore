---
trigger: model_decision
description: Enforces the Human-in-the-loop AI Research Experiment Loop protocol across the repository.
---

# Human-in-the-Loop AI Research Loop Protocol

When working on machine learning experiments, boundary detection, model tuning, or evaluation in this repository, strictly adhere to the **AI Research Loop**:

```
[Experiment / Run] ──► [Evaluator] ──► [Diagnoser] ──► [Researcher] ──► [Debate] ──► [Human Sign-off] ──► [Implementor] ──► [Human Runs]
```

## Directory Structure Standards

```
experiments/
├── <track>/                     # 'action_segment' or 'step_segment'
│   ├── TRACKER.md               # Single Source of Truth for the track
│   └── iter_XX/                 # One folder per iteration cycle
│       ├── 01_eval_report.md    # Evaluator: facts, errors, metrics
│       ├── eval_report.json     # Evaluator: structured data
│       ├── 02_diagnosis.md      # Diagnoser: 2-3 root-cause hypotheses
│       ├── 03_research_plan.md  # Researcher: solutions & isolated batch exp plan
│       ├── 04_debate_verdict.md # Debate: challenges & Human sign-off
│       ├── batch_eval_summary.md# Evaluator: comparative batch evaluation
│       ├── run_all.sh           # Batch runner for Human
│       ├── exp_001_<name>/      # Experiment 1 in batch
│       │   ├── run.py
│       │   ├── config.yaml
│       │   └── README.md
│       └── exp_002_<name>/      # Experiment 2 in batch
└── templates/                   # Standardized schemas and templates

outputs/                         # [GIT-IGNORED] Heavy artifacts
└── <track>/
    └── iter_XX/
        ├── exp_001_<name>/      # Checkpoints, predictions, raw logs
        └── exp_002_<name>/
```

## The 6 Immutable Invariants

1. **Role Separation**:
   - `Evaluator` states **what** happened (evidence only). NEVER root causes, NEVER solutions.
   - `Diagnoser` states **why** it happened (2-3 hypotheses with confidence & evidence). NEVER code solutions.
   - `Researcher` states **what to try** (survey literature, isolate variables, propose batch).
   - `Debater` **challenges** the plan as Devil's Advocate.
   - `Human` **decides and approves**.
   - `Implementor` **writes code and configs**.
   - `Human` **runs the experiment**.
2. **Two Mandatory Human Checkpoints**:
   - Checkpoint 1 (Debate & Approval): Agent must not write experiment code without explicit Human sign-off in `04_debate_verdict.md`.
   - Checkpoint 2 (Execution): Agent must never run training/heavy compute commands. It provides the exact bash commands for the Human to execute.
3. **Single Source of Truth**:
   - Keep `experiments/<track>/TRACKER.md` updated after every iteration batch finishes.
4. **Variable Isolation**:
   - Each sub-experiment (`exp_00X`) must isolate exactly one variable against the baseline control.
5. **Context Protection**:
   - Heavy outputs (weights, 100k-line logs, video cuts) stay in `outputs/`. Only lightweight structured reports live in `experiments/`.
6. **Skills Invocation**:
   - Use the dedicated skills: `research-evaluator`, `research-diagnoser`, `research-planner`, `research-debater`, `research-implementor`.
