# exp_002_premerge_boundary_audit

Targets **Hypothesis 2: Overlap+Merge Interaction Nets Fewer Surviving Predictions** (Confidence: 60%).

> [!IMPORTANT]
> **Execution order**: Run this **FIRST** (before `exp_001_overlap_treatment_noise_floor`)
> per the Debate verdict — CPU-only, zero-cost, zero-risk, and its result is a
> precondition for correctly interpreting Exp 1's noise-band comparison.

## What this does
Pure read-only, zero-inference audit of the **already-saved**
`model_pred_dict_ep-1.pkl` from
`outputs/step_segment/diff_gebd/iter_01/exp_002_val_overlap_context/main_overlap_run/`.
No re-run, no GPU. Uses a new tool, `tools/audit_diffgebd_premerge_candidates.py`,
which reconstructs the pre-merge `boundaries_by_source` dict (same remapping
logic as `tools/export_diffgebd_predictions.py::scores_to_preds_json`, stopping
just before `merge_close_boundaries()`), then reports **dataset-wide across
all 9 val videos**:

1. Total pre-merge candidate count vs. total post-merge candidate count
   (aggregate reduction %).
2. For each merged cluster with >1 pre-merge candidate: whether the
   retained/averaged timestamp is within ±0.5s of a GT boundary (legitimately-
   duplicated true-positive-adjacent detection) vs. not (spurious candidate
   averaged into a wrong location, or a real candidate merged away from its
   correct GT-adjacent timestamp).
3. Per-video breakdown, with `cd12_chuyen1` / `cd12_chuyen2` always reported
   individually (standing per-worker verification requirement from
   `04_debate_verdict.md`, iter_01).

## Run

```bash
python experiments/step_segment/diff_gebd/iter_02/exp_002_premerge_boundary_audit/run.py
```

CPU-only. Estimated runtime: <5 min.

## Outputs
`outputs/step_segment/diff_gebd/iter_02/exp_002_premerge_boundary_audit/premerge_audit_report.json`
- `dataset_wide_summary`: total pre/post-merge counts, aggregate reduction %,
  and gt-adjacent vs. spurious multi-candidate cluster counts.
- `worker_pair_breakdown`: `cd12_chuyen1` / `cd12_chuyen2` specific stats.
- `per_video`: full per-video cluster details.

## What to observe
- If a **majority** of the dataset-wide multi-candidate clusters are
  classified `gt_adjacent_legit_duplicate` (i.e., merge is discarding
  higher-confidence TP-adjacent candidates), **Hypothesis 2 is supported**.
- If most collapsed clusters are classified `spurious_or_merged_away_from_gt`
  (benign, expected merge behavior), **Hypothesis 2 is weakened** — the
  recall drop is more likely attributable to Hypothesis 1 (noise, see
  `exp_001`) or a genuine but currently unexplained overlap-context effect.
- Review the `worker_pair_breakdown` for `cd12_chuyen1` vs `cd12_chuyen2`
  before drawing any dataset-wide conclusion.
