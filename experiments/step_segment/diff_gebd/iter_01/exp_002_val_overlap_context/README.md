# exp_002_val_overlap_context

Targets **Hypothesis 1: Chunk-Boundary Temporal Context Starvation** (Confidence: 55%).
**Prerequisite**: `exp_000_training_health_audit` and `exp_000b_aggregate_error_histograms` must both PASS/confirm before trusting this experiment's results.

## What this does
On the **existing trained checkpoint** (`model_best.pth`, no retraining):

1. **Step 0 — Seed-variance noise-floor control**: re-runs inference on the
   **unmodified** (`val-overlap-seconds=0.0`) val chunks 3x with different
   seeds (`42, 123, 2024`) to establish how much Macro F1@0.5s swings from
   pure DDIM/CFG sampling stochasticity alone (`DIFFUSION.DETERMINISTIC: false`).
2. **Step 0.5 — Shifted single-window control**: re-chunks val with a single
   non-overlapping grid shifted `+2.0s`, so GT boundaries that used to sit
   near a chunk edge now sit mid-chunk — isolates "more context" from
   "multi-view averaging" (no `merge_close_boundaries` load-bearing here).
3. **Main run**: re-chunks val with `--val-overlap-seconds 2.5` (overlapping
   chunks) and re-runs inference — the original proposed intervention
   (entangled context + multi-view-averaging effect).
4. Restores the unmodified val chunking on disk when done.

**No retraining** — pure inference-side data re-slicing + re-inference on the
same checkpoint. Single-GPU only.

## Run

```bash
python experiments/step_segment/diff_gebd/iter_01/exp_002_val_overlap_context/run.py
```

Each of the 5 sub-runs (`3x seed`, `shifted-window`, `main-overlap`) calls
`torchrun ... train.py --test-only` (GPU) + `tools/export_diffgebd_predictions.py`
(CPU). Estimated total runtime: ~45-65 min on a single GPU.

## Outputs
`outputs/step_segment/diff_gebd/iter_01/exp_002_val_overlap_context/`
- `step0_noise_floor_seed42/`, `step0_noise_floor_seed123/`, `step0_noise_floor_seed2024/`: each with `predictions.json`, `metrics.json`, `model_pred_dict_ep-1.pkl`.
- `step0_5_shifted_window/`: same structure.
- `main_overlap_run/`: same structure.

## What to observe
1. **Noise floor**: compute Macro F1@0.5s spread across the 3 seed runs. Any
   claimed improvement from the main/shifted run below must **exceed** this
   spread to be considered real (not sampling noise).
2. **Merge-logic sanity check (manual, required before trusting results)**:
   inspect 3-5 chunk-overlap regions' raw (pre-merge) predicted boundary
   lists in `main_overlap_run/<video>/boundaries.json` to confirm
   `merge_close_boundaries` (`merge-eps=0.35`) is deduplicating as intended.
3. **Severe FN cases** (`cd5_chuyen3` @3.05s, `cd4_chuyen1` @3.43s, or
   equivalents surfaced by `exp_000b`): check whether they disappear or
   shrink in discrepancy distance in BOTH `main_overlap_run` and
   `step0_5_shifted_window`. If only `main_overlap_run` improves and
   `step0_5_shifted_window` doesn't, the effect is likely multi-view
   averaging, not context restoration.
4. Target: Macro F1@0.5s +3-6 pts in `main_overlap_run` vs. baseline,
   exceeding the noise floor, without precision collapsing.
