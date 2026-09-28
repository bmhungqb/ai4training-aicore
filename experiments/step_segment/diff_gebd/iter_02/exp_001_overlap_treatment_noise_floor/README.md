# exp_001_overlap_treatment_noise_floor

Targets **Hypothesis 1: Stochastic DDIM Sampling Noise Dominates Small Effect Sizes** (Confidence: 70%).

> [!IMPORTANT]
> **Execution order**: Run this **AFTER** `exp_002_premerge_boundary_audit` (CPU-only,
> zero-cost) per the Debate verdict's stated execution order — Exp 2's result
> is a precondition for correctly interpreting this experiment's noise-band
> comparison.

## What this does
On the **existing trained checkpoint** (`model_best.pth`, no retraining), with
`--val-overlap-seconds` **fixed at 2.5** (the treatment value):

1. Copies the existing seed=42 `main_overlap_run` result (Macro F1=0.2975,
   already on disk at `outputs/step_segment/diff_gebd/iter_01/exp_002_val_overlap_context/main_overlap_run/`)
   into this experiment's output directory as the 1st data point — **not
   re-run**.
2. Runs 2 additional seeds (`123`, `2024`) against the same val-overlap=2.5
   chunking, reusing `regenerate_val_chunks()` / `infer_and_export()` imported
   directly from `experiments/step_segment/diff_gebd/iter_01/exp_002_val_overlap_context/run.py`
   (no modification to that logic).
3. Restores the unmodified (`val-overlap-seconds=0.0`) val chunking on disk
   when done.

**Revision 2 correction**: this experiment does **not** claim the numbered
seeds (123, 2024) are paired/matched against the same-numbered baseline
(overlap=0.0) noise-floor seeds — the two 3-seed samples are treated as
**independent, unpaired** distributions (see `03_research_plan.md` §Exp 1
Implementation Changes for the code-level justification).

## Run

```bash
python experiments/step_segment/diff_gebd/iter_02/exp_001_overlap_treatment_noise_floor/run.py
```

2 sub-runs (`overlap_seed123`, `overlap_seed2024`), each calling
`torchrun ... train.py --test-only` (GPU) + `tools/export_diffgebd_predictions.py`
(CPU). Estimated total runtime: ~15-25 min on a single GPU, ~6-8GB VRAM.

## Outputs
`outputs/step_segment/diff_gebd/iter_02/exp_001_overlap_treatment_noise_floor/`
- `overlap_seed42_reference_copy/`: copy of the existing seed=42 result (not re-run).
- `overlap_seed123/`, `overlap_seed2024/`: each with `predictions.json`, `metrics.json`, `model_pred_dict_ep-1.pkl`.

## What to observe
1. Compute the 3-seed mean, min, max, range of Macro F1@0.5s across
   `overlap_seed42_reference_copy`, `overlap_seed123`, `overlap_seed2024`
   (the overlap-treatment noise band).
2. Compare against the already-established baseline (overlap=0.0) 3-seed
   noise band `[0.3435, 0.3790]` from `outputs/step_segment/diff_gebd/iter_01/exp_002_val_overlap_context/step0_noise_floor_seed*/metrics.json`.
3. Report explicitly whether the two bands overlap:
   - If they do NOT overlap (all 3 overlap-seed runs score below ~0.34):
     evidence of a systematic (non-noise) degradation from the overlap
     intervention — cross-reference with `exp_002_premerge_boundary_audit`'s
     result before concluding Hypothesis 2 (merge interaction) is the cause.
   - If they DO overlap: Hypothesis 1 (pure sampling noise) is favored as the
     primary explanation for the original single-sample 0.2975 result.
