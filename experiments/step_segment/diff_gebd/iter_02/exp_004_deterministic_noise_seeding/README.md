# exp_004_deterministic_noise_seeding (NEW — Corrective Intervention)

Targets **Hypothesis 1: Stochastic DDIM Sampling Noise Dominates Small Effect Sizes** (Confidence: 70%).

Unlike Experiments 1-3 (diagnostic-only), this is a **fix**, not a measurement:
it removes the noise source itself rather than characterizing its size.
Added per Human's explicit direction and the Debate verdict's open policy
question, responding to the risk of the research loop spending excessive
cycles on measurement without ever testing a corrective intervention.

## Code-level grounding
`DIFFUSION.DDIM_SAMPLING_ETA: 0.0` already makes every individual
reverse-diffusion **step** deterministic (`sigma = eta * (...) = 0`, so the
`sigma * noise` term is always exactly zero). The **only** remaining
stochastic source is the single `x_time = torch.randn(shape, device=self.device)`
initial-noise draw at the start of `ddim_sample()` — drawn from whatever
state the global RNG happens to be in when that sample is reached, which
changes depending on `--val-overlap-seconds` chunk count.

## What changed (single, isolated code path)
1. `src/step_segment/DiffGEBD/modeling/diffusion_model.py::ddim_sample()`
   now accepts an optional `sample_seeds: list[int]` (length B). If provided,
   each batch row's initial noise is drawn from its **own** seeded
   `torch.Generator` instead of the shared global RNG stream. If `None`
   (default), behavior is **completely unchanged** from before this experiment.
2. `src/step_segment/DiffGEBD/train.py::validate_end_to_end()` computes a
   per-sample seed as `sha256(f"{video_id}::{chunk_index}")` and passes it
   through **only if** the new config flag `DIFFUSION.DETERMINISTIC_SAMPLE_SEEDING`
   is `True` (default `False` — legacy behavior is unaffected unless this
   flag is explicitly set).
3. `src/step_segment/DiffGEBD/modeling/config.py`: new config key
   `_C.DIFFUSION.DETERMINISTIC_SAMPLE_SEEDING = False`.

No change to model weights, thresholds, merge logic, or training — this is
purely an inference-time sampling determinism fix.

## What this does (verification runs)
On the **existing trained checkpoint** (`model_best.pth`, no retraining),
with `DIFFUSION.DETERMINISTIC_SAMPLE_SEEDING True` passed as a CLI override:

1. **Verification 1**: baseline (`val-overlap-seconds=0.0`) chunking, 2
   different **global** `--seed` values (`42`, `123`). Expectation: Macro F1
   should now be **identical or near-identical** (down to floating-point
   determinism limits), collapsing the previously-measured `0.0355`
   noise-floor spread to ~0.
2. **Verification 2 (spot-check)**: overlap=2.5 config, 1 seed. Confirms the
   per-sample noise draw for a given `video_id`+`chunk_index` pair is stable
   regardless of total chunk count / other chunks preceding it in the run.
3. Restores the unmodified val chunking on disk when done.

## Run

```bash
python experiments/step_segment/diff_gebd/iter_02/exp_004_deterministic_noise_seeding/run.py
```

3 sub-runs total (`verif1_baseline_seed42`, `verif1_baseline_seed123`,
`verif2_overlap_spotcheck`), each calling `torchrun ... train.py --test-only`
(GPU) + `tools/export_diffgebd_predictions.py` (CPU). Estimated total
runtime: ~20-30 min on a single GPU, ~6-8GB VRAM.

## Outputs
`outputs/step_segment/diff_gebd/iter_02/exp_004_deterministic_noise_seeding/`
- `verif1_baseline_seed42/`, `verif1_baseline_seed123/`: each with `predictions.json`, `metrics.json`, `model_pred_dict_ep-1.pkl`.
- `verif2_overlap_spotcheck/`: same structure.

## What to observe
1. Compare Macro F1@0.5s between `verif1_baseline_seed42` and
   `verif1_baseline_seed123`. If they are now identical/near-identical, this
   both **confirms** Hypothesis 1 (the prior 0.0355 spread was pure
   sampling-order artifact) and **fixes** it going forward.
2. If Macro F1 still differs meaningfully between the two seeds even with
   `DETERMINISTIC_SAMPLE_SEEDING=True`, this indicates either (a) a residual
   stochastic source not yet identified, or (b) a bug in the seeding
   implementation — do not conclude the fix succeeded without this check.
3. Inspect `verif2_overlap_spotcheck` qualitatively against the baseline runs
   for the same source videos, to sanity-check that per-sample seeding
   behaves consistently across different chunk counts.
