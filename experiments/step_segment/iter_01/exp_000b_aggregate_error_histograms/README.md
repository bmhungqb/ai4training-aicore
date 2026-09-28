# exp_000b_aggregate_error_histograms

**Status**: BLOCKING GATE — must be run before funding `exp_002_val_overlap_context` / `exp_003_min_peak_distance_suppression`.

## What this does
Reprocesses the existing baseline `predictions.json` against GT
(`step_segments.json`) across **all 9 val videos** (not just the 2 anecdotal
FN / 2 anecdotal FP cases in `02_diagnosis.md`):
- **FN histogram**: bucket every false-negative GT boundary by its position
  relative to its nearest val chunk start (near-start = first 1.5s of a 5s chunk).
- **FP histogram**: distance from every false-positive prediction to its
  nearest other prediction in the same video.

No GPU, no re-inference — pure re-processing of existing JSON/pkl. Runtime: ~5 min.

## Run

```bash
python experiments/step_segment/iter_01/exp_000b_aggregate_error_histograms/run.py
```

## Outputs
- `outputs/step_segment/iter_01/exp_000b_aggregate_error_histograms/error_distribution_report.json`
- `outputs/step_segment/iter_01/exp_000b_aggregate_error_histograms/error_histograms.png`

## What to observe
- `hypothesis_1_generalizes` (bool): must be `true` to proceed with `exp_002_val_overlap_context`.
- `hypothesis_2_generalizes` (bool): must be `true` to proceed with `exp_003_min_peak_distance_suppression`.
- `verdict.exp_002_val_overlap_context` / `verdict.exp_003_min_peak_distance_suppression`: `PROCEED` or `DE-PRIORITIZE (...)`.
