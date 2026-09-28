# exp_003_session_confound_profile (Revised, Rev 2)

Targets **Hypothesis 3: Session-Level Heterogeneity Dominates** (Confidence: 55%).

## What this does
Descriptive, **non-inferential** case comparison — no model inference, no
re-chunking, CPU-only (PIL + numpy). For each of the 9 val videos' **untouched
raw source frames** at `data/diff_gebd_dataset/images/val/<video_id>/frame*.jpg`
(corrected path per Debate verdict — NOT the non-existent
`outputs/.../<video_id>/` path, and NOT the stateful, repeatedly-regenerated
`<video_id>__chunk<i>/` symlink directories), computes over **all frames**
(no subsampling, per Debate verdict WARN correction):

1. **Motion-energy proxy**: mean absolute inter-frame pixel difference
   (grayscale, [0,1]).
2. **Brightness**: mean luminance across all frames.
3. **Contrast**: mean per-frame pixel standard deviation across all frames.

Joins these against each video's existing F1@0.5s from
`outputs/step_segment/diff_gebd/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/metrics.json`'s
`per_video_performance` field.

> [!IMPORTANT]
> **Statistical framing (Rev 2, per Debate verdict FAIL)**: This experiment
> does **NOT** compute or report a Pearson correlation coefficient — with
> n=9, any correlation coefficient would not be statistically meaningful.
> Results are reported as a **plain descriptive table** plus an explicit
> `cd12_chuyen1` vs `cd12_chuyen2` vs rest-of-dataset-range comparison,
> labeled throughout as **hypothesis-generating / non-inferential**.

## Run

```bash
python experiments/step_segment/diff_gebd/iter_02/exp_003_session_confound_profile/run.py
```

CPU-only. Estimated runtime: ~15-20 min for 9 videos (all frames, no subsampling).

## Outputs
`outputs/step_segment/diff_gebd/iter_02/exp_003_session_confound_profile/`
- `video_stats_profile.json` / `.csv`: `video_id, num_frames, motion_energy, brightness, contrast, f1_at_0_5s` for all 9 videos.
- `named_pair_comparison.json`: `cd12_chuyen1` vs `cd12_chuyen2` vs rest-of-dataset range for each of the 3 stats.

## What to observe
- Report the 3 stats specifically for `cd12_chuyen1` vs `cd12_chuyen2`
  alongside the full 9-video table.
- If `cd12_chuyen1` sits clearly outside the range spanned by the other 8
  videos on any one of the 3 stats, that stat becomes a concrete, named
  candidate confound for future (separately-designed, adequately-powered)
  investigation — **explicitly not proof of causation**.
- If no stat stands out for `cd12_chuyen1`, this weakens Hypothesis 3's
  currently-proposed proxies and flags that the true confound (if any)
  remains unidentified.
