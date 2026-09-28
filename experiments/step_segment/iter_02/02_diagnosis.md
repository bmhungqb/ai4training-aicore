# Root-Cause Diagnosis: iter_02

- **Track**: `step_segment` (DiffGEBD)
- **Input Report**: `01_eval_report.md` (`eval_report.json`)
- **Reference**: `experiments/step_segment/iter_01/02_diagnosis.md` (prior hypotheses H1/H2/H3 — see "Relationship to Prior Hypotheses" below), `experiments/step_segment/step_segment_overview.md`
- **Timestamp**: 2026-09-28 (post-evaluation, iter_02)

---

## Failure Pattern Synthesis

Brief summary of the primary failure modes identified by the Evaluator in `01_eval_report.md`:

1. **Primary failure mode 1 — The two anecdotal iter_01 hypotheses (chunk-start FN clustering, sub-1s FP clustering) do not hold at dataset scale.** `exp_000b`'s dataset-wide histogram over all 264 FN and 262 FP cases (9 videos) found near-chunk-start FN fraction = 30.68% and sub-1s FP fraction = 31.68% — both statistically indistinguishable from the ~30% fraction expected under a uniform/null distribution. The n=2/n=2 anecdotal cases that drove iter_01's H1 (65%) and H2 (55%) confidence scores were not representative of the population.
2. **Primary failure mode 2 — The `exp_002` overlap-context treatment underperforms its own measurement noise floor, and its interventions (overlap + merge) coincide with a large drop in surviving predictions.** The 3-seed noise floor for the *unmodified* baseline alone spans Macro F1 0.3435–0.3790 (±3.55 pts) — already exceeding the ±0.31 pt gain later claimed by `exp_003`'s LOVO tuning. The actual `+2.5s` overlap treatment scored **0.2975**, below every single control run (3 seeds + shifted-window), with `num_pred` dropping from 373 (all controls) to 281 — a 25% reduction in surviving predictions after `merge_close_boundaries` is applied to the overlapping-chunk output.
3. **Secondary observation — Per-video (session-level) performance heterogeneity is large and consistent across every run variant.** `cd12_chuyen1` and `cd12_chuyen2` (same worker/station, different session) differ by 15–29 F1 points in every single control and treatment run examined (e.g. seed42: 0.3103 vs 0.4944; shifted-window: 0.2931 vs 0.5843; LOVO: 0.3091 vs 0.5176), independent of which post-processing/inference variant is applied.

---

## Root-Cause Hypotheses (2–3 Hypotheses)

### Hypothesis 1: Stochastic DDIM Sampling Noise Is Large Enough Relative to the Dataset Size to Dominate Small Reported Effect Sizes

- **Hypothesis Statement**: With `DIFFUSION.DETERMINISTIC: false` and only 9 val videos / 445 GT boundaries, run-to-run variance from DDIM's stochastic reverse-diffusion sampling alone produces a Macro F1 spread of ~3.5 points at fixed hyperparameters — a magnitude comparable to or larger than the effect sizes claimed by post-hoc interventions (`exp_003`'s +0.31 pt LOVO gain), meaning small "improvements" measured on this val set cannot currently be distinguished from sampling noise.
- **Domain**: `Optimization / Evaluation Reliability (measurement noise)`
- **Confidence Level**: `High` (70%)
- **Supporting Evidence**:
  - *Training Dynamics Evidence (Pillar 1)*: N/A — this is an inference-time (sampling), not training-time, noise source; no loss-curve evidence bears directly on this hypothesis.
  - *Visual Frame Evidence (Pillar 2)*: N/A — this hypothesis concerns aggregate metric variance, not localized visual failure cases.
  - *Quantitative Evidence (`01_eval_report.md` §2.2/§2.3)*: The 3-seed noise floor for the identical unmodified baseline config spans Macro F1 = 0.3435 (seed2024) to 0.3790 (seed42), a spread of 0.0355. `exp_003`'s reported LOVO improvement over the no-suppression baseline is only +0.0031 (0.3776 → 0.3807) — roughly 1/10th the magnitude of the seed-to-seed noise floor spread. Additionally, the originally-reported iter_01 single-run baseline (0.4077) sits *above* all 3 re-sampled noise-floor seeds, meaning the number used as "the" baseline in `01_eval_report.md`/iter_01 is itself an unrepresentative single draw rather than a stable point estimate.
- **Contradicting Evidence**:
  - Contradiction A: The `exp_002` main overlap treatment (0.2975) falls *below* the entire 3-seed noise floor range (0.3435–0.3790), a gap (~5–8 pts) larger than the noise floor spread itself — suggesting that specific result is not fully explained by sampling noise alone and likely reflects an additional systematic effect (see Hypothesis 2).
- **What Evidence Is Still Missing**:
  - Missing probe: Re-run the `exp_002` main overlap treatment (2.5s overlap) itself across 3+ different seeds to establish its own noise floor — currently only 1 sample of the overlap treatment exists, so it is unknown whether 0.2975 is a stable overlap-treatment value or itself an unlucky draw within a wider overlap-specific noise band.

---

### Hypothesis 2: Overlapping-Chunk Inference Triggers a Merge/Deduplication Behavior That Nets Fewer Surviving Predictions, Independent of Any "Added Context" Benefit

- **Hypothesis Statement**: Feeding overlapping (2.5s-overlap) chunks through inference produces multiple candidate predictions per physical timestamp near former chunk boundaries; `merge_close_boundaries` (eps=0.35s) then collapses or discards a larger share of these than in the non-overlapping baseline, reducing `num_pred` from 373 (baseline/shifted-window controls) to 281 — a 25% reduction that could suppress correct-but-duplicated detections along with genuinely redundant ones, independent of whether extra temporal context improved the underlying per-timestamp score signal.
- **Domain**: `Feature Representation / Post-processing interaction`
- **Confidence Level**: `Medium` (60%)
- **Supporting Evidence**:
  - *Training Dynamics Evidence (Pillar 1)*: N/A — this is a val-time post-processing interaction on a fixed checkpoint, not a training dynamic.
  - *Quantitative Evidence*: `main_overlap_run` recall collapses to 0.2427 (vs 0.3034–0.3483 across all 4 non-overlap-treatment runs) while precision is comparatively preserved (0.3843, within the 0.3941–0.4155 range of controls) — a pattern consistent with predictions being removed/merged away (hurting recall) rather than new low-confidence spurious predictions being added (which would hurt precision instead). The per-video table shows `cd12_chuyen1` and `cd12_chuyen2` recall both drop to the identical value 0.2885 under the overlap treatment — an unusual exact coincidence across two different videos with otherwise very different score distributions in every control run, and `cd12_chuyen2`'s `num_pred` falls to 21 (vs. 26–37 in every control run for the same video).
- **Contradicting Evidence**:
  - Contradiction A: `exp_000b`'s dataset-wide histogram already found that FN cases are *not* disproportionately concentrated near chunk-starts (30.68% vs ~30% expected), which was the original motivating rationale for adding overlap in the first place — so even if the merge-mechanism explanation in this hypothesis is correct, it does not by itself establish that overlap *should* have helped; the intervention's premise (H1 from iter_01) was already weakened before this run was executed.
- **What Evidence Is Still Missing**:
  - Missing probe: Inspect the raw (pre-merge) predicted boundary lists for the `main_overlap_run` at a handful of former-chunk-boundary timestamps to confirm whether `merge_close_boundaries` is discarding higher-confidence true-positive-adjacent candidates in favor of lower-confidence ones, or simply collapsing legitimately-duplicated detections down to 1 (which would be expected/benign behavior and would point back toward Hypothesis 1/pure noise as the explanation for the recall drop instead).

---

### Hypothesis 3: Session-Level Performance Heterogeneity (Not Chunk-Position or Local-Motion Artifacts) Is the Dominant Source of Variance the iter_01 Anecdotal Hypotheses Were Actually Picking Up On

- **Hypothesis Statement**: The large, consistent F1/recall/precision gap between `cd12_chuyen1` and `cd12_chuyen2` (same worker/station, different recording session) — present in every one of the 5 run variants examined in iter_02 regardless of post-processing method (15–29 F1 points apart every time) — indicates that a session-specific factor (e.g., camera angle, lighting, worker pacing/fatigue on that specific day) is a stronger driver of per-video score variance than either the chunk-boundary-context (iter_01 H1) or fine-local-motion (iter_01 H2) mechanisms originally hypothesized; the n=2 anecdotal FN/FP cases used to build those hypotheses in iter_01 were very likely samples of this broader session-level effect rather than independent evidence of a chunk-position or motion-granularity mechanism.
- **Domain**: `Overfitting to specific stations / sessions (Domain Shift within worker identity)`
- **Confidence Level**: `Medium` (55%)
- **Supporting Evidence**:
  - *Training Dynamics Evidence (Pillar 1)*: N/A — this is a validation-time, per-video effect; no training-log evidence directly measures per-session behavior.
  - *Quantitative Evidence*: Across seed42/seed123/seed2024/shifted-window/LOVO runs, `cd12_chuyen1` F1 ranges narrowly between 0.2881–0.3103 while `cd12_chuyen2` F1 ranges between 0.4944–0.5843 — the two videos never overlap in F1 across any of the 5 variants tested, and the gap (0.18–0.29) is consistently larger than either (a) the seed-to-seed noise floor spread (0.0355) or (b) any single intervention's claimed effect size (`exp_003`'s +0.0031). This indicates the session identity itself explains more variance than any post-processing change tested so far.
- **Contradicting Evidence**:
  - Contradiction A: This is consistent with, not contradictory to, the debate verdict's own prior observation (`04_debate_verdict.md` §1.1, "Overfitting Risk" row) that `cd12` sessions are split across train/val at the session level — that concern was raised before any of these 4 experiments ran, and this diagnosis's evidence corroborates rather than newly discovers it. It is listed here as a hypothesis (not merely a restatement) because iter_01's H1/H2 diagnosis attributed the `cd12_chuyen1` failures specifically to chunk-position and local-motion mechanisms, while the iter_02 evidence suggests a more general per-session factor may be the better-supported explanation for that video's poor scores specifically.
- **What Evidence Is Still Missing**:
  - Missing probe: A per-video/per-session breakdown of a session-identifiable confound (e.g., average optical-flow motion energy, mean frame brightness/contrast, or fraction of `cd12_chuyen3`-visually-similar frames in train) correlated against per-session F1, to determine which specific session attribute (rather than "session identity" as an unexplained catch-all) is driving the `cd12_chuyen1` vs `cd12_chuyen2` gap.

---

## Relationship to Prior Hypotheses (iter_01 → iter_02)

- **iter_01 H1 (Chunk-Boundary Temporal Context Starvation, 55% confidence)**: **Refuted at dataset scale** by `exp_000b` (30.68% near-chunk-start FN fraction ≈ 30% null expectation) and further undermined by `exp_002`'s main overlap treatment underperforming its own noise floor rather than improving on it. Not carried forward as a standalone hypothesis in iter_02.
- **iter_01 H2 (Feature Representation Over-Sensitive to Fine Local Motion, 65% confidence)**: **Refuted at dataset scale** by `exp_000b` (31.68% sub-1s FP fraction ≈ 30% null expectation). The specific video (`cd12_chuyen1`) that motivated H2 is re-examined in iter_02 Hypothesis 3 above, but reframed around session-level heterogeneity rather than a motion-granularity mechanism.
- **iter_01 H3 (Ground-Truth Boundary Definition Ambiguity, 35% confidence)**: Not addressed by any of the 4 iter_02 experiments (none inspected multi-annotator agreement or label sharpness); status unchanged, still an open, untested hypothesis from iter_01.

---

> [!CAUTION]
> **Diagnoser Rule**: Diagnoser isolates root cause hypotheses and missing evidence. It does not propose concrete code solutions or experiment implementations. That responsibility belongs to `03_research_plan.md`.

---

## Conclusion

Diagnosis complete with training dynamics and dataset-wide/per-video quantitative triangulation. Proceed to run `@research-planner` to investigate solutions and plan experiments — in particular, addressing the missing probes above: (1) establishing a noise floor specifically for the overlap-treatment condition, (2) inspecting pre-merge overlap predictions, and (3) identifying the concrete session-level confound behind the `cd12_chuyen1`/`cd12_chuyen2` gap.
