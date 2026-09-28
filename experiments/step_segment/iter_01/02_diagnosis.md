# Root-Cause Diagnosis: iter_01

- **Track**: `step_segment`
- **Input Report**: `01_eval_report.md` (`eval_report.json`)
- **Reference**: `experiments/step_segment/step_segment_overview.md` (no prior tested hypotheses — this is the first evaluated iteration for DiffGEBD)
- **Timestamp**: 2026-09-27 (post-evaluation)

---

## Failure Pattern Synthesis

Brief summary of the primary failure modes identified by Evaluator:

1. **Primary failure mode 1 — Missed boundaries clustered near clip start (~3s)**: Both severe False Negative cases (`cd5_chuyen3_FN_t3.05s`, `cd4_chuyen1_FN_t3.43s`) occur at ~3–3.4s into their respective clips, and both show continuous, visually gradual hand/fabric feeding motion rather than an abrupt visual cut. The two lowest-F1 videos in the per-video ranking (`cd4_chuyen1`: 21.1% F1, only 9 GT events; `cd10_chuyen1`: 33.3% F1, only 9 GT events) are also the shortest clips, reinforcing that early/short-context boundaries are disproportionately missed.
2. **Primary failure mode 2 — Spurious boundaries triggered by fine repetitive hand micro-motion**: Both severe False Positive cases (`cd12_chuyen1_FP_t94.29s`, `cd12_chuyen1_FP_t66.25s`) occur in the same video, `cd12_chuyen1`, which also has the worst precision in the entire set (22.5% precision, 71 predictions vs. only 52 GT — a 37% over-prediction rate). The 3-frame motion strips for both FP cases show near-identical, subtle continuous hand adjustments on fabric under the needle — no coarse task-level transition is visible — suggesting the model is firing on local visual jitter/self-similarity dips rather than true step semantics.

Overall: Macro F1 collapses sharply from 58.6% (±1.0s) → 40.8% (±0.5s) → 22.75% (±0.25s), indicating the model detects *approximate* boundary regions reasonably but has **poor temporal localization precision** — consistent with both failure modes above (imprecise, drifting boundary placement rather than complete absence of signal).

---

## Root-Cause Hypotheses (2–3 Hypotheses)

### Hypothesis 1: Chunk-Boundary Temporal Context Starvation

- **Hypothesis Statement**: The model is trained/inferred on fixed 5-second (75-frame) chunks (`SEQUENCE_LENGTH: 75`, config `sewing_diffgebd_resnet50_chunk5s`). Boundaries that fall near the start of a chunk (e.g., ~3s into a clip) have insufficient "before" temporal context within the diffusion decoder's window (`WINDOW_SIZE: 17`) to compute a reliable dissimilarity signal, causing the model to systematically under-detect transitions occurring in the first ~half of each chunk.
- **Domain**: `Architecture representation capacity / Temporal receptive field`
- **Confidence Level**: `Medium` (55%)
- **Supporting Evidence**:
  - *Training Dynamics Evidence (Pillar 1)*: N/A — no epoch-level loss/F1 trend directly isolates chunk-position error, this is inferred from evaluation-time evidence only.
  - *Visual Frame Evidence (Pillar 2)*: Both FN severe cases are located at 3.05s and 3.43s — both well within the first chunk of their respective clips and both showing continuous ongoing hand motion (no sharp visual cut) at the exact GT timestamp, consistent with a boundary that requires more surrounding context to disambiguate from within-step continuous motion.
- **Contradicting Evidence**:
  - Contradiction A: The two shortest videos (`cd4_chuyen1`, `cd10_chuyen1` — only 9 GT boundaries each) also have the worst F1 scores overall (21.1%, 33.3%), which could equally be explained by small-sample variance rather than a chunk-position effect specifically. We do not have a breakdown of error rate as a function of "distance from chunk start" across the full dataset — only 2 anecdotal cases.
- **What Evidence Is Still Missing**:
  - Missing probe: Bucket all False Negative timestamps by their relative position within their assigned inference chunk (e.g., first 1s vs. last 1s of a 5s chunk) across the full `predictions.json` / GT set, and compare recall rates. If recall is materially worse in the first ~1–1.5s of each chunk, this hypothesis is confirmed.

---

### Hypothesis 2: Feature Representation Over-Sensitive to Fine Local Motion (Granularity Mismatch)

- **Hypothesis Statement**: The GEBD backbone's dissimilarity/diffusion-based boundary scoring reacts to any localized frame-to-frame visual change (e.g., subtle micro-adjustments of fabric/hand position during a single sewing sub-action), rather than being selective for the coarser, semantically-defined step-transition boundaries the ground truth captures — causing spurious over-triggering in continuously fine-motor-active segments.
- **Domain**: `Feature Representation`
- **Confidence Level**: `High` (65%)
- **Supporting Evidence**:
  - *Training Dynamics Evidence (Pillar 1)*: Training log (`train.log`) shows a low ceiling for `Rel@0.05 F1` (peaking around 0.29 at epoch 13–14) with recall (~0.44) consistently higher than precision (~0.22), indicating the model was already biased toward over-predicting boundaries relative to a tight/relative tolerance window during training — i.e., low precision is a trained-in property, not just a test-time artifact.
  - *Visual Frame Evidence (Pillar 2)*: Both FP strips (`cd12_chuyen1` at 66.25s and 94.29s) show visually near-identical, continuous fine hand manipulation across all 3 frames (0.35s apart) with no coarse task-level change — yet the model fired a boundary. `cd12_chuyen1` is also the video with the single worst precision (22.5%) and highest absolute over-prediction (71 predicted vs. 52 GT, +19 spurious).
- **Contradicting Evidence**:
  - Contradiction A: Not all high-GT-density videos show this same precision collapse — `cd8_chuyen2` (121 GT, 126 pred) achieves much closer alignment (38.9% precision) despite a similarly dense boundary schedule, suggesting the effect may be specific to certain visual/lighting/occlusion conditions in `cd12_chuyen1` rather than a universal architectural flaw.
- **What Evidence Is Still Missing**:
  - Missing probe: Inspect the raw per-frame dissimilarity/diffusion score curve (`score_curve.png`, if available) around the two FP timestamps to confirm whether score peaks correlate with hand-motion magnitude (e.g., optical flow energy) rather than any semantic step change. Comparing score-curve "peakiness" between `cd12_chuyen1` (worst precision) and `cd12_chuyen2` (best precision, 60%) — same camera/station, different clips — would isolate whether the issue is content-specific (e.g., worker's motion style) or systemic to the model.

---

### Hypothesis 3: Ground-Truth Boundary Definition Ambiguity for Subtle Sub-Step Transitions

- **Hypothesis Statement**: The annotated boundaries near clip starts (e.g., cd5_chuyen3 @3.05s, cd4_chuyen1 @3.43s) mark a transition embedded within a visually continuous fabric-feeding action, where the true "step change" cue (e.g., a new stitch line beginning) is subtle and possibly under-specified by a single annotator (`ANNOTATOR_SELECT: best`, `ANNOTATORS: 1`), making this a hard-to-learn, potentially noisy label rather than a genuine model capacity failure.
- **Domain**: `Data Quality / Label Noise / Ground Truth Ambiguity`
- **Confidence Level**: `Low` (35%)
- **Supporting Evidence**:
  - *Training Dynamics Evidence (Pillar 1)*: Config confirms only a single annotator's boundary was used for training/eval (`ANNOTATORS: 1`, `ANNOTATOR_SELECT: best`), with no multi-annotator agreement/consensus signal to gauge boundary sharpness or inter-rater consistency.
  - *Visual Frame Evidence (Pillar 2)*: In both FN strips, the 3-frame sequence (t-0.35s, t, t+0.35s) shows the worker performing what appears to be the *same* continuous fabric-guiding action with no obvious visual discontinuity at the labeled timestamp — consistent with a boundary that is temporally "soft" rather than a hard cut.
- **Contradicting Evidence**:
  - Contradiction A: The `GAUSSIAN_TARGET`/`ADAPTIVE_GAUS_SIGMA` settings in the config already imply the training pipeline soft-labels boundaries with a Gaussian kernel around the GT timestamp specifically to accommodate such ambiguity — so some tolerance for imprecise visual cues is already built in, weakening the claim that label sharpness alone explains the miss.
- **What Evidence Is Still Missing**:
  - Missing probe: If multi-annotator labels exist for any subset of clips, compute inter-annotator boundary-time variance near these specific timestamps (~3s marks) to quantify whether this is a genuinely high-disagreement/ambiguous region compared to boundaries the model correctly detects elsewhere.

---

> [!CAUTION]
> **Diagnoser Rule**: Diagnoser isolates root cause hypotheses and missing evidence. It does not propose concrete code solutions or experiment implementations. That responsibility belongs to `03_research_plan.md`.
