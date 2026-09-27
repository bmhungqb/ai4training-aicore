# Root-Cause Diagnosis: iter_01

- **Track**: `step_segment`
- **Model**: `DiffGEBD` (`sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75`)
- **Input Report**: [`01_eval_report.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/step_segment/iter_01/01_eval_report.md) ([`eval_report.json`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/step_segment/iter_01/eval_report.json))
- **Reference**: [`step_segment_overview.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/step_segment/step_segment_overview.md)
- **Timestamp**: 2026-09-27T23:40:00+07:00

---

## Failure Pattern Synthesis

Based on the evidence presented in [`01_eval_report.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/step_segment/iter_01/01_eval_report.md), two dominant failure patterns characterize this baseline:

1. **Severe Temporal Displacement Across Close Boundaries (0.25s – 1.0s shift)**:
   The total number of predicted boundaries across the dataset (443) matches the ground-truth total (445) with exceptional numerical calibration. However, Macro F1 drops steeply when tightening tolerance:
   $$\text{F1@1.0s} = 58.56\% \longrightarrow \text{F1@0.5s} = 40.77\% \longrightarrow \text{F1@0.25s} = 22.75\%$$
   This quantitatively proves that the model consistently detects events proximate to ground truth, but suffers from systematic temporal jitter of 0.3s–0.8s.

2. **Spurious In-Seam Detections on Continuous Operations**:
   In complex continuous sewing operations (`cd12_chuyen1`), the model produces 71 predictions against 52 ground-truth boundaries (Precision: 22.54%). Frame strips (`cd12_chuyen1_FP_t94.29s`, `cd12_chuyen1_FP_t66.25s`) show the model triggering false boundaries mid-seam during steady continuous fabric feeding.

---

## Root-Cause Hypotheses (3 Grounded Hypotheses)

### Hypothesis 1: Temporal Context Truncation due to 5s Window Length
- **Hypothesis Statement**: The 5-second chunk length (75 frames at 15 fps) is shorter than typical industrial sewing sub-cycles (8–15s). Lacking wider contextual anchor points before and after a seam, the model misinterprets intra-cycle micro-pauses (e.g., finger readjustment on fabric) as major step boundaries, causing spurious detections and temporal drift.
- **Domain**: `Model Capacity / Temporal Receptive Field`
- **Confidence Level**: **85% (High)**
- **Supporting Evidence**:
  - *Training Dynamics & Metrics (Pillar 1 & 2)*: F1 scales dramatically with window tolerance (22.75% -> 40.77% -> 58.56%). In long multi-step videos like `cd12_chuyen1`, 19 spurious boundaries were created mid-motion, driving Precision down to 22.54%.
  - *Visual Frame Evidence (Pillar 2)*: Frame strips [`cd12_chuyen1_FP_t94.29s.jpg`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd12_chuyen1_FP_t94.29s.jpg) and [`cd12_chuyen1_FP_t66.25s.jpg`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd12_chuyen1_FP_t66.25s.jpg) show continuous posture without step change, yet a 5s window has no prior context to distinguish an ongoing seam from a fresh start.
- **Contradicting Evidence**:
  - On short cycle videos (`cd10_chuyen1`, `cd4_chuyen1`), prediction count does not overshoot (9 and 10 predictions).
- **What Evidence Is Still Missing**:
  - *Missing Probe*: Controlled comparison evaluating the same model trained/evaluated with a 10s or 12s temporal window (150 frames) to measure whether intra-cycle false positives diminish.

---

### Hypothesis 2: Inadequate Motion Differentiation in 2D Spatial Features
- **Hypothesis Statement**: 2D ResNet-50 features represent static visual appearance per frame rather than explicit temporal velocity or optical flow. In industrial sewing, the background, machine, and fabric color remain static; boundaries are defined primarily by needle start/stop and pedal release. Without explicit motion differencing, the model misses subtle transitions where static appearance barely shifts.
- **Domain**: `Feature Representation / Architecture`
- **Confidence Level**: **75% (Medium-High)**
- **Supporting Evidence**:
  - *Visual Frame Evidence (Pillar 2)*: In [`cd5_chuyen3_FN_t3.05s.jpg`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd5_chuyen3_FN_t3.05s.jpg) and [`cd4_chuyen1_FN_t3.43s.jpg`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd4_chuyen1_FN_t3.43s.jpg), fabric color and worker position across `[t-0.35s, t, t+0.35s]` are visually homogeneous, yielding near-identical ResNet feature embeddings across consecutive frames.
  - *Comparative Performance*: Videos with dynamic fabric rotation and relocation (`cd12_chuyen2`) reach 52.17% F1, whereas repetitive straight-line stitching (`cd4_chuyen1`) drops to 21.05% F1.
- **Contradicting Evidence**:
  - DiffGEBD includes a temporal Transformer decoder intended to correlate cross-frame features, though its input representation remains single-frame 2D embeddings.
- **What Evidence Is Still Missing**:
  - *Missing Probe*: Feature cosine similarity probe comparing adjacent frame features around missed GT timestamps vs true positive timestamps, or testing a 3D spatiotemporal backbone (Video CSN) or Dense Difference Motion stream (DDM).

---

### Hypothesis 3: Human Annotation Margin Variance and Single-Frame Ground Truth Rigidity
- **Hypothesis Statement**: Industrial sewing boundaries lack instantaneous visual phase changes (a seam transition takes 0.3s–0.8s to physically unfold: slowing needle -> releasing pedal -> pulling fabric). The ground-truth single-frame timestamp exhibits annotator reaction latency (typically $\pm 0.3s$), causing strict point evaluation at $\pm 0.25s$ to penalize semantically accurate detections.
- **Domain**: `Label Noise / Data Quality / Evaluation Rigidity`
- **Confidence Level**: **70% (Medium)**
- **Supporting Evidence**:
  - *Pillar 2 Metrics*: Precision and Recall jump from 22.75% to 58.56% when widening tolerance from 0.25s to 1.0s, with total count perfectly matched (445 GT vs 443 Pred).
  - *Visual Inspection*: Inspecting [`cd5_chuyen3_FN_t3.05s.jpg`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/error_cases/cd5_chuyen3_FN_t3.05s.jpg) reveals that the physical transition occurred slightly before/after the exact 3.05s marker.
- **Contradicting Evidence**:
  - Even at $\pm 1.0s$, F1 is 58.56% (not $\ge 80\%$), indicating that genuine false positives and false negatives exist independently of annotator variance.
- **What Evidence Is Still Missing**:
  - *Missing Probe*: Evaluating multi-annotator agreement spread across the validation videos, or testing adaptive Gaussian boundary smoothing with $\sigma \in [0.8s, 1.2s]$ during training.

---

> [!CAUTION]
> **Diagnoser Rule**: Diagnoser isolates root cause hypotheses and missing evidence. It does not propose concrete code solutions or experiment implementations. That responsibility belongs to `03_research_plan.md`.
