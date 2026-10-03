# Proposal for Fixing and Improving BaFormer (Iteration 01)

**Target Dataset:** `dataset_tas_instance` (Instance-Level Temporal Action Segmentation)  
**Baseline Run:** `BaFormer/experiments/tas_instance/bk_fde_tde/final/1/`  
**Reference Docs:** `problem_definition.md`, `ba_former++.md`  

---

## 1. Executive Summary & Root-Cause Diagnosis of Experiment 1

In the initial experiment (`BaFormer/experiments/tas_instance/bk_fde_tde/final/1/log_plain.txt`), training produced the following validation metrics across all later epochs:

```text
Val loss ~800+ | acc% 44.5545 | ins_acc% 44.5545 | edit 0.0000 | f1@10 0.0000 | f1@25 0.0000 | f1@50 0.0000
```

This complete failure (`edit = 0.0`, `f1 = 0.0`, flat `44.55%` accuracy) is not primarily caused by model capacity or learning failure, but by **critical evaluation and inference bugs**:

1. **Segment Obliteration by `_relabeling(r, 100)`:** In `dataset_tas_instance`, the average instance duration is **32.4 frames**, and **97.4% of all instances are shorter than 100 frames** (2,201 out of 2,260 instances). Applying `_relabeling(r, 100)` at inference forcibly overwrites any contiguous segment $\le 100$ frames with its predecessor. This collapses virtually the entire video into a single class prediction.
2. **Exclusion of the Majority Class (`Sewing/Joining`) in Metrics:** In `action_segmentation/utils/metrics.py`, `bg_class = [0]` is hardcoded for `tas_instance` under the false assumption that Class 0 is "background". In reality, Class 0 is **`Sewing/Joining`**, accounting for **47.0% of all ground-truth instances** (1,062 / 2,260). When `_relabeling` collapses the prediction to Class 0, `metrics.py` discards Class 0 completely, leaving zero predicted segments $\rightarrow$ **Edit score = 0.0, F1 = 0.0**. The constant `44.55%` frame accuracy reflects the exact background proportion of `Sewing/Joining` in the validation set.
3. **Flawed Masked Cross-Attention:** The decoder calculates `attn = F.softmax(energy * mask * self.scale, dim=-1)`. Because `mask \in [0, 1]` is multiplied by logits that are often negative, it pulls negative logits toward 0, making them *larger* (closer to 0), which inadvertently increases attention to the masked-out temporal background.

---

## 2. Priority 1: Critical Bug Fixes

### Bug 1.1: Remove or Adapt Post-Processing Segment Thresholding
* **Module:** `BaFormer/main.py`
* **Function:** `inference_bd_peak()`
* **Current Lines (353):**
  ```python
  r = _relabeling(r, 100)
  ```
* **Problem:** Overwrites segments $\le 100$ frames. In this dataset, minimum segment length is 4 frames, mean is 32.4 frames.
* **Proposed Fix:**
  Either disable `_relabeling` completely or lower the threshold to 5 frames:
  ```python
  # Option A (Recommended): disable hardcoded smoothing during eval
  # r = _relabeling(r, 100)

  # Option B: use a tiny threshold matching dataset minimums
  r = _relabeling(r, theta_t=5)
  ```

---

### Bug 1.2: Remove Class 0 from `bg_class` in Metric Computation
* **Module:** `BaFormer/action_segmentation/utils/metrics.py`
* **Functions:** `edit_score()`, `f_score()`
* **Current Lines (99-101, 110-112):**
  ```python
  elif dataset == 'tas_instance':
      bg_class = [0]  # class_id 0 == "background" in classes.json/mapping.txt
  ```
* **Problem:** Class 0 is `Sewing/Joining` (47% of instances). Discarding it breaks both Edit Distance and F1 computation.
* **Proposed Fix:**
  ```python
  elif dataset == 'tas_instance':
      bg_class = []  # tas_instance has NO background class; all 4 classes are valid actions
  ```

---

### Bug 1.3: Fix Boundary Peak Detection Index Offset
* **Module:** `BaFormer/main.py`
* **Function:** `inference_bd_peak()`
* **Current Lines (343-345):**
  ```python
  peak = torch.where((mask_bd[ :-2] < mask_bd[1:-1])
                  & (mask_bd[ 2:] < mask_bd[1:-1] ))[0]
  indices = [0] + peak.tolist() + [mask_bd.shape[-1]]
  ```
* **Problem:** `peak` indexes the slice `mask_bd[1:-1]`. The actual peak index in the full sequence is `peak + 1`. Using `peak` creates a 1-frame systematic shift for all temporal intervals.
* **Proposed Fix:**
  ```python
  peak = torch.where((mask_bd[:-2] < mask_bd[1:-1]) & (mask_bd[2:] < mask_bd[1:-1]))[0]
  indices = [0] + (peak + 1).tolist() + [mask_bd.shape[-1]]
  ```

---

### Bug 1.4: Correct Dataset Class Count in Configs
* **Modules:** `BaFormer/configs/tas_instance.yaml`, `BaFormer/experiments/tas_instance/.../config.yaml`
* **Current Setting:**
  ```yaml
  dataset:
    n_classes: 5
  ```
* **Problem:** `dataset_tas_instance/classes.json` contains only 4 classes (`0: Sewing/Joining`, `1: Positioning/Handling`, `2: Adjustment/Alignment/Preparation`, `3: Inspection/Auxiliary`). Setting `n_classes: 5` prompts the classification head to output $5 + 1 = 6$ logits (including the Hungarian no-object class). Class 4 is a phantom class never present in ground truth, absorbing probability mass and skewing softmax distributions.
* **Proposed Fix:**
  ```yaml
  dataset:
    n_classes: 4
  ```

---

### Bug 1.5: Fix Best Model Checkpointing Metric
* **Module:** `BaFormer/main.py`
* **Class:** `Record_dict` & `main()`
* **Current Lines (160-165, 715):**
  ```python
  if acc > self.acc_max:
      self.acc_max = acc
      self.edit_max = edit
      self.f1_max = f1
      self.best_epoch = epoch
      if_update = True
  ```
* **Problem:** Checkpoints are saved only when frame-wise `acc` increases. In TAS datasets with class imbalance, a degenerate model predicting only the majority class achieves ~45% accuracy while F1 and Edit are 0.0.
* **Proposed Fix:**
  Update `Record_dict` to trigger checkpoint saving on **Mean F1 score** or a combination of Edit and F1:
  ```python
  # Track mean F1: (f1@10 + f1@25 + f1@50) / 3
  f1_mean = sum(f1) / len(f1)
  if f1_mean > self.f1_mean_max:
      self.f1_mean_max = f1_mean
      self.acc_max = acc
      self.edit_max = edit
      self.f1_max = f1
      self.best_epoch = epoch
      if_update = True
  ```

---

### Bug 1.6: Synchronize Validation Loss with Boundary Loss
* **Module:** `BaFormer/main.py`
* **Function:** `validate()`
* **Current Lines (549-560):**
  ```python
  weight_dict = {"loss_ce": config.model.ce_weight,
                 "loss_mask": config.model.mask_weight,
                 "loss_dice": config.model.dice_weight}
  ```
* **Problem:** In `train()`, `weight_dict` includes `"loss_bd": config.model.bd_weight`. In `validate()`, `"loss_bd"` is omitted, so `losses.pop("loss_bd")` discards the boundary loss. The reported train and validation losses are computed with different loss components.
* **Proposed Fix:**
  Add `"loss_bd": config.model.bd_weight` to `validate()`'s `weight_dict`.

---

## 3. Priority 2: Architectural & Loss Improvements

### Improvement 2.1: Additive Masked Cross-Attention in Decoder
* **Module:** `BaFormer/action_segmentation/models/transformer_decoder/transformer_decoder_mask_bd_mulkv.py`
* **Class:** `MultiHeadAttention`
* **Current Lines (233-236):**
  ```python
  energy = torch.einsum('bhqc, bhkc -> bhqk', q, k)
  if mask is None:
      attn = F.softmax(energy * self.scale, dim=-1)
  else:
      attn = F.softmax(energy * mask * self.scale, dim=-1)
  ```
* **Analysis:** In standard Mask2Former, cross-attention is constrained to the predicted mask region by adding a large negative penalty to masked-out frames:
  $$\text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right), \quad M(t) = \begin{cases} 0 & \text{if frame } t \text{ is inside mask} \\ -\infty & \text{otherwise} \end{cases}$$
  Multiplying `energy * mask` does not gate out negative values; it increases them toward zero.
* **Proposed Fix:**
  Convert the sigmoid mask probability into an additive boolean/log mask:
  ```python
  energy = torch.einsum('bhqc, bhkc -> bhqk', q, k) * self.scale
  if mask is not None:
      # mask has shape (B, heads, Q, L) with values in [0, 1]
      # Mask out frames where mask response is low (< 0.5)
      attn_bias = torch.zeros_like(energy)
      attn_bias = attn_bias.masked_fill(mask < 0.5, float('-inf'))
      energy = energy + attn_bias
  attn = F.softmax(energy, dim=-1)
  ```

---

### Improvement 2.2: Proper Positive Weighting in Boundary BCE Loss
* **Module:** `BaFormer/action_segmentation/models/criterion_bd.py`
* **Function:** `loss_boundarys()`
* **Current Lines (310-313):**
  ```python
  bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=target_boundarys * self.pos_weight)
  ```
* **Problem:** In PyTorch, `pos_weight` in `BCEWithLogitsLoss` is a scalar or 1D tensor weighting the positive term $y \cdot \log(\sigma(x))$. Supplying `target_boundarys * self.pos_weight` is redundant and can cause device mismatches.
* **Proposed Fix:**
  ```python
  pos_weight_tensor = torch.tensor([float(self.pos_weight)], device=target_boundarys.device)
  bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
  loss_bd = bce_loss(src_boundayrs, target_boundarys)
  ```

---

### Improvement 2.3: Query Count & Hungarian Matching Costs
* **Modules:** `BaFormer/configs/tas_instance.yaml`, `matcher.py`
* **Current Setting:** `num_queries: 150`
* **Analysis:**
  Videos in `dataset_tas_instance` have an average of 10–25 instances (max $< 50$). Using 150 queries forces the Hungarian matcher to match against ~125 empty-class queries per step.
* **Recommendation:**
  1. Reduce `num_queries` from 150 to **60–80**. This provides sufficient capacity while reducing matching ambiguity and speeding up decoder forward passes.
  2. Increase `cost_dice` relative to `cost_class` in `matcher.py` (e.g. `cost_mask: 5.0, cost_dice: 5.0, cost_class: 1.0`) to prioritize temporal overlap during Hungarian bipartite matching.

---

## 4. Priority 3: Alignment with Problem Definition (ITAS Protocol)

### 4.1 Resolving the Same-Class Boundary Evaluation Bottleneck
As defined in `problem_definition.md` (§2, §8, §11), the core goal of Instance-Level TAS is:
$$\text{Sewing \#1} \mid \text{Sewing \#2} \implies \text{Keep distinct, do NOT merge.}$$

However, the current pipeline converts predicted queries into a dense frame-wise 1D label array `seg_pred`, and `metrics.py::get_labels_start_end_time()` extracts segments by:
```python
if frame_wise_labels[i] != last_label:
    ...
```
This means even if the model predicts two separate instances with Class 0, converting them into frame labels merges them into a single monolithic segment.

### 4.2 Proposed Direct Instance Evaluation (`evaluate_instance_metrics`)
Implement an instance-level evaluation routine that operates directly on predicted instance tuples $(\hat{s}_i, \hat{e}_i, \hat{c}_i)$ and compares them with ground-truth instances $(s_j, e_j, c_j)$ using **Temporal IoU (tIoU)**:

$$tIoU = \frac{\min(e_i, e_j) - \max(s_i, s_j)}{\max(e_i, e_j) - \min(s_i, s_j)}$$

An instance prediction is counted as a True Positive if:
1. $tIoU \ge \tau$ (where $\tau \in \{0.25, 0.50, 0.75\}$)
2. $\hat{c}_i == c_j$
3. It has not been claimed by a higher-scoring match.

This directly measures whether adjacent same-class occurrences were correctly segmented.

---

## 5. Implementation Roadmap

| Phase | Target Items | Affected Files | Expected Impact |
|---|---|---|---|
| **Phase 1** | 1. Remove `_relabeling(r, 100)`<br>2. Set `bg_class = []` for `tas_instance`<br>3. Fix peak offset `peak + 1`<br>4. Set `n_classes: 4`<br>5. Checkpoint by Mean F1<br>6. Add `loss_bd` to val `weight_dict` | `main.py`<br>`metrics.py`<br>`tas_instance.yaml` | **Immediate recovery:** Val Edit and F1 scores become non-zero and accurately reflect model segmentation. |
| **Phase 2** | 1. Fix additive masked cross-attention<br>2. Clean up `pos_weight` in BCE loss<br>3. Tune queries (`num_queries = 60`) | `transformer_decoder_mask_bd_mulkv.py`<br>`criterion_bd.py`<br>`defaults.py` | Eliminates attention leakage outside masks; improves temporal boundary sharpness. |
| **Phase 3** | Implement direct Instance-Level tIoU evaluation | `utils/metrics.py`<br>`main.py` | Full compliance with `problem_definition.md` (no merging of adjacent same-class instances). |
| **Phase 4** | 1. Gaussian Boundary Smoothing ($\sigma=1.5$)<br>2. Full Telemetry & Confusion Matrix<br>3. Component loss logging & 6-panel curves | `main.py`<br>`configs/tas_instance.yaml`<br>`action_segmentation/config/` | Eliminates hard 1-frame jitter/noise in boundary learning; enables rigorous loss balancing and per-class diagnostics. |
