# Proposal & Configuration for BaFormer Run 3 (Iteration 03)

**Dataset:** `dataset_tas_instance` (4 classes, 58 clips, validation split: 10 clips)  
**Previous Checkpoint:** `BaFormer/experiments/tas_instance/bk_fde_tde/final/1/checkpoint_best.pth` (Run 2, Best Epoch 183)  
**Target Run Directory:** `BaFormer/experiments/tas_instance/bk_fde_tde/final_exp03/1/`

---

## 1. Run 2 Performance Summary & Diagnostic Roots

In Run 2 (Gaussian Boundary Smoothing with $\sigma=1.5$, 233 epochs before early stopping triggered):
- **Best Validation Epoch:** 183
- **F1 Mean:** Improved from `26.19` (Run 1) to **`27.84`** (+1.65 improvement, new all-time high).
- **F1@10:** `37.12` $\rightarrow$ **`38.05`** (+0.93)
- **F1@25:** `28.13` $\rightarrow$ **`31.09`** (+2.96, strongest gain)
- **F1@50:** `14.39` (steady)
- **Edit Score:** Dropped from `51.45` to `45.76` (-5.69)
- **Train vs Val Accuracy:** Train Acc reached `71.16%` while Val Acc was `48.09%` (~23% generalization gap).

### Root-Cause Diagnosis
1. **Boundary Over-Cutting (False Positive Explosion):**
   - Telemetry showed **Boundary Recall** was **45.2%**, but **Boundary Precision** was only **9.39%**.
   - For every 1 true boundary cut, the model predicted ~9 false alarms.
   - At inference, `inference_bd_peak` splits action segments at every boundary peak. Excessive false-positive cuts chopped continuous segments into tiny fragments, severely degrading the Edit score (down to 45.76).
2. **Loss Scale Imbalance:**
   - At Epoch 183, `loss_bd` was **7.01** (accounting for **76%** of total loss), while `loss_dice` was **0.60**, `loss_ce` was **1.56**, and `loss_mask` was **0.19**.
   - Boundary loss completely dominated the gradients, starving mask segmentation and classification objectives.
3. **Class 2 Inflation & Class Imbalance:**
   - Ground truth Class 2 (`Adjustment/Alignment/Preparation`) was 1,976 frames, but model predicted **4,240 frames** (+214% inflation) with precision only 17.03%.
   - Class 3 (`Inspection/Auxiliary`) had recall of only 25.1%.
4. **Generalization Gap (Overfitting):**
   - 23% gap between training accuracy (71.16%) and validation accuracy (48.09%), with zero weight decay and low dropout (0.1).

---

## 2. Implemented Solutions for Run 3

| Component | Run 2 Value | Run 3 Value | Rationale |
| :--- | :--- | :--- | :--- |
| `model.bd_weight` | `1.0` | **`0.4`** | Re-balances loss gradient so boundary loss does not monopolize optimization (prevents 76% loss domination). |
| `model.dice_weight` | `1.0` | **`2.5`** | Strongly penalizes segment fragmentation and rewards contiguous temporal overlap. |
| `dataset.pos_weight` | `20.0` | **`6.0`** | Re-calibrated for Gaussian smoothed boundary target mass (effective positive mass is ~21x larger than delta impulses). Suppresses false-positive boundary cuts. |
| `dataset.threshold` | `0.2` | **`0.35`** | Raises peak confidence threshold in `inference_bd_peak` to eliminate noisy low-confidence cuts. |
| `dataset.class_weights` | `None` | **`[0.6, 0.7, 1.0, 1.7]`** | Inverse-frequency weighting in `SetCriterion_bd` to suppress over-prediction of Class 2 and boost recall for minority Class 3. |
| `model.action_seg.transformer_decoder.dropout` | `0.1` | **`0.2`** | Regularization against the ~23% train/val generalization gap. |
| `train.weight_decay` | `0.0` | **`0.0001`** | $L_2$ regularization on network parameters to mitigate overfitting. |
| `model.note` | `final` | **`final_exp03`** | Clean separation of experiment artifacts and checkpoints into its own directory. |

---

## 3. Configuration Verification

All parameters have been synchronized across:
1. `BaFormer/configs/tas_instance.yaml`
2. `BaFormer/action_segmentation/config/defaults.py`
3. `BaFormer/action_segmentation/config/__init__.py`

---

## 4. Run Execution Instructions

Run training directly in your virtual environment:

```bash
cd /home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer
python main.py --config configs/tas_instance.yaml model.note final_exp03
```
