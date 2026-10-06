import os
import argparse
import pathlib
import time
import random
# import matplotlib.colors as mcolors

import math
import numpy as np
# import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer
import matplotlib.pyplot as plt
from PIL import Image
# import shutil



from action_segmentation import(
    augment_crop,
    augment_crop_instances,
    create_dataloader,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
    SetCriterion_bd,
)
from action_segmentation.config.config_node import ConfigNode
from action_segmentation.utils import (
    AverageMeter,
    AverageMeter_acc,
    AverageMeter_f1,
    compute_dense_acc,
    compute_f1,
    edit_score,
    DummyWriter,
    compute_metrics,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)



global_step = 0

def load_config():
    parser = argparse.ArgumentParser(description = "BaFormer for efficient temporal action segmentation.")
    parser.add_argument('--config', default="configs/framed_en_de.yaml", type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()

    if args.config is not None:
        config.merge_from_file(args.config, allow_unsafe=True)
    config.merge_from_list(args.options)

    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config = update_config(config)
    config.freeze()
    return config

def get_set_label(frame_label, num_cls):
    device = frame_label.device
    frame_label = frame_label.tolist()[0]
    frame_label.append(num_cls)
    frame_label = torch.tensor(frame_label)

    exist_set = torch.bincount(frame_label)[:-1].bool().int().to(device)
    return exist_set


def get_loss(input, target):# binary cross-entropy
    input = input.squeeze(0)
    target = target.float()
    assert input.shape == target.shape and len(input.shape) ==1
    act_func = nn.Sigmoid()
    loss_func = nn.BCELoss()
    loss = loss_func(act_func(input), target)
    return loss

class Meter_dict():
    def __init__(self,):
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter_acc()
        self.edit_meter = AverageMeter()
        self.f1_meter = AverageMeter_f1()

        self.ins_acc_meter = AverageMeter_acc()
        self.ord_acc_meter = AverageMeter_acc()
        self.cat_acc_meter = AverageMeter_acc()


    def get_update_metric(self,  dataset, seg_pred, frame_target, loss, ins_seg_pred): # for a batch, to record the metric value
        L = frame_target.shape[1]
        frame_target = frame_target.view(-1)

        seg_pred = seg_pred.view(-1, seg_pred.shape[-1])  # if use 50salads, it need interpeave
        assert seg_pred.shape[0] == frame_target.shape[0]
        num_correct, acc, edit, f_scores = compute_metrics(dataset, seg_pred, frame_target)

        ##-----ins seg
        ins_seg_pred = ins_seg_pred.view(-1, ins_seg_pred.shape[-1])  # if use 50salads, it need interpeave
        assert ins_seg_pred.shape[0] == frame_target.shape[0]
        ins_num_correct, ins_acc, ins_edit, ins_f_scores = compute_metrics(dataset, ins_seg_pred, frame_target)


        loss = loss.item()
        num_correct = num_correct.item()
        acc = acc.item()

        num = frame_target.shape[0]
        self.loss_meter.update(loss, num)
        self.acc_meter.update(num_correct, acc, num)
        self.edit_meter.update(edit, 1)
        self.f1_meter.update(f_scores)

        self.ins_acc_meter.update(ins_num_correct, ins_acc, num)


        metric_results = { 'loss_meter': self.loss_meter,
                            'acc_meter': self.acc_meter,
                            'edit_meter': self.edit_meter,
                            'f1_meter': self.f1_meter,
                           'ins_acc_meter': self.ins_acc_meter,
                           }
        return metric_results

class Record_dict():
    def __init__(self):
        self.reset()
    def reset(self):
        self.acc_max = 0.0
        self.edit_max = 0.0
        self.f1_max = (0.0, 0.0, 0.0)
        self.f1_mean_max = 0.0
        self.composite_max = 0.0
        self.best_epoch = 0
    def update_record(self, epoch, acc, edit, f1):
        if_update = False
        if epoch == 0:
            self.reset()
        else:
            f1_mean = sum(f1) / len(f1) if len(f1) > 0 else 0.0
            # Pareto Composite Score (Pillar 2, proposal_imprv_baformer05):
            # Balances F1 Mean (40%), Edit Score (30%), Frame Accuracy (30%)
            # acc is in [0, 1] so acc * 100.0 is in [0, 100]
            composite_score = 0.4 * f1_mean + 0.3 * edit + 0.3 * (acc * 100.0)
            if composite_score > self.composite_max:
                self.composite_max = composite_score
                self.f1_mean_max = f1_mean
                self.acc_max = acc
                self.edit_max = edit
                self.f1_max = f1
                self.best_epoch = epoch
                if_update = True
        return if_update

def save_training_curves(history, save_path):
    if len(history['epoch']) == 0:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = history['epoch']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Total Loss (Top-Left)
        ax = axes[0, 0]
        if any(v is not None for v in history.get('train_loss', [])):
            ax.plot(epochs, history['train_loss'], label='Train Total Loss', color='tab:blue', lw=1.8)
        val_losses = [v for v in history.get('val_loss', []) if v is not None]
        val_loss_epochs = [e for e, v in zip(epochs, history.get('val_loss', [])) if v is not None]
        if len(val_losses) > 0:
            ax.plot(val_loss_epochs, val_losses, label='Val Total Loss', color='tab:red', lw=1.8, marker='o', markersize=3)
        ax.set_title('Total Loss vs Epoch', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # 2. Train Loss Breakdown (Top-Center)
        ax = axes[0, 1]
        for key, label, col, ls in [
            ('train_loss_ce', 'CE Loss', 'tab:blue', '-'),
            ('train_loss_mask', 'Mask (Focal)', 'tab:orange', '-'),
            ('train_loss_dice', 'Dice Loss', 'tab:green', '-'),
            ('train_loss_bd', 'Boundary BCE', 'tab:red', '-'),
            ('train_loss_enc', 'Enc CE', 'tab:cyan', '-'),
            ('train_loss_smooth', 'Enc Smooth', 'tab:brown', '-'),
            ('train_loss_aux', 'Auxiliary Total', 'tab:purple', '--'),
        ]:
            vals = history.get(key, [])
            if any(v is not None for v in vals):
                ax.plot(epochs, vals, label=label, color=col, linestyle=ls, lw=1.6)
        ax.set_title('Train Loss Breakdown (Components)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Component Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # 3. Val Loss Breakdown (Top-Right)
        ax = axes[0, 2]
        for key, label, col, ls in [
            ('val_loss_ce', 'Val CE', 'tab:blue', '-'),
            ('val_loss_mask', 'Val Mask', 'tab:orange', '-'),
            ('val_loss_dice', 'Val Dice', 'tab:green', '-'),
            ('val_loss_bd', 'Val BD', 'tab:red', '-'),
            ('val_loss_enc', 'Val Enc CE', 'tab:cyan', '-'),
            ('val_loss_smooth', 'Val Smooth', 'tab:brown', '-'),
        ]:
            vals = history.get(key, [])
            v_epochs = [e for e, v in zip(epochs, vals) if v is not None]
            v_vals = [v for v in vals if v is not None]
            if len(v_vals) > 0:
                ax.plot(v_epochs, v_vals, label=label, color=col, linestyle=ls, lw=1.6, marker='.', markersize=3)
        ax.set_title('Val Loss Breakdown (Components)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Component Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # 4. Accuracy (Bottom-Left)
        ax = axes[1, 0]
        if any(v is not None for v in history.get('train_acc', [])):
            ax.plot(epochs, history['train_acc'], label='Train Acc %', color='tab:blue', lw=1.8)
        val_accs = [v for v in history.get('val_acc', []) if v is not None]
        val_acc_epochs = [e for e, v in zip(epochs, history.get('val_acc', [])) if v is not None]
        if len(val_accs) > 0:
            ax.plot(val_acc_epochs, val_accs, label='Val Acc %', color='tab:green', lw=1.8, marker='o', markersize=3)
        ax.set_title('Frame Accuracy (%) vs Epoch', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # 5. Edit Score (Bottom-Center)
        ax = axes[1, 1]
        if any(v is not None for v in history.get('train_edit', [])):
            ax.plot(epochs, history['train_edit'], label='Train Edit', color='tab:blue', lw=1.8)
        val_edits = [v for v in history.get('val_edit', []) if v is not None]
        val_edit_epochs = [e for e, v in zip(epochs, history.get('val_edit', [])) if v is not None]
        if len(val_edits) > 0:
            ax.plot(val_edit_epochs, val_edits, label='Val Edit', color='tab:orange', lw=1.8, marker='o', markersize=3)
        ax.set_title('Edit Score vs Epoch', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Edit Score')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # 6. F1 Scores (Bottom-Right)
        ax = axes[1, 2]
        val_f1_epochs = [e for e, v in zip(epochs, history.get('val_f1_mean', [])) if v is not None]
        if len(val_f1_epochs) > 0:
            val_f1_10 = [v for v in history['val_f1_10'] if v is not None]
            val_f1_25 = [v for v in history['val_f1_25'] if v is not None]
            val_f1_50 = [v for v in history['val_f1_50'] if v is not None]
            val_f1_mean = [v for v in history['val_f1_mean'] if v is not None]
            ax.plot(val_f1_epochs, val_f1_10, label='Val F1@10', linestyle='--', lw=1.5)
            ax.plot(val_f1_epochs, val_f1_25, label='Val F1@25', linestyle='-.', lw=1.5)
            ax.plot(val_f1_epochs, val_f1_50, label='Val F1@50', linestyle=':', lw=1.5)
            ax.plot(val_f1_epochs, val_f1_mean, label='Val F1 Mean', color='tab:purple', lw=2.2, marker='s', markersize=3)
        ax.set_title('Validation F1 Scores vs Epoch', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        pass


def plot_confusion_matrix(cm_norm, class_names, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted Class", fontweight='bold', fontsize=11)
        ax.set_ylabel("True Class", fontweight='bold', fontsize=11)
        ax.set_title("Validation Confusion Matrix (Normalized)", fontweight='bold', fontsize=12)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j]
                text_col = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", color=text_col, fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        pass


def get_git_info():
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        return f"{branch}@{commit}"
    except Exception:
        return "unknown"


def save_experiment_artifacts(output_dir, config, epoch, val_metrics, val_record, history, early_stopped=False, is_final=False):
    try:
        import datetime
        import json
        out_path = pathlib.Path(output_dir)

        # 1. Confusion Matrix & Per-Class Metrics
        if val_metrics is not None and "cm" in val_metrics:
            cm = val_metrics["cm"]
            class_names = val_metrics.get("class_names", [f"Class {i}" for i in range(cm.shape[0])])
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, cm_sum, where=cm_sum != 0, out=np.zeros_like(cm, dtype=float))

            plot_confusion_matrix(cm_norm, class_names, out_path / "confusion_matrix.png")

            support = cm.sum(axis=1)
            pred_counts = cm.sum(axis=0)
            true_pos = np.diag(cm)

            with open(out_path / "per_class_metrics.csv", "w", encoding="utf-8") as f:
                f.write("class_id,class_name,support_frames,predicted_frames,precision,recall,f1\n")
                for k in range(len(class_names)):
                    prec = float(true_pos[k] / max(1, pred_counts[k]))
                    rec = float(true_pos[k] / max(1, support[k]))
                    f1 = float(2 * prec * rec / max(1e-8, prec + rec))
                    f.write(f"{k},{class_names[k]},{int(support[k])},{int(pred_counts[k])},{prec:.4f},{rec:.4f},{f1:.4f}\n")

        # 2. Experiment Summary JSON
        summary = {
            "meta": {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "git_commit": get_git_info(),
                "dataset": config.dataset.name,
                "model": config.model.name,
                "epochs_trained": epoch,
                "best_epoch": getattr(val_record, 'best_epoch', epoch),
                "is_final": is_final,
                "status": "early_stopped" if early_stopped else ("completed" if is_final else "in_progress"),
            },
            "hyperparameters": {
                "batch_size": getattr(config.train, 'batch_size', 1),
                "base_lr": getattr(config.train, 'base_lr', 0.0005),
                "optimizer": getattr(config.train, 'optimizer', 'adam'),
                "weight_decay": getattr(config.train, 'weight_decay', 0.0),
                "early_stopping_patience": getattr(config.train, 'early_stopping_patience', 0),
                "ce_weight": getattr(config.model, 'ce_weight', 1.0),
                "mask_weight": getattr(config.model, 'mask_weight', 5.0),
                "dice_weight": getattr(config.model, 'dice_weight', 1.0),
                "bd_weight": getattr(config.model, 'bd_weight', 1.0),
                "pos_weight": getattr(config.dataset, 'pos_weight', 32.0),
                "num_queries": getattr(config.model.action_seg.transformer_decoder, 'num_queries', 150),
                "dec_layers": getattr(config.model.action_seg.transformer_decoder, 'dec_layers', 10),
                "threshold": getattr(config.dataset, 'threshold', 0.2),
                "enc_smooth_weight": getattr(config.model, 'enc_smooth_weight', 0.15),
            },
            "best_validation_scores": {
                "epoch": getattr(val_record, 'best_epoch', epoch),
                "composite_score": round(float(getattr(val_record, 'composite_max', 0.0)), 2),
                "f1_mean": round(float(getattr(val_record, 'f1_mean_max', 0.0)), 2),
                "f1@10": round(float(getattr(val_record, 'f1_max', (0,0,0))[0]), 2),
                "f1@25": round(float(getattr(val_record, 'f1_max', (0,0,0))[1]), 2),
                "f1@50": round(float(getattr(val_record, 'f1_max', (0,0,0))[2]), 2),
                "accuracy": round(float(getattr(val_record, 'acc_max', 0.0) * 100), 2),
                "edit_score": round(float(getattr(val_record, 'edit_max', 0.0)), 2),
            },
        }

        if val_metrics is not None:
            summary["latest_validation_metrics"] = {
                "loss_total": round(float(val_metrics.get("loss", 0.0)), 4),
                "loss_ce": round(float(val_metrics.get("loss_ce", 0.0)), 4),
                "loss_mask": round(float(val_metrics.get("loss_mask", 0.0)), 4),
                "loss_dice": round(float(val_metrics.get("loss_dice", 0.0)), 4),
                "loss_bd": round(float(val_metrics.get("loss_bd", 0.0)), 4),
                "loss_enc": round(float(val_metrics.get("loss_enc", 0.0)), 4),
                "loss_smooth": round(float(val_metrics.get("loss_smooth", 0.0)), 4),
                "loss_aux": round(float(val_metrics.get("loss_aux", 0.0)), 4),
                # Legacy aliases: headline boundary metric at +/-0.25s tolerance
                # (kept for backward compatibility with existing dashboards/scripts
                # that expect a single "tol3"-named boundary metric).
                "boundary_f1_tol3": round(float(val_metrics.get("bd_f1_3", 0.0)), 2),
                "boundary_precision_tol3": round(float(val_metrics.get("bd_prec_3", 0.0)), 2),
                "boundary_recall_tol3": round(float(val_metrics.get("bd_rec_3", 0.0)), 2),
                # Full per-tolerance breakdown (+/-0.1s, +/-0.25s, +/-0.5s), replacing
                # the old fixed +/-3 frame tolerance.
                "boundary_f1_by_tol_seconds": {str(k): round(float(v), 2) for k, v in val_metrics.get("bd_f1_by_tol", {}).items()},
                "boundary_precision_by_tol_seconds": {str(k): round(float(v), 2) for k, v in val_metrics.get("bd_prec_by_tol", {}).items()},
                "boundary_recall_by_tol_seconds": {str(k): round(float(v), 2) for k, v in val_metrics.get("bd_rec_by_tol", {}).items()},
            }
            if "per_class" in val_metrics:
                summary["per_class_summary"] = val_metrics["per_class"]

        with open(out_path / "experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    except Exception as e:
        pass


def create_heatmap(sequence, sigma=2.0):
    device = sequence.device
    # Find the positions of '1' in the sequence
    sigma = torch.tensor([sigma]).to(device)
    peak_positions = torch.nonzero(sequence).squeeze(dim=-1)

    # Create a 1D tensor representing the indices
    indices = torch.arange(0, len(sequence)).to(device)

    # Calculate Gaussian functions for each peak position
    gaussians = [torch.exp(-(indices - peak_pos).float() ** 2 / (2 * sigma ** 2)) for peak_pos in peak_positions]

    # Sum up the Gaussian functions
    heatmap = torch.max(torch.stack(gaussians), dim=0)[0]

    return heatmap


def create_boundary_heatmap(sequence: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Applies Gaussian boundary smoothing around boundary transition points (s_i).
    Creates a smooth temporal Gaussian bell curve of standard deviation `sigma`
    around each boundary cut, mitigating hard boundary label noise and jitter.
    If sigma is None or <= 0, returns the input sequence (hard 0/1).
    """
    if sigma is None or sigma <= 0:
        return sequence
    device = sequence.device
    peak_positions = torch.nonzero(sequence).squeeze(dim=-1)
    if peak_positions.numel() == 0:
        return sequence
    if peak_positions.dim() == 0:
        peak_positions = peak_positions.unsqueeze(0)

    indices = torch.arange(0, len(sequence), device=device, dtype=torch.float32)
    diffs = indices.unsqueeze(0) - peak_positions.unsqueeze(1).float()
    gaussians = torch.exp(-(diffs ** 2) / (2.0 * (float(sigma) ** 2)))
    heatmap = torch.max(gaussians, dim=0)[0]
    return heatmap


def extract_peaks_nms(
    bd_prob: np.ndarray,
    threshold: float = 0.18,
    min_distance: int = 8,
    prominence: float = 0.065,
) -> list[int]:
    """Adaptive Shoulder-Suppression NMS for Action Boundaries (Pillar 4, proposal_imprv_baformer13).

    1. Finds local maxima above threshold that exceed local 7-frame floor by prominence.
    2. Enforces min_distance by Non-Maximum Suppression (keeps highest peak within window).
    Eliminates multi-frame shoulder ripples and non-prominent background jitter.
    """
    L = len(bd_prob)
    if L < 3:
        return []

    # Local floor via 7-frame min pool
    pad_bd = np.pad(bd_prob, (3, 3), mode='edge')
    local_min = np.array([np.min(pad_bd[i : i + 7]) for i in range(L)])
    prom = bd_prob - local_min

    # Candidate peaks
    cand_cond = (
        (bd_prob[1:-1] >= threshold)
        & (bd_prob[1:-1] >= bd_prob[:-2])
        & (bd_prob[1:-1] >= bd_prob[2:])
        & (prom[1:-1] >= prominence)
    )
    raw_peaks = (np.where(cand_cond)[0] + 1).tolist()

    if not raw_peaks:
        return []

    # Sort peaks by probability descending for greedy NMS
    sorted_peaks = sorted(raw_peaks, key=lambda p: bd_prob[p], reverse=True)
    kept_peaks = []
    suppressed = np.zeros(L, dtype=bool)

    for p in sorted_peaks:
        if suppressed[p]:
            continue
        kept_peaks.append(p)
        low = max(0, p - min_distance)
        high = min(L, p + min_distance + 1)
        suppressed[low:high] = True

    return sorted(kept_peaks)


def prepare_target(frame_tgs, sigma, boundary_sigma=None):
    b, l = frame_tgs.shape #just 1 sample, b=1
    frame_tg = frame_tgs.squeeze(dim=0)
    gt_mask = []
    gt_class = []
    current = frame_tg[0]
    start = 0
    frame_bd = torch.zeros_like(frame_tg, dtype=torch.float32)
    for j in range(l):
        if j != l-1:
            if frame_tg[j] != current:
                end = j
                mask = torch.zeros(l).to(frame_tgs)
                mask[start: end] = 1
                if sigma is not None:
                    mask = create_heatmap(mask, sigma)

                gt_mask.append(mask.unsqueeze(dim=0))
                gt_class.append(current.unsqueeze(dim=0))
                frame_bd[end] = 1.0 # start of the next segment
                start = j
                current = frame_tg[j]

        else: #the final frame
            end = l # j=l-1
            mask = torch.zeros(l).to(frame_tgs)
            mask[start: end] = 1
            if sigma is not None:
                mask = create_heatmap(mask, sigma)
            gt_mask.append(mask.unsqueeze(dim=0))
            gt_class.append(current.unsqueeze(dim=0))
    gt_mask = torch.cat(gt_mask)
    gt_class = torch.cat(gt_class)

    if boundary_sigma is not None and boundary_sigma > 0:
        frame_bd = create_boundary_heatmap(frame_bd, boundary_sigma)

    target = {  'labels': gt_class,
                'masks': gt_mask,
                'boundarys': frame_bd,
                'frame_target': frame_tg}
    targets = [target] # if we have 1 sample in a batch
    return targets


def prepare_target_from_instances(instances, seq_len, sigma, boundary_sigma=None, frame_target=None):
    """Instance-level equivalent of `prepare_target()`, for datasets that
    provide real `(start_frame, end_frame, class_id)` instance annotations
    (see `InstanceDatasetFolder` / `src/TAS-instance-style/problem_definition.md`)
    instead of only a frame-wise class sequence.

    Unlike `prepare_target()` -- which re-derives segments by run-length
    splitting a frame-wise label sequence on class *change*, and therefore
    can never separate two adjacent instances of the same class -- this
    builds one GT mask/class/boundary target directly PER INSTANCE, so
    same-class adjacent instances are always kept as distinct Hungarian-
    matching targets.

    Args:
        instances: (N, 3) LongTensor (or ndarray-like) of
                   [start_frame, end_frame, class_id] rows, batch dim
                   squeezed out (batch_size == 1).
        seq_len:   int, total number of frames the masks/boundary should
                   span (matches the model's temporal resolution).
        sigma:     optional Gaussian heatmap sigma for masks; None => hard 0/1 masks.
        boundary_sigma: optional Gaussian smoothing sigma (e.g. 1.0 or 1.5) for boundary cuts.
        frame_target: optional frame-wise ground truth class IDs tensor [seq_len].
    """
    if instances.dim() == 3:  # (batch=1, N, 3) from default_collate -> squeeze
        instances = instances.squeeze(dim=0)
    device = instances.device if torch.is_tensor(instances) else None

    gt_mask = []
    gt_class = []
    frame_bd = torch.zeros(seq_len, dtype=torch.float32, device=device)

    for idx in range(instances.shape[0]):
        start = int(instances[idx, 0])
        end = int(instances[idx, 1])
        class_id = instances[idx, 2]
        end = min(end, seq_len)
        start = min(start, end)

        mask = torch.zeros(seq_len, dtype=torch.float32, device=device)
        mask[start:end] = 1
        if sigma is not None:
            mask = create_heatmap(mask, sigma)
        gt_mask.append(mask.unsqueeze(dim=0))
        gt_class.append(class_id.view(1) if torch.is_tensor(class_id) else torch.tensor([class_id]))

        if idx > 0:  # instance start == boundary; skip the very first instance (frame 0, no left neighbor)
            frame_bd[start] = 1.0

    gt_mask = torch.cat(gt_mask)
    gt_class = torch.cat(gt_class).long()

    if boundary_sigma is not None and boundary_sigma > 0:
        frame_bd = create_boundary_heatmap(frame_bd, boundary_sigma)

    if frame_target is not None:
        ft = frame_target.squeeze(0) if frame_target.dim() > 1 else frame_target
    else:
        ft = torch.zeros(seq_len, dtype=torch.long, device=device)
        for idx in range(instances.shape[0]):
            s = int(instances[idx, 0])
            e = min(int(instances[idx, 1]), seq_len)
            c = int(instances[idx, 2])
            ft[s:e] = c

    target = {'labels': gt_class,
              'masks': gt_mask,
              'boundarys': frame_bd,
              'frame_target': ft,
              'instances': instances}
    targets = [target]  # if we have 1 sample in a batch
    return targets


def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum('qc,ql->cl', mask_cls, mask_pred).transpose(0,1)
    return semseg
def inference(prediction):
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    processed_results = []
    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
        r = semantic_inference(mask_cls_result, mask_pred_result)
        processed_results.append(r)
    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred


def inference_bd(prediction):
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    assert 'pred_boundarys' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    mask_boundary_results = prediction['pred_boundarys']
    processed_results = []
    for mask_cls, mask_pred, mask_bd in zip(mask_cls_results, mask_pred_results, mask_boundary_results):
        mask_ref = torch.zeros_like(mask_pred)
        mask_bd = (mask_bd.sigmoid()>0.3).float().squeeze(dim=0)
        mask_bd[0] = 1
        mask_bd[-1] = 1
        indices = torch.nonzero(mask_bd)
        for i in range(len(indices)-1):
            star, end = indices[i], indices[i+1]
            mask_split = mask_pred[:, star:end].sigmoid().sum(dim=-1).argmax(0)
            mask_ref[mask_split,star:end] = 1
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        r = torch.einsum('qc,ql->cl', mask_cls, mask_ref).transpose(0, 1)
        processed_results.append(r)
    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred

def inference_bd_peak(prediction, threshold):
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    assert 'pred_boundarys' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    mask_boundary_results = prediction['pred_boundarys']
    processed_results = []
    for mask_cls, mask_pred, mask_bd in zip(mask_cls_results, mask_pred_results, mask_boundary_results):
        mask_bd = mask_bd.sigmoid().float().squeeze(dim=0)
        mask_bd[mask_bd < threshold] = 0.0
        peak = torch.where((mask_bd[ :-2] < mask_bd[1:-1])
                        & (mask_bd[ 2:] < mask_bd[1:-1] ))[0]
        indices = [0] + (peak + 1).tolist() + [mask_bd.shape[-1]]
        mask_ref = torch.zeros_like(mask_pred)
        for i in range(len(indices)-1):
            star, end = indices[i], indices[i+1]
            mask_split = mask_pred[:, star:end].sigmoid().sum(dim=-1).argmax(0)
            mask_ref[mask_split,star:end] = 1
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        r = torch.einsum('qc,ql->cl', mask_cls, mask_ref).transpose(0, 1)
        r = _relabeling(r, theta_t=5)
        processed_results.append(r)
    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred


def inference_energy_fusion(prediction, threshold=0.15, min_duration=7, theta_t=7, prominence=0.035):
    """Pillar 5 (proposal_imprv_baformer14): Bipartite Boundary Snapping Energy Fusion.

    Harmonizes T-ASPP boundary peaks and Transformer query representations:
    1. Per-clip dynamic boundary threshold:
       tau_clip = clamp(mu_bd + 1.0 * std_bd, threshold, 0.30)
    2. Candidate cut extraction:
       - Local boundary peaks above tau_clip (NMS with min_distance=min_duration, prominence=prominence)
       - Also inspects query transfer points: if dominant query switches, snaps to local peak
    3. Merges and sorts unique cuts, enforcing min_duration constraint
    4. Interval Query Assignment:
       For each interval [s_k, s_{k+1}], select winning query by joint score:
       score(q) = sum_{t=s_k}^{s_{k+1}} mask_prob[q, t] * fg_prob[q]
    5. Builds segment predictions and applies micro-segment relabeling (theta_t).
    """
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    assert 'pred_boundarys' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    mask_boundary_results = prediction['pred_boundarys']
    processed_results = []

    for mask_cls, mask_pred, mask_bd in zip(mask_cls_results, mask_pred_results, mask_boundary_results):
        mask_prob = mask_pred.sigmoid()  # [Q, L]
        Q, L = mask_prob.shape
        bd_prob = mask_bd.sigmoid().squeeze(0)  # [L]

        # 1. Video-Adaptive Dynamic Threshold (VDT, Pillar 4, proposal_imprv_baformer14)
        mu_bd = bd_prob.mean()
        std_bd = bd_prob.std()
        tau_clip = torch.clamp(mu_bd + 1.0 * std_bd, min=threshold, max=0.30).item()

        # 2. Extract boundary peaks via Adaptive NMS
        candidate_cuts = extract_peaks_nms(
            bd_prob.detach().cpu().numpy(),
            threshold=tau_clip,
            min_distance=min_duration,
            prominence=prominence,
        )

        # 3. Query transfer candidate points with peak snapping
        raw_dominant_q = mask_prob.argmax(dim=0)  # [L]
        filtered_q = raw_dominant_q.clone()
        for t in range(1, L - 1):
            if raw_dominant_q[t - 1] == raw_dominant_q[t + 1] and raw_dominant_q[t] != raw_dominant_q[t - 1]:
                filtered_q[t] = raw_dominant_q[t - 1]

        for t in range(1, L):
            if filtered_q[t] != filtered_q[t - 1]:
                w_start = max(0, t - 3)
                w_end = min(L, t + 4)
                local_peak_idx = w_start + int(torch.argmax(bd_prob[w_start:w_end]).item())
                if bd_prob[local_peak_idx] >= 0.12 and 0 < local_peak_idx < L:
                    candidate_cuts.append(local_peak_idx)

        unique_cuts = sorted(list(set(candidate_cuts)))

        # 4. Minimum duration constraint (min_duration = 8 frames)
        pruned_cuts = []
        last_cut = 0
        for cut in unique_cuts:
            if cut - last_cut >= min_duration:
                pruned_cuts.append(cut)
                last_cut = cut
            elif len(pruned_cuts) > 0 and bd_prob[cut] > bd_prob[pruned_cuts[-1]]:
                pruned_cuts[-1] = cut
                last_cut = cut

        indices = [0] + pruned_cuts + [L]

        # 5. Semantic-Dominant Interval Query Voting (Pillar 2, proposal_imprv_baformer12)
        # Prevents queries far from temporal anchor from being falsely suppressed.
        mask_ref = torch.zeros_like(mask_pred)
        mask_cls_prob = F.softmax(mask_cls, dim=-1)
        fg_prob = 1.0 - mask_cls_prob[:, -1]  # [Q], query foreground confidence

        for i in range(len(indices) - 1):
            star, end = indices[i], indices[i + 1]
            if end <= star:
                continue
            interval_mass = mask_prob[:, star:end].sum(dim=-1)  # [Q]
            joint_score = interval_mass * fg_prob
            winning_query = joint_score.argmax(0)
            mask_ref[winning_query, star:end] = 1.0

        r = torch.einsum('qc,ql->cl', mask_cls_prob[:, :-1], mask_ref).transpose(0, 1)
        r = _relabeling(r, theta_t=theta_t)
        processed_results.append(r)

    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred


def inference_snap_dual(prediction, threshold=0.25, min_duration=8, window_radius=2, theta_t=8):
    """Pillar 2 (proposal_imprv_baformer08): Boundary-Snapping Dual-Confirmation Inference.

    Synthesizes the boundary precision of Run 7 with the structural stability of Run 5:
    1. Micro-flickering suppression: 3-frame mode filter on dominant query sequence.
    2. Dual Confirmation with Peak Snapping:
       Candidate cuts only originate when dominant query switches (filtered_q[t] != filtered_q[t-1]).
       Searches a local window [t - window_radius, t + window_radius] for local boundary peak tau*.
       If b(tau*) >= threshold, tau* is accepted as a confirmed cut.
       If b(tau*) < threshold, candidate is rejected as an intra-segment query fluctuation.
    3. Minimum Duration Pruning (min_duration = 8 frames):
       Prunes cuts closer than min_duration, preserving the cut with higher boundary probability.
    4. Interval Query Voting:
       For each confirmed interval [s_i, s_{i+1}], query is selected by argmax accumulated mask probability.
    5. Consolidation relabeling: Micro-segments < theta_t frames are merged into neighbors.
    """
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    assert 'pred_boundarys' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    mask_boundary_results = prediction['pred_boundarys']
    processed_results = []
    for mask_cls, mask_pred, mask_bd in zip(mask_cls_results, mask_pred_results, mask_boundary_results):
        mask_prob = mask_pred.sigmoid()  # [Q, L]
        L = mask_prob.shape[-1]
        raw_dominant_q = mask_prob.argmax(dim=0)  # [L]

        # 1. Micro-flickering suppression (3-frame local mode filter)
        filtered_q = raw_dominant_q.clone()
        for t in range(1, L - 1):
            if raw_dominant_q[t - 1] == raw_dominant_q[t + 1] and raw_dominant_q[t] != raw_dominant_q[t - 1]:
                filtered_q[t] = raw_dominant_q[t - 1]

        bd_prob = mask_bd.sigmoid().squeeze(0)  # [L]

        # 2. Query transfer candidate points
        transfer_frames = []
        for t in range(1, L):
            if filtered_q[t] != filtered_q[t - 1]:
                transfer_frames.append(t)

        # Snap to local boundary peak in [t - window_radius, t + window_radius]
        candidate_cuts = []
        for t in transfer_frames:
            w_start = max(0, t - window_radius)
            w_end = min(L, t + window_radius + 1)
            local_peak_idx = w_start + int(torch.argmax(bd_prob[w_start:w_end]).item())
            if bd_prob[local_peak_idx] >= threshold and 0 < local_peak_idx < L:
                candidate_cuts.append(local_peak_idx)

        unique_cuts = sorted(list(set(candidate_cuts)))

        # 3. Minimum duration constraint
        pruned_cuts = []
        last_cut = 0
        for cut in unique_cuts:
            if cut - last_cut >= min_duration:
                pruned_cuts.append(cut)
                last_cut = cut
            elif len(pruned_cuts) > 0 and bd_prob[cut] > bd_prob[pruned_cuts[-1]]:
                pruned_cuts[-1] = cut
                last_cut = cut

        indices = [0] + pruned_cuts + [L]

        # 4. Interval-level query voting
        mask_ref = torch.zeros_like(mask_pred)
        for i in range(len(indices) - 1):
            star, end = indices[i], indices[i + 1]
            if end <= star:
                continue
            winning_query = mask_prob[:, star:end].sum(dim=-1).argmax(0)
            mask_ref[winning_query, star:end] = 1.0

        mask_cls_prob = F.softmax(mask_cls, dim=-1)[..., :-1]
        r = torch.einsum('qc,ql->cl', mask_cls_prob, mask_ref).transpose(0, 1)
        r = _relabeling(r, theta_t=theta_t)
        processed_results.append(r)

    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred


def inference_window_voting(prediction, threshold=0.15, min_duration=4, prominence=0.035, theta_t=4):
    """Pillar 2 (proposal_imprv_baformer07): Prominence-Filtered Window-Voting Inference.

    1. Boundary Peak Extraction with Prominence Filtering:
       Detects local probability maxima in pred_boundarys exceeding threshold AND exceeding
       the local 7-frame minimum by at least `prominence` (prom >= 0.035).
       This rejects false plateau jitters while reliably capturing genuine focal-loss transitions (p in [0.15, 0.35]).
    2. Dynamic Cut Filtering: Enforces min_duration=4. If two candidate peaks are closer than min_duration,
       the peak with the higher probability is kept.
    3. Interval-Level Query Voting: For each interval [start, end] between cuts, every query votes
       using its integrated sigmoid mask response:
           q* = argmax_q sum_{t=start}^{end} sigmoid(pred_masks[q, t])
       This completely eliminates the 1-frame temporal disalignment between query switches and boundary peaks.
    4. Relabeling: Smooths spurious micro-segments < theta_t frames.
    """
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    assert 'pred_boundarys' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    mask_boundary_results = prediction['pred_boundarys']
    processed_results = []
    for mask_cls, mask_pred, mask_bd in zip(mask_cls_results, mask_pred_results, mask_boundary_results):
        mask_prob = mask_pred.sigmoid()  # [Q, L]
        bd_prob = mask_bd.sigmoid().float().squeeze(0)  # [L]

        # 1. Local maxima boundary peaks with prominence filter
        if len(bd_prob) >= 3:
            pad_bd = F.pad(bd_prob.unsqueeze(0).unsqueeze(0), (3, 3), mode='replicate')
            local_min = -F.max_pool1d(-pad_bd, kernel_size=7, stride=1, padding=0).squeeze()
            prom = bd_prob - local_min
            peak_cond = (bd_prob[1:-1] >= threshold) & \
                        (bd_prob[1:-1] >= bd_prob[:-2]) & \
                        (bd_prob[1:-1] >= bd_prob[2:]) & \
                        (prom[1:-1] >= prominence)
            raw_peaks = (torch.where(peak_cond)[0] + 1).tolist()
        else:
            raw_peaks = []

        # 2. Prune peaks closer than min_duration (keep higher peak)
        pruned_cuts = []
        last_cut = 0
        for peak in raw_peaks:
            if peak - last_cut >= min_duration:
                pruned_cuts.append(peak)
                last_cut = peak
            elif len(pruned_cuts) > 0 and bd_prob[peak] > bd_prob[pruned_cuts[-1]]:
                pruned_cuts[-1] = peak
                last_cut = peak

        indices = [0] + pruned_cuts + [mask_prob.shape[-1]]

        # 3. Interval-level query voting
        mask_ref = torch.zeros_like(mask_pred)
        for i in range(len(indices) - 1):
            star, end = indices[i], indices[i + 1]
            if end <= star:
                continue
            winning_query = mask_prob[:, star:end].sum(dim=-1).argmax(0)
            mask_ref[winning_query, star:end] = 1.0

        mask_cls_prob = F.softmax(mask_cls, dim=-1)[..., :-1]
        r = torch.einsum('qc,ql->cl', mask_cls_prob, mask_ref).transpose(0, 1)
        r = _relabeling(r, theta_t=theta_t)
        processed_results.append(r)
    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred


def inference_query_dominance(prediction, threshold, min_duration=4, theta_t=5):
    """Pillar 3 (proposal_imprv_baformer05): Duration-Constrained Query-Dominance inference.

    1. Micro-flickering suppression: Applies a 3-frame local mode filter on dominant_q
       to eliminate 1-frame query fluttering.
    2. Dual confirmation: A cut point is considered when dominant query changes AND
       the boundary head probability exceeds threshold: (query_transfer & (b(t) > threshold)).
    3. Minimum duration constraint: Enforces min_duration=4 (verified from dataset audit:
       all ground truth instances are >= 4 frames). Cuts closer than min_duration are pruned.
    4. Optimal interval query assignment: Each interval [start, end] is assigned the query
       with the highest accumulated sigmoid mask probability over that span.
    """
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    assert 'pred_boundarys' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    mask_boundary_results = prediction['pred_boundarys']
    processed_results = []
    for mask_cls, mask_pred, mask_bd in zip(mask_cls_results, mask_pred_results, mask_boundary_results):
        mask_prob = mask_pred.sigmoid()  # [Q, L]
        raw_dominant_q = mask_prob.argmax(dim=0)  # [L] - query owning each frame

        # Micro-flickering filter: eliminate isolated 1-frame spikes in dominant_q
        filtered_q = raw_dominant_q.clone()
        for t in range(1, len(filtered_q) - 1):
            if raw_dominant_q[t - 1] == raw_dominant_q[t + 1] and raw_dominant_q[t] != raw_dominant_q[t - 1]:
                filtered_q[t] = raw_dominant_q[t - 1]

        bd_prob = mask_bd.sigmoid().squeeze(0)  # [L]

        # Query transfer signal: dominant query changes at frame t vs t-1
        query_transfer = torch.zeros_like(bd_prob, dtype=torch.bool)
        query_transfer[1:] = filtered_q[1:] != filtered_q[:-1]

        # Internal candidate cut points (exclude endpoints 0 and L-1)
        confirmed_bd = query_transfer & (bd_prob > threshold)
        confirmed_bd[0] = False
        raw_cuts = torch.where(confirmed_bd)[0].tolist()

        # Enforce minimum duration constraint (min_duration = 4)
        pruned_cuts = []
        last_cut = 0
        for cut in raw_cuts:
            if cut - last_cut >= min_duration:
                pruned_cuts.append(cut)
                last_cut = cut

        indices = [0] + pruned_cuts + [mask_prob.shape[-1]]

        mask_ref = torch.zeros_like(mask_pred)
        for i in range(len(indices) - 1):
            star, end = indices[i], indices[i + 1]
            if end <= star:
                continue
            mask_split = mask_pred[:, star:end].sigmoid().sum(dim=-1).argmax(0)
            mask_ref[mask_split, star:end] = 1

        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        r = torch.einsum('qc,ql->cl', mask_cls, mask_ref).transpose(0, 1)
        r = _relabeling(r, theta_t=theta_t)
        processed_results.append(r)
    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred


def _relabeling(outputs, theta_t):
    preds = outputs.argmax(dim=1)
    last = preds[0]
    cnt = 1
    for j in range(1, len(preds)):
        if last == preds[j]:
            cnt += 1
        else:
            if cnt > theta_t:
                cnt = 1
                last = preds[j]
            else:
                outputs[j - cnt : j, :] = outputs[j - cnt - 1, :]
                cnt = 1
                last = preds[j]

    if cnt <= theta_t:
        outputs[j - cnt: j, :] = outputs[j - cnt - 1, :]

    return outputs


# train one epoch
def train(epoch, config, model, optimizer, scheduler, train_loader, logger, tensorboard_writer, tensorboard_writer2):
    global global_step

    logger.info(f'Train {epoch} {global_step}')
    device = torch.device(config.device)
    model.train()

    #-----init meter_dict
    meter_dict = Meter_dict()
    new_edit_meter = AverageMeter()
    before_edit_meter = AverageMeter()

    # Track individual loss components
    loss_ce_meter = AverageMeter()
    loss_mask_meter = AverageMeter()
    loss_dice_meter = AverageMeter()
    loss_bd_meter = AverageMeter()
    loss_contra_meter = AverageMeter()
    loss_enc_meter = AverageMeter()
    loss_smooth_meter = AverageMeter()
    loss_tv_meter = AverageMeter()
    loss_repulse_meter = AverageMeter()
    loss_aux_meter = AverageMeter()

    criterion = SetCriterion_bd(config)

    step = 0
    is_instance_dataset = (config.dataset.name == 'tas_instance')

    # Effective batch size via gradient accumulation
    accum_steps = max(1, getattr(config.train, 'batch_size', 1))
    optimizer.zero_grad()

    for idx_sample, batch in enumerate(train_loader):# data(1, 2048, l) , target (1,l)
        idx_sample += 1
        if is_instance_dataset:
            data, frame_target, instances, fname, noise = batch
        else:
            data, frame_target, fname, noise = batch
        if config.augmentation.is_use:
            # Pillar 4 (proposal_imprv_baformer04): keep the cropped target/instances in
            # sync with the cropped data. The previous code called `augment_crop` but
            # discarded its cropped target (kept using the un-cropped `frame_target`),
            # and never cropped `instances` at all -- making the augmentation flag a
            # silent no-op (or worse, misaligned) for the instance dataset.
            if is_instance_dataset:
                data, frame_target, instances = augment_crop_instances(data, frame_target, instances)
            else:
                data, frame_target = augment_crop(data, frame_target)

        if config.dataset.noise_weight is not None and config.dataset.noise_weight > 0:
            data = data + float(config.dataset.noise_weight) * torch.randn_like(data, dtype=torch.float32)
        data = data.to(device, non_blocking=config.train.dataloader.non_blocking)
        frame_target = frame_target.to(device, non_blocking=config.train.dataloader.non_blocking).long() #[1, L] frame-wise
        bd_sigma = getattr(config.dataset, 'boundary_sigma', None)
        if is_instance_dataset:
            instances = instances.to(device, non_blocking=config.train.dataloader.non_blocking).long()
            targets = prepare_target_from_instances(instances, data.shape[-1], config.dataset.guassian_sigma, boundary_sigma=bd_sigma, frame_target=frame_target)
        else:
            targets = prepare_target(frame_target, config.dataset.guassian_sigma, boundary_sigma=bd_sigma)

        outputs = model(data)

        ##-----multi-query: instance, ordinal, categorical
        weight_dict = {"loss_ce": config.model.ce_weight,
                       "loss_mask": config.model.mask_weight,
                       "loss_dice": config.model.dice_weight,
                       "loss_bd": config.model.bd_weight,
                       "loss_mask_tv": getattr(config.model, 'mask_tv_weight', 0.15),
                       "loss_contra": getattr(config.model, 'contra_weight', 0.20),
                       "loss_repulse": getattr(config.model, 'repulse_weight', 0.15),
                       "loss_enc_ce": getattr(config.model, 'enc_ce_weight', 0.3),
                       "loss_enc_smooth": getattr(config.model, 'enc_smooth_weight', 0.15)}
        if config.model.action_seg.transformer_decoder.deep_supervision:
            dec_layers = config.model.action_seg.transformer_decoder.dec_layers
            aux_discount = getattr(config.model, 'aux_discount', 0.4)
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v * aux_discount for k, v in weight_dict.items() if k not in ["loss_bd", "loss_enc_ce", "loss_enc_smooth", "loss_contra", "loss_repulse"]})
            weight_dict.update(aux_weight_dict)

        if config.model.pose_weight_single != 0:
            pos_weight_info = (config.dataset.name, config.dataset.dataset_dir, fname, config.model.pose_weight_single)
        else:
            pos_weight_info = None

        losses = criterion(outputs, targets, pos_weight_info)

        # Scale by weight_dict and separate final-layer components vs aux
        weighted_losses = {}
        aux_loss_val = 0.0
        for k in list(losses.keys()):
            if k in weight_dict:
                wv = losses[k] * weight_dict[k]
                weighted_losses[k] = wv
                if any(k.endswith(f"_{i}") for i in range(20)):
                    aux_loss_val += wv.item()
            else:
                losses.pop(k)

        loss = sum(weighted_losses.values())

        # Extract weighted final layer components
        w_ce = weighted_losses.get("loss_ce", torch.tensor(0.0)).item()
        w_mask = weighted_losses.get("loss_mask", torch.tensor(0.0)).item()
        w_dice = weighted_losses.get("loss_dice", torch.tensor(0.0)).item()
        w_bd = weighted_losses.get("loss_bd", torch.tensor(0.0)).item()
        w_contra = weighted_losses.get("loss_contra", torch.tensor(0.0)).item()
        w_repulse = weighted_losses.get("loss_repulse", torch.tensor(0.0)).item()
        w_enc = weighted_losses.get("loss_enc_ce", torch.tensor(0.0)).item()
        w_smooth = weighted_losses.get("loss_enc_smooth", torch.tensor(0.0)).item()
        w_tv = weighted_losses.get("loss_mask_tv", torch.tensor(0.0)).item()

        loss_ce_meter.update(w_ce, 1)
        loss_mask_meter.update(w_mask, 1)
        loss_dice_meter.update(w_dice, 1)
        loss_bd_meter.update(w_bd, 1)
        loss_contra_meter.update(w_contra, 1)
        loss_repulse_meter.update(w_repulse, 1)
        loss_enc_meter.update(w_enc, 1)
        loss_smooth_meter.update(w_smooth, 1)
        loss_tv_meter.update(w_tv, 1)
        loss_aux_meter.update(aux_loss_val, 1)

        # Scale loss by accumulation steps and backward
        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        # Step optimizer every accum_steps or on the last sample
        if (idx_sample % accum_steps == 0) or (idx_sample == len(train_loader)):
            if getattr(config.train, 'clip_grad', 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        # ------inference (honoring config.test.infer_mode)
        infer_mode = getattr(config.test, 'infer_mode', 'energy_fusion')
        min_dur = getattr(config.dataset, 'min_duration', 7)
        th_t = getattr(config.dataset, 'theta_t', 7)
        prom = getattr(config.dataset, 'peak_prominence', 0.035)
        if infer_mode == 'energy_fusion':
            ins_seg_pred = inference_energy_fusion(outputs, config.dataset.threshold, min_duration=min_dur, theta_t=th_t, prominence=prom)
        elif infer_mode == 'snap_dual':
            ins_seg_pred = inference_snap_dual(outputs, config.dataset.threshold, min_duration=min_dur, theta_t=th_t)
        elif infer_mode == 'window_voting':
            ins_seg_pred = inference_window_voting(outputs, config.dataset.threshold, min_duration=min_dur, prominence=prom, theta_t=th_t)
        elif infer_mode == 'query_dominance':
            ins_seg_pred = inference_query_dominance(outputs, config.dataset.threshold, min_duration=min_dur, theta_t=th_t)
        elif infer_mode == 'semantic':
            ins_seg_pred = inference(outputs)
        else:
            ins_seg_pred = inference_bd_peak(outputs, config.dataset.threshold)
        seg_pred = ins_seg_pred.softmax(dim=-1)

        #------compute metrics
        metric_results = meter_dict.get_update_metric(config.dataset.name, seg_pred, frame_target, loss, ins_seg_pred)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step += 1
        global_step += 1

        if config.train.use_tensorboard:
            tensorboard_writer2.add_scalar('Train/StepLoss', loss.item(), global_step)

        if step % config.train.log_period == 0 or idx_sample == len(train_loader):
            loss_meter = metric_results['loss_meter']
            acc_meter = metric_results['acc_meter']
            edit_meter = metric_results['edit_meter']
            f1_meter = metric_results['f1_meter']
            ins_acc_meter = metric_results['ins_acc_meter']

            total = math.ceil(len(train_loader) / accum_steps)
            current_step = math.ceil(idx_sample / accum_steps)
            logger.info(
                f'Epoch {epoch} '
                f'Step {current_step}/{total} '
                f'lr {scheduler.get_last_lr()[0]:.6f} '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                f'[CE: {loss_ce_meter.avg:.3f}, Mask: {loss_mask_meter.avg:.3f}, Dice: {loss_dice_meter.avg:.3f}, BD: {loss_bd_meter.avg:.3f}, Contra: {loss_contra_meter.avg:.3f}, Repulse: {loss_repulse_meter.avg:.3f}, Enc: {loss_enc_meter.avg:.3f}, Smooth: {loss_smooth_meter.avg:.3f}, TV: {loss_tv_meter.avg:.3f}, Aux: {loss_aux_meter.avg:.3f}] '
                f'acc% {acc_meter.val*100:.2f} ({acc_meter.avg*100:.2f}) '
                f'ins_acc% {ins_acc_meter.val * 100:.2f} ({ins_acc_meter.avg * 100:.2f}) '
                f'edit {edit_meter.val:.2f}({edit_meter.avg:.2f}) '
                f'f1@10 {f1_meter.val[0]:.2f}({f1_meter.avg[0]:.2f}) '
                f'f1@25 {f1_meter.val[1]:.2f}({f1_meter.avg[1]:.2f}) '
                f'f1@50 {f1_meter.val[2]:.2f}({f1_meter.avg[2]:.2f}) '
            )

    loss_meter = metric_results['loss_meter']
    acc_meter = metric_results['acc_meter']
    edit_meter = metric_results['edit_meter']
    f1_meter = metric_results['f1_meter']
    f1_mean = sum(f1_meter.avg) / len(f1_meter.avg) if len(f1_meter.avg) > 0 else 0.0
    ins_acc_meter = metric_results['ins_acc_meter']

    if config.train.use_tensorboard:
        tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_CE', loss_ce_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_Mask', loss_mask_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_Dice', loss_dice_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_BD', loss_bd_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_Contra', loss_contra_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_Repulse', loss_repulse_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_Enc', loss_enc_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_Smooth', loss_smooth_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_TV', loss_tv_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Loss_Aux', loss_aux_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch)
        tensorboard_writer.add_scalar('Train/Acc', acc_meter.avg * 100, epoch)
        tensorboard_writer.add_scalar('Train/Ins_Acc', ins_acc_meter.avg * 100, epoch)
        tensorboard_writer.add_scalar('Train/Edit', edit_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/F1@10', f1_meter.avg[0], epoch)
        tensorboard_writer.add_scalar('Train/F1@25', f1_meter.avg[1], epoch)
        tensorboard_writer.add_scalar('Train/F1@50', f1_meter.avg[2], epoch)
        tensorboard_writer.add_scalar('Train/F1_Mean', f1_mean, epoch)

    scheduler.step()

    return {
        'loss': loss_meter.avg,
        'loss_ce': loss_ce_meter.avg,
        'loss_mask': loss_mask_meter.avg,
        'loss_dice': loss_dice_meter.avg,
        'loss_bd': loss_bd_meter.avg,
        'loss_enc': loss_enc_meter.avg,
        'loss_smooth': loss_smooth_meter.avg,
        'loss_aux': loss_aux_meter.avg,
        'acc': acc_meter.avg * 100,
        'ins_acc': ins_acc_meter.avg * 100,
        'edit': edit_meter.avg,
        'f1_10': f1_meter.avg[0],
        'f1_25': f1_meter.avg[1],
        'f1_50': f1_meter.avg[2],
        'f1_mean': f1_mean,
    }

@torch.no_grad()
def validate(epoch, config, model,  val_loader, val_record, logger, tensorboard_writer):
    device = torch.device(config.device)
    meter_dict = Meter_dict()
    criterion = SetCriterion_bd(config)


    new_edit_meter = AverageMeter()
    before_edit_meter = AverageMeter()

    # Track individual loss components
    loss_ce_meter = AverageMeter()
    loss_mask_meter = AverageMeter()
    loss_dice_meter = AverageMeter()
    loss_bd_meter = AverageMeter()
    loss_contra_meter = AverageMeter()
    loss_repulse_meter = AverageMeter()
    loss_enc_meter = AverageMeter()
    loss_smooth_meter = AverageMeter()
    loss_tv_meter = AverageMeter()
    loss_aux_meter = AverageMeter()

    # Telemetry: Confusion Matrix and Class Names
    num_classes = config.dataset.n_classes
    val_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    classes_json_path = pathlib.Path(config.dataset.dataset_dir) / "classes.json"
    if classes_json_path.exists():
        try:
            import json
            with open(classes_json_path, "r", encoding="utf-8") as f:
                classes_list = json.load(f)
            class_names = [item["class_name"] for item in sorted(classes_list, key=lambda x: x["class_id"])]
        except Exception:
            class_names = [f"Class {i}" for i in range(num_classes)]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Boundary metrics accumulators. Tolerance is now a fixed time window
    # (+/-0.1s, +/-0.25s, +/-0.5s) instead of a fixed +/-3 frame window, since
    # a fixed frame count corresponds to a different real-world time window
    # depending on each video's fps. Converted to a per-video frame tolerance
    # via that video's fps (see `_bd_tol_frames` below). Keys kept as
    # `bd_tp_3`/etc. (suffix now means "tol index", not "3 frames") to avoid
    # touching every downstream consumer; see BOUNDARY_TOL_SECONDS for the
    # actual meaning of each tolerance.
    BOUNDARY_TOL_SECONDS = (0.1, 0.25, 0.5)
    bd_tp = {tol: 0 for tol in BOUNDARY_TOL_SECONDS}
    bd_fp = {tol: 0 for tol in BOUNDARY_TOL_SECONDS}
    bd_fn = {tol: 0 for tol in BOUNDARY_TOL_SECONDS}
    _ann_fps_cache = {}

    def _get_video_fps(fname_str):
        """Look up a video's fps from its annotation json (cached per-call).
        Falls back to 1 frame of tolerance worth of fps=25 if unavailable (so
        a lookup failure degrades to a sane fixed-frame tolerance rather than
        crashing boundary evaluation)."""
        if fname_str in _ann_fps_cache:
            return _ann_fps_cache[fname_str]
        fps_val = 25.0
        try:
            ann_json_path = pathlib.Path(config.dataset.dataset_dir) / 'annotations' / f'{fname_str}.json'
            if ann_json_path.exists():
                import json as _json
                with open(ann_json_path, 'r', encoding='utf-8') as _f:
                    fps_val = float(_json.load(_f).get('fps', fps_val))
        except Exception:
            pass
        _ann_fps_cache[fname_str] = fps_val
        return fps_val

    model.eval()

    is_instance_dataset = (config.dataset.name == 'tas_instance')

    for step, batch in enumerate(val_loader): # batchsize=1, one by one
        #----- data process
        if is_instance_dataset:
            data, frame_target, instances, fname = batch
        else:
            data, frame_target, fname = batch
        data = data.to(device, non_blocking=config.validation.dataloader.non_blocking)
        frame_target = frame_target.to(device, non_blocking=config.validation.dataloader.non_blocking).long()
        bd_sigma = getattr(config.dataset, 'boundary_sigma', None)
        if is_instance_dataset:
            instances = instances.to(device, non_blocking=config.validation.dataloader.non_blocking).long()
            targets = prepare_target_from_instances(instances, frame_target.shape[-1], config.dataset.guassian_sigma, boundary_sigma=bd_sigma, frame_target=frame_target)
        else:
            targets = prepare_target(frame_target, config.dataset.guassian_sigma, boundary_sigma=bd_sigma)

        outputs = model(data)

        ##-------other sample rate
        if config.dataset.sample_rate != 1:
            L = frame_target.shape[-1]
            outputs['pred_masks'] = outputs['pred_masks'].repeat_interleave(config.dataset.sample_rate, dim = -1)[:, :, :L]
            outputs['pred_boundarys'] = F.interpolate(outputs['pred_boundarys'],size = L, mode='linear' )

            aux_outputs_resize = []
            if "aux_outputs" in outputs:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    aux_outputs['pred_masks'] = aux_outputs['pred_masks'].repeat_interleave(config.dataset.sample_rate, dim = -1)[:, :, :L]
                    aux_outputs['pred_boundarys'] = aux_outputs['pred_boundarys'].repeat_interleave(config.dataset.sample_rate, dim = -1)[:, :, :L]
                    aux_outputs_resize.append({'pred_logits':aux_outputs['pred_logits'],
                                                'pred_masks':aux_outputs['pred_masks'],
                                                'pred_boundarys': aux_outputs['pred_boundarys'] })
            outputs['aux_outputs'] = aux_outputs_resize

        ##-----multi-query: instance, ordinal, categorical
        weight_dict = {"loss_ce": config.model.ce_weight,
                       "loss_mask": config.model.mask_weight,
                       "loss_dice": config.model.dice_weight,
                       "loss_bd": config.model.bd_weight,
                       "loss_mask_tv": getattr(config.model, 'mask_tv_weight', 0.15),
                       "loss_contra": getattr(config.model, 'contra_weight', 0.20),
                       "loss_repulse": getattr(config.model, 'repulse_weight', 0.15),
                       "loss_enc_ce": getattr(config.model, 'enc_ce_weight', 0.3),
                       "loss_enc_smooth": getattr(config.model, 'enc_smooth_weight', 0.15)}
        if config.model.action_seg.transformer_decoder.deep_supervision:
            dec_layers = config.model.action_seg.transformer_decoder.dec_layers
            aux_discount = getattr(config.model, 'aux_discount', 0.4)
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v * aux_discount for k, v in weight_dict.items() if k not in ["loss_bd", "loss_enc_ce", "loss_enc_smooth", "loss_contra", "loss_repulse"]})
            weight_dict.update(aux_weight_dict)

        losses = criterion(outputs, targets)
        weighted_losses = {}
        aux_loss_val = 0.0
        for k in list(losses.keys()):
            if k in weight_dict:
                wv = losses[k] * weight_dict[k]
                weighted_losses[k] = wv
                if any(k.endswith(f"_{i}") for i in range(20)):
                    aux_loss_val += wv.item()
            else:
                losses.pop(k)

        loss = sum(weighted_losses.values())

        w_ce = weighted_losses.get("loss_ce", torch.tensor(0.0)).item()
        w_mask = weighted_losses.get("loss_mask", torch.tensor(0.0)).item()
        w_dice = weighted_losses.get("loss_dice", torch.tensor(0.0)).item()
        w_bd = weighted_losses.get("loss_bd", torch.tensor(0.0)).item()
        w_contra = weighted_losses.get("loss_contra", torch.tensor(0.0)).item()
        w_repulse = weighted_losses.get("loss_repulse", torch.tensor(0.0)).item()
        w_enc = weighted_losses.get("loss_enc_ce", torch.tensor(0.0)).item()
        w_smooth = weighted_losses.get("loss_enc_smooth", torch.tensor(0.0)).item()
        w_tv = weighted_losses.get("loss_mask_tv", torch.tensor(0.0)).item()

        loss_ce_meter.update(w_ce, 1)
        loss_mask_meter.update(w_mask, 1)
        loss_dice_meter.update(w_dice, 1)
        loss_bd_meter.update(w_bd, 1)
        loss_contra_meter.update(w_contra, 1)
        loss_repulse_meter.update(w_repulse, 1)
        loss_enc_meter.update(w_enc, 1)
        loss_smooth_meter.update(w_smooth, 1)
        loss_tv_meter.update(w_tv, 1)
        loss_aux_meter.update(aux_loss_val, 1)

        #------inference (honoring config.test.infer_mode)
        infer_mode = getattr(config.test, 'infer_mode', 'energy_fusion')
        min_dur = getattr(config.dataset, 'min_duration', 7)
        th_t = getattr(config.dataset, 'theta_t', 7)
        prom = getattr(config.dataset, 'peak_prominence', 0.035)
        if infer_mode == 'energy_fusion':
            ins_seg_pred = inference_energy_fusion(outputs, config.dataset.threshold, min_duration=min_dur, theta_t=th_t, prominence=prom)
        elif infer_mode == 'snap_dual':
            ins_seg_pred = inference_snap_dual(outputs, config.dataset.threshold, min_duration=min_dur, theta_t=th_t)
        elif infer_mode == 'window_voting':
            ins_seg_pred = inference_window_voting(outputs, config.dataset.threshold, min_duration=min_dur, prominence=prom, theta_t=th_t)
        elif infer_mode == 'query_dominance':
            ins_seg_pred = inference_query_dominance(outputs, config.dataset.threshold, min_duration=min_dur, theta_t=th_t)
        elif infer_mode == 'semantic':
            ins_seg_pred = inference(outputs)
        else:
            ins_seg_pred = inference_bd_peak(outputs, config.dataset.threshold)
        seg_pred = ins_seg_pred.softmax(dim=-1)

        #------compute metrics
        metric_results = meter_dict.get_update_metric(config.dataset.name, seg_pred, frame_target, loss, ins_seg_pred)

        #------telemetry: accumulate confusion matrix & boundary metrics
        pred_cls_np = seg_pred.argmax(dim=-1).squeeze().cpu().numpy()
        gt_cls_np = frame_target.squeeze().cpu().numpy()
        for p, g in zip(pred_cls_np, gt_cls_np):
            if 0 <= g < num_classes and 0 <= p < num_classes:
                val_cm[g, p] += 1

        if "pred_boundarys" in outputs:
            bd_prob = outputs["pred_boundarys"][0].sigmoid().squeeze().cpu().numpy()
            prom = getattr(config.dataset, 'peak_prominence', 0.065)
            min_dist = getattr(config.dataset, 'min_distance', 8)
            pred_bd_peaks = extract_peaks_nms(
                bd_prob,
                threshold=config.dataset.threshold,
                min_distance=min_dist,
                prominence=prom,
            )
            if is_instance_dataset and 'instances' in locals() and instances.numel() > 3:
                ins_arr = instances.squeeze(0).cpu().numpy() if instances.dim() > 1 else instances.cpu().numpy()
                if ins_arr.ndim == 2 and ins_arr.shape[0] > 1:
                    gt_bd_frames = ins_arr[1:, 0].tolist()
                else:
                    gt_bd_frames = (np.where(gt_cls_np[1:] != gt_cls_np[:-1])[0] + 1).tolist()
            else:
                gt_bd_frames = (np.where(gt_cls_np[1:] != gt_cls_np[:-1])[0] + 1).tolist()

            video_fps = _get_video_fps(fname[0] if isinstance(fname, (list, tuple)) else fname)
            for tol_s in BOUNDARY_TOL_SECONDS:
                tol_frames = tol_s * video_fps
                matched_gt = set()
                for pb in pred_bd_peaks:
                    matched = False
                    for gi, gb in enumerate(gt_bd_frames):
                        if abs(pb - gb) <= tol_frames and gi not in matched_gt:
                            matched_gt.add(gi)
                            matched = True
                            break
                    if matched:
                        bd_tp[tol_s] += 1
                    else:
                        bd_fp[tol_s] += 1
                bd_fn[tol_s] += (len(gt_bd_frames) - len(matched_gt))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    loss_avg = metric_results['loss_meter'].avg
    acc_avg = metric_results['acc_meter'].avg
    edit_avg = metric_results['edit_meter'].avg
    f1_avg = metric_results['f1_meter'].avg
    ins_acc_avg = metric_results['ins_acc_meter'].avg

    if_update = val_record.update_record(epoch, acc_avg, edit_avg, f1_avg)

    acc_max = val_record.acc_max
    edit_max = val_record.edit_max
    f1_max = val_record.f1_max
    f1_mean_max = val_record.f1_mean_max
    f1_mean_avg = sum(f1_avg) / len(f1_avg) if len(f1_avg) > 0 else 0.0

    # Boundary metrics calculation, one set of Prec/Rec/F1 per time tolerance.
    bd_prec = {}
    bd_rec = {}
    bd_f1 = {}
    for tol_s in BOUNDARY_TOL_SECONDS:
        bd_prec[tol_s] = (bd_tp[tol_s] / max(1, bd_tp[tol_s] + bd_fp[tol_s])) * 100
        bd_rec[tol_s] = (bd_tp[tol_s] / max(1, bd_tp[tol_s] + bd_fn[tol_s])) * 100
        bd_f1[tol_s] = (2 * bd_prec[tol_s] * bd_rec[tol_s] / max(1e-8, bd_prec[tol_s] + bd_rec[tol_s]))

    # Backward-compat aliases: existing consumers (telemetry export, log line,
    # tensorboard tags, val_metrics dict) expect a single "bd_f1_3"-style
    # headline boundary metric. Use the middle tolerance (+/-0.25s) as that
    # headline value; the full per-tolerance breakdown is also exposed below
    # under 'bd_f1_by_tol'/'bd_prec_by_tol'/'bd_rec_by_tol'.
    _headline_tol = 0.25
    bd_prec_3 = bd_prec[_headline_tol]
    bd_rec_3 = bd_rec[_headline_tol]
    bd_f1_3 = bd_f1[_headline_tol]
    bd_tp_3, bd_fp_3, bd_fn_3 = bd_tp[_headline_tol], bd_fp[_headline_tol], bd_fn[_headline_tol]

    # Per-class summary
    support = val_cm.sum(axis=1)
    pred_counts = val_cm.sum(axis=0)
    true_pos = np.diag(val_cm)
    per_class = []
    for k in range(num_classes):
        prec_k = float(true_pos[k] / max(1, pred_counts[k])) * 100
        rec_k = float(true_pos[k] / max(1, support[k])) * 100
        f1_k = (2 * prec_k * rec_k / max(1e-8, prec_k + rec_k))
        per_class.append({
            "class_id": k,
            "class_name": class_names[k],
            "support_frames": int(support[k]),
            "precision": round(prec_k, 2),
            "recall": round(rec_k, 2),
            "f1": round(f1_k, 2),
        })

    logger.info(
        f'----------------------------------------------------------------------------------------------------------------------------------------------------\n'
        f'Val {epoch:3d} | '
        f'loss {loss_avg:.4f} [CE: {loss_ce_meter.avg:.3f}, Mask: {loss_mask_meter.avg:.3f}, Dice: {loss_dice_meter.avg:.3f}, BD: {loss_bd_meter.avg:.3f}, Contra: {loss_contra_meter.avg:.3f}, Repulse: {loss_repulse_meter.avg:.3f}, Enc: {loss_enc_meter.avg:.3f}, Smooth: {loss_smooth_meter.avg:.3f}, TV: {loss_tv_meter.avg:.3f}, Aux: {loss_aux_meter.avg:.3f}] | '
        f'acc% {acc_avg*100:.2f} | '
        f'edit {edit_avg:.2f} | '
        f'f1@10/25/50: {f1_avg[0]:.2f}/{f1_avg[1]:.2f}/{f1_avg[2]:.2f} (Mean: {f1_mean_avg:.2f}) | '
        f'Boundary F1@0.1s/0.25s/0.5s: {bd_f1[0.1]:.2f}/{bd_f1[0.25]:.2f}/{bd_f1[0.5]:.2f}% '
        f'(Prec: {bd_prec[0.1]:.2f}/{bd_prec[0.25]:.2f}/{bd_prec[0.5]:.2f}%, '
        f'Rec: {bd_rec[0.1]:.2f}/{bd_rec[0.25]:.2f}/{bd_rec[0.5]:.2f}%) | '
        f'Best Composite: {getattr(val_record, "composite_max", 0.0):.2f} (F1: {f1_mean_max:.2f}, Edit: {edit_max:.2f}, Acc: {acc_max*100:.2f}%)'
    )

    if config.train.use_tensorboard:
        tensorboard_writer.add_scalar('Val/Loss', loss_avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_CE', loss_ce_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_Mask', loss_mask_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_Dice', loss_dice_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_BD', loss_bd_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_Contra', loss_contra_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_Repulse', loss_repulse_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_Enc', loss_enc_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_Smooth', loss_smooth_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_TV', loss_tv_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Loss_Aux', loss_aux_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Acc', acc_avg * 100, epoch)
        tensorboard_writer.add_scalar('Val/Ins_Acc', ins_acc_avg * 100, epoch)
        tensorboard_writer.add_scalar('Val/Edit', edit_avg, epoch)
        tensorboard_writer.add_scalar('Val/F1@10', f1_avg[0], epoch)
        tensorboard_writer.add_scalar('Val/F1@25', f1_avg[1], epoch)
        tensorboard_writer.add_scalar('Val/F1@50', f1_avg[2], epoch)
        tensorboard_writer.add_scalar('Val/F1_Mean', f1_mean_avg, epoch)
        tensorboard_writer.add_scalar('Val/F1_Mean_Max', f1_mean_max, epoch)
        tensorboard_writer.add_scalar('Val/Acc_Max', acc_max * 100, epoch)
        for tol_s in BOUNDARY_TOL_SECONDS:
            tensorboard_writer.add_scalar(f'Val/Boundary_F1@{tol_s}s', bd_f1[tol_s], epoch)
            tensorboard_writer.add_scalar(f'Val/Boundary_Prec@{tol_s}s', bd_prec[tol_s], epoch)
            tensorboard_writer.add_scalar(f'Val/Boundary_Rec@{tol_s}s', bd_rec[tol_s], epoch)
        for pc in per_class:
            cname = pc["class_name"].replace("/", "_").replace(" ", "_")
            tensorboard_writer.add_scalar(f'Val_Class_F1/{cname}', pc["f1"], epoch)

    val_metrics = {
        'loss': loss_avg,
        'loss_ce': loss_ce_meter.avg,
        'loss_mask': loss_mask_meter.avg,
        'loss_dice': loss_dice_meter.avg,
        'loss_bd': loss_bd_meter.avg,
        'loss_contra': loss_contra_meter.avg,
        'loss_enc': loss_enc_meter.avg,
        'loss_smooth': loss_smooth_meter.avg,
        'loss_tv': loss_tv_meter.avg,
        'loss_aux': loss_aux_meter.avg,
        'acc': acc_avg * 100,
        'ins_acc': ins_acc_avg * 100,
        'edit': edit_avg,
        'f1_10': f1_avg[0],
        'f1_25': f1_avg[1],
        'f1_50': f1_avg[2],
        'f1_mean': f1_mean_avg,
        'cm': val_cm,
        'class_names': class_names,
        'per_class': per_class,
        'bd_f1_3': bd_f1_3,
        'bd_prec_3': bd_prec_3,
        'bd_rec_3': bd_rec_3,
        'bd_f1_by_tol': bd_f1,
        'bd_prec_by_tol': bd_prec,
        'bd_rec_by_tol': bd_rec,
    }
    return if_update, val_metrics

def main():
    global global_step

    config = load_config()
    set_seed(config)
    setup_cudnn(config)

    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
                                    size=config.scheduler.epochs)
    out_dir = os.path.join(config.train.output_dir, config.dataset.name, config.model.name, config.model.note, str(config.dataset.split))
    output_dir = pathlib.Path(out_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    if not config.train.resume:
        save_config(config, output_dir / 'config.yaml')
        save_config(get_env_info(config), output_dir / 'env.yaml')
        diff = find_config_diff(config)
        if diff is not None:
            save_config(diff, output_dir / 'config_min.yaml')

    logger = create_logger(name=__name__,
                           distributed_rank=get_rank(),
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    logger.info(get_env_info(config))

    train_loader = create_dataloader(config, is_train=True)
    val_loader = create_dataloader(config, is_train=False)

    model = create_model(config)

    optimizer = create_optimizer(config, model)

    scheduler = create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch = len(train_loader))

    checkpointer = Checkpointer(model,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              save_dir=output_dir,
                              save_to_disk=get_rank() == 0)

    start_epoch = config.train.start_epoch
    scheduler.last_epoch = start_epoch
    if config.train.resume:
        checkpoint_config = checkpointer.resume_or_load('', resume=True)
        global_step = checkpoint_config['global_step']
        start_epoch = checkpoint_config['epoch']
        config.defrost()
        config.merge_from_other_cfg(ConfigNode(checkpoint_config['config']))
        config.freeze()
    elif config.train.checkpoint != '':
        checkpoint = torch.load(config.train.checkpoint, map_location='cpu')
        if isinstance(model,
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])

    if os.path.exists(output_dir / 'epoch_logs'):
        os.removedirs(output_dir / 'epoch_logs')
        print('Remove the epoch_logs successuflly!')
    if os.path.exists(output_dir / 'step_logs'):
        os.removedirs(output_dir / 'step_logs')
        print('Remove the step_logs successuflly!')

    if config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(
            config, output_dir/'logs_epoch', purge_step=config.train.start_epoch +1)
        tensorboard_writer2 = create_tensorboard_writer(
            config, output_dir / 'logs_epoch'/'logs_step', purge_step=global_step + 1)
    else:
        tensorboard_writer = DummyWriter()
        tensorboard_writer2 = DummyWriter()

    val_record = Record_dict()

    history = {
        'epoch': [],
        'train_loss': [],
        'train_loss_ce': [],
        'train_loss_mask': [],
        'train_loss_dice': [],
        'train_loss_bd': [],
        'train_loss_enc': [],
        'train_loss_smooth': [],
        'train_loss_aux': [],
        'val_loss': [],
        'val_loss_ce': [],
        'val_loss_mask': [],
        'val_loss_dice': [],
        'val_loss_bd': [],
        'val_loss_enc': [],
        'val_loss_smooth': [],
        'val_loss_aux': [],
        'train_acc': [],
        'val_acc': [],
        'train_edit': [],
        'val_edit': [],
        'val_f1_10': [],
        'val_f1_25': [],
        'val_f1_50': [],
        'val_f1_mean': [],
    }

    csv_path = output_dir / 'metrics_history.csv'
    with open(csv_path, 'w') as f:
        f.write(
            'epoch,train_loss,val_loss,'
            'train_ce,train_mask,train_dice,train_bd,train_enc,train_smooth,train_aux,'
            'val_ce,val_mask,val_dice,val_bd,val_enc,val_smooth,val_aux,'
            'train_acc,val_acc,train_edit,val_edit,'
            'val_f1_10,val_f1_25,val_f1_50,val_f1_mean\n'
        )

    patience = getattr(config.train, 'early_stopping_patience', 0)
    no_improve_epochs = 0

    for epoch, seed in enumerate(epoch_seeds[start_epoch:], start_epoch):
        epoch += 1
        train_metrics = train(epoch, config, model, optimizer, scheduler, train_loader, logger, tensorboard_writer, tensorboard_writer2)

        val_metrics = None
        if_update = False
        if config.train.val_period > 0 and (epoch % config.train.val_period == 0):
            if_update, val_metrics = validate(epoch, config, model, val_loader, val_record, logger, tensorboard_writer)

            if if_update:
                no_improve_epochs = 0
                checkpoint_config = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'config': config.as_dict(),
                }
                checkpointer.save(f'checkpoint_best', **checkpoint_config)
                save_experiment_artifacts(
                    output_dir=output_dir,
                    config=config,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    val_record=val_record,
                    history=history,
                    early_stopped=False,
                    is_final=False,
                )
            else:
                no_improve_epochs += 1

        tensorboard_writer.flush()
        tensorboard_writer2.flush()

        # Update history & save plots
        history['epoch'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_loss_ce'].append(train_metrics['loss_ce'])
        history['train_loss_mask'].append(train_metrics['loss_mask'])
        history['train_loss_dice'].append(train_metrics['loss_dice'])
        history['train_loss_bd'].append(train_metrics['loss_bd'])
        history['train_loss_enc'].append(train_metrics.get('loss_enc', 0.0))
        history['train_loss_smooth'].append(train_metrics.get('loss_smooth', 0.0))
        history['train_loss_aux'].append(train_metrics['loss_aux'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_edit'].append(train_metrics['edit'])

        if val_metrics is not None:
            history['val_loss'].append(val_metrics['loss'])
            history['val_loss_ce'].append(val_metrics['loss_ce'])
            history['val_loss_mask'].append(val_metrics['loss_mask'])
            history['val_loss_dice'].append(val_metrics['loss_dice'])
            history['val_loss_bd'].append(val_metrics['loss_bd'])
            history['val_loss_enc'].append(val_metrics.get('loss_enc', 0.0))
            history['val_loss_smooth'].append(val_metrics.get('loss_smooth', 0.0))
            history['val_loss_aux'].append(val_metrics['loss_aux'])
            history['val_acc'].append(val_metrics['acc'])
            history['val_edit'].append(val_metrics['edit'])
            history['val_f1_10'].append(val_metrics['f1_10'])
            history['val_f1_25'].append(val_metrics['f1_25'])
            history['val_f1_50'].append(val_metrics['f1_50'])
            history['val_f1_mean'].append(val_metrics['f1_mean'])
        else:
            for k in [
                'val_loss', 'val_loss_ce', 'val_loss_mask', 'val_loss_dice', 'val_loss_bd', 'val_loss_enc', 'val_loss_smooth', 'val_loss_aux',
                'val_acc', 'val_edit', 'val_f1_10', 'val_f1_25', 'val_f1_50', 'val_f1_mean'
            ]:
                history[k].append(None)

        with open(csv_path, 'a') as f:
            v_loss = f"{val_metrics['loss']:.4f}" if val_metrics else ""
            v_ce = f"{val_metrics['loss_ce']:.4f}" if val_metrics else ""
            v_mask = f"{val_metrics['loss_mask']:.4f}" if val_metrics else ""
            v_dice = f"{val_metrics['loss_dice']:.4f}" if val_metrics else ""
            v_bd = f"{val_metrics['loss_bd']:.4f}" if val_metrics else ""
            v_enc = f"{val_metrics.get('loss_enc', 0.0):.4f}" if val_metrics else ""
            v_smooth = f"{val_metrics.get('loss_smooth', 0.0):.4f}" if val_metrics else ""
            v_aux = f"{val_metrics['loss_aux']:.4f}" if val_metrics else ""
            v_acc = f"{val_metrics['acc']:.4f}" if val_metrics else ""
            v_edit = f"{val_metrics['edit']:.4f}" if val_metrics else ""
            v_f1_10 = f"{val_metrics['f1_10']:.4f}" if val_metrics else ""
            v_f1_25 = f"{val_metrics['f1_25']:.4f}" if val_metrics else ""
            v_f1_50 = f"{val_metrics['f1_50']:.4f}" if val_metrics else ""
            v_f1_mean = f"{val_metrics['f1_mean']:.4f}" if val_metrics else ""
            f.write(
                f"{epoch},{train_metrics['loss']:.4f},{v_loss},"
                f"{train_metrics['loss_ce']:.4f},{train_metrics['loss_mask']:.4f},{train_metrics['loss_dice']:.4f},{train_metrics['loss_bd']:.4f},{train_metrics.get('loss_enc', 0.0):.4f},{train_metrics.get('loss_smooth', 0.0):.4f},{train_metrics['loss_aux']:.4f},"
                f"{v_ce},{v_mask},{v_dice},{v_bd},{v_enc},{v_smooth},{v_aux},"
                f"{train_metrics['acc']:.4f},{v_acc},{train_metrics['edit']:.4f},{v_edit},"
                f"{v_f1_10},{v_f1_25},{v_f1_50},{v_f1_mean}\n"
            )

        save_training_curves(history, output_dir / 'training_curves.png')

        # Early stopping check
        if patience and patience > 0 and no_improve_epochs >= patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch}: "
                f"no validation improvement for {no_improve_epochs} consecutive epochs (patience={patience})."
            )
            break

    # Save final experiment summary artifacts
    is_early_stop = bool(patience and patience > 0 and no_improve_epochs >= patience)
    save_experiment_artifacts(
        output_dir=output_dir,
        config=config,
        epoch=epoch,
        val_metrics=val_metrics,
        val_record=val_record,
        history=history,
        early_stopped=is_early_stop,
        is_final=True,
    )
    logger.info(f"Experiment completed. Summary artifacts saved to {output_dir}")

    tensorboard_writer.close()
    tensorboard_writer2.close()

if __name__ =='__main__':
    main()