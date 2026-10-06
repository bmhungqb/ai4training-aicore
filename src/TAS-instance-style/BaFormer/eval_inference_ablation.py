#!/usr/bin/env python3
"""Offline Inference Strategy Ablation Tool for BaFormer.

Evaluates multiple post-processing / inference decoding strategies on validation clips
using a single forward pass over an existing checkpoint (e.g. Run 7 checkpoint_best.pth).

Strategies compared:
1. window_voting (Run 7 default): threshold=0.15, prominence=0.035, min_duration=4
2. query_dominance (Run 5 default): threshold=0.20, min_duration=4, theta_t=5
3. snap_dual (Pillar 2 default): threshold=0.25, min_duration=8, window_radius=2, theta_t=8
4. snap_dual parameter grid (threshold in {0.20, 0.25, 0.30}, min_duration in {6, 8, 10})

Usage:
    python eval_inference_ablation.py
    python eval_inference_ablation.py --checkpoint experiments/tas_instance/bk_fde_tde/final_exp07/1/checkpoint_best.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

baformer_root = Path(__file__).resolve().parent
if str(baformer_root) not in sys.path:
    sys.path.insert(0, str(baformer_root))

import numpy as np
import torch
import torch.nn.functional as F

from action_segmentation import (
    create_model,
    get_default_config,
    update_config,
)
from action_segmentation.datasets import create_instance_dataset
from action_segmentation.utils.metrics import (
    accuracy,
    edit_score,
    f_score,
)
from main import (
    inference_energy_fusion,
    inference_snap_dual,
    inference_window_voting,
    inference_query_dominance,
    inference_bd_peak,
    inference,
)


def compute_f1_tuple(f_scores):
    tp, fp, fn = f_scores
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    f1 = torch.nan_to_num(f1) * 100.0
    return (float(f1[0]), float(f1[1]), float(f1[2]))


def evaluate_decoding_strategy(
    cached_predictions: list[dict],
    strategy_name: str,
    infer_fn,
    num_classes: int = 4,
) -> dict:
    total_acc = 0.0
    total_edit = 0.0
    total_f_scores = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
    total_cuts = 0
    gt_total_cuts = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    n_samples = len(cached_predictions)

    for item in cached_predictions:
        outputs = item['outputs']
        frame_target = item['frame_target']  # LongTensor [L]
        gt_cls = frame_target.cpu().numpy()
        L = len(gt_cls)

        # Count GT cuts
        gt_cuts = np.where(gt_cls[1:] != gt_cls[:-1])[0] + 1
        gt_total_cuts += len(gt_cuts)

        # Run inference function
        ins_seg_pred = infer_fn(outputs)  # [1, C, L]
        seg_pred = ins_seg_pred.softmax(dim=-1)
        pred_cls = seg_pred.argmax(dim=-1).squeeze(0).cpu().numpy()

        # Count predicted cuts
        pred_cuts = np.where(pred_cls[1:] != pred_cls[:-1])[0] + 1
        total_cuts += len(pred_cuts)

        # Accuracy & Edit Score
        acc_val = float(accuracy(seg_pred, frame_target.unsqueeze(0)))
        edit_val = float(edit_score(seg_pred, frame_target.unsqueeze(0)))
        f_scores_sample = f_score(seg_pred, frame_target.unsqueeze(0))

        total_acc += acc_val
        total_edit += edit_val
        for i in range(3):
            total_f_scores[i] += f_scores_sample[i].cpu()

        for g, p in zip(gt_cls, pred_cls):
            if 0 <= g < num_classes and 0 <= p < num_classes:
                cm[g, p] += 1

    mean_acc = (total_acc / n_samples) * 100.0
    mean_edit = total_edit / n_samples
    f1_tuple = compute_f1_tuple(total_f_scores)
    f1_mean = sum(f1_tuple) / 3.0
    composite = 0.4 * f1_mean + 0.3 * mean_edit + 0.3 * mean_acc

    # Per-class F1
    support = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    true_pos = np.diag(cm)
    per_class_f1 = []
    for k in range(num_classes):
        prec_k = float(true_pos[k] / max(1, pred_counts[k]))
        rec_k = float(true_pos[k] / max(1, support[k]))
        f1_k = float(2 * prec_k * rec_k / max(1e-8, prec_k + rec_k)) * 100.0
        per_class_f1.append(f1_k)

    return {
        "strategy": strategy_name,
        "acc": mean_acc,
        "edit": mean_edit,
        "f1_10": f1_tuple[0],
        "f1_25": f1_tuple[1],
        "f1_50": f1_tuple[2],
        "f1_mean": f1_mean,
        "composite": composite,
        "c0_f1": per_class_f1[0],
        "c1_f1": per_class_f1[1],
        "c2_f1": per_class_f1[2],
        "c3_f1": per_class_f1[3],
        "total_cuts": total_cuts,
        "gt_total_cuts": gt_total_cuts,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation of Inference Strategies on BaFormer")
    parser.add_argument("--config", type=str, default="configs/tas_instance.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/tas_instance/bk_fde_tde/final_exp13/1/checkpoint_best.pth",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading config from {args.config}...")
    cfg = get_default_config()
    cfg.merge_from_file(args.config, allow_unsafe=True)
    cfg = update_config(cfg)
    cfg.device = args.device

    print(f"Loading checkpoint from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        # Check alternative locations
        alt_paths = [
            "experiments/tas_instance/bk_fde_tde/final_exp12/1/checkpoint_best.pth",
            "experiments/tas_instance/bk_fde_tde/final_exp06/1/checkpoint_best.pth",
            "experiments/tas_instance/bk_fde_tde/final_exp05/1/checkpoint_best.pth",
            "experiments/tas_instance/bk_fde_tde/final/1/checkpoint_best.pth",
        ]
        found = None
        for ap in alt_paths:
            if os.path.exists(ap):
                found = ap
                break
        if found:
            print(f"Note: {args.checkpoint} not found. Falling back to {found}")
            args.checkpoint = found
        else:
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    model = create_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print("Building validation dataloader...")
    val_dataset = create_instance_dataset(cfg, is_train=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"Running single forward pass over {len(val_loader)} validation clips...")
    cached_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            data, frame_target, instances, fname = batch
            data = data.to(device)
            outputs = model(data)
            # Move needed tensors to CPU/device for repeated inference
            pred_logits = outputs["pred_logits"].detach()
            pred_masks = outputs["pred_masks"].detach()
            pred_bd = outputs["pred_boundarys"].detach() if "pred_boundarys" in outputs else None

            cached_predictions.append({
                "outputs": {
                    "pred_logits": pred_logits,
                    "pred_masks": pred_masks,
                    "pred_boundarys": pred_bd,
                },
                "frame_target": frame_target.squeeze(0).to(device),
                "fname": fname[0] if isinstance(fname, (list, tuple)) else fname,
            })

    print("\nRunning decoding ablations...")
    strategies = [
        (
            "1. energy_fusion (Run 12 default: th=0.18, min_d=8, th_t=8)",
            lambda outs: inference_energy_fusion(outs, threshold=0.18, min_duration=8, theta_t=8),
        ),
        (
            "2. energy_fusion (th=0.15, min_d=6, th_t=6)",
            lambda outs: inference_energy_fusion(outs, threshold=0.15, min_duration=6, theta_t=6),
        ),
        (
            "3. energy_fusion (th=0.20, min_d=8, th_t=8)",
            lambda outs: inference_energy_fusion(outs, threshold=0.20, min_duration=8, theta_t=8),
        ),
        (
            "4. energy_fusion (th=0.22, min_d=10, th_t=8)",
            lambda outs: inference_energy_fusion(outs, threshold=0.22, min_duration=10, theta_t=8),
        ),
        (
            "5. snap_dual (th=0.20, min_d=6, th_t=6)",
            lambda outs: inference_snap_dual(outs, threshold=0.20, min_duration=6, window_radius=2, theta_t=6),
        ),
        (
            "6. snap_dual (th=0.25, min_d=8, th_t=8)",
            lambda outs: inference_snap_dual(outs, threshold=0.25, min_duration=8, window_radius=2, theta_t=8),
        ),
        (
            "7. window_voting (th=0.15, min_d=4, prom=0.035)",
            lambda outs: inference_window_voting(outs, threshold=0.15, min_duration=4, prominence=0.035, theta_t=4),
        ),
        (
            "8. query_dominance (th=0.20, min_d=4, th_t=5)",
            lambda outs: inference_query_dominance(outs, threshold=0.20, min_duration=4, theta_t=5),
        ),
        (
            "9. semantic (Mask2Former soft argmax, no boundary cuts)",
            lambda outs: inference(outs),
        ),
    ]

    results = []
    for name, fn in strategies:
        res = evaluate_decoding_strategy(cached_predictions, name, fn, num_classes=cfg.dataset.n_classes)
        results.append(res)

    print("\n" + "=" * 125)
    print(f"{'Decoding Strategy':<45} | {'Acc %':<7} | {'Edit':<6} | {'F1@10':<6} | {'F1@25':<6} | {'F1@50':<6} | {'F1 Mean':<7} | {'Comp':<6} | {'C2 F1':<6} | {'Ccuts':<6} (GT=177)")
    print("-" * 125)
    for r in results:
        print(
            f"{r['strategy']:<45} | "
            f"{r['acc']:>6.2f}% | "
            f"{r['edit']:>6.2f} | "
            f"{r['f1_10']:>6.2f} | "
            f"{r['f1_25']:>6.2f} | "
            f"{r['f1_50']:>6.2f} | "
            f"{r['f1_mean']:>6.2f}% | "
            f"{r['composite']:>6.2f} | "
            f"{r['c2_f1']:>6.2f}% | "
            f"{r['total_cuts']:>6d}"
        )
    print("=" * 125)

    # Save to json
    out_json = Path(args.checkpoint).parent / "inference_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed ablation results saved to: {out_json}")


if __name__ == "__main__":
    main()
