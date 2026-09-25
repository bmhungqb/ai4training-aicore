import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EfficientGEBD'))
from datasets import build_dataloader
from modeling import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import OnGEBDModel

def smooth_labels(labels, sigma=1.0):
    """
    Turn hard 0/1 boundary labels (B, T) into soft Gaussian-smoothed labels.
    GEBD annotations have inherent human timing jitter, so a single exact
    frame is an overly strict target; nearby frames get partial credit.
    Smoothing only uses a symmetric kernel over the *label* sequence (not
    model inputs), so it does not violate the model's causal inference path.
    """
    if sigma <= 0:
        return labels
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=labels.device, dtype=labels.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.max()  # peak-normalized so exact boundary frame stays label=1
    kernel = kernel.view(1, 1, -1)
    padded = F.pad(labels.unsqueeze(1), (radius, radius), mode="constant", value=0)
    smoothed = F.conv1d(padded, kernel)
    return smoothed.squeeze(1).clamp(0, 1)


def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    """
    Binary focal loss to counter the severe boundary/non-boundary class
    imbalance (boundaries are a tiny minority of frames).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    focal_term = alpha * (1 - p_t) ** gamma
    return (focal_term * bce).mean()


def train_one_epoch(model, dataloader, optimizer, device, lambda_anticipation=0.5,
                     loss_type="focal", label_sigma=1.0, pos_weight=None):
    """
    Train the OnGEBD model for one epoch.
    lambda_anticipation: Weight for the auxiliary anticipation MSE loss.
    loss_type: "focal" (default, handles class imbalance) or "bce".
    label_sigma: std-dev (in frames) for Gaussian label smoothing; 0 disables it.
    pos_weight: optional scalar tensor for BCEWithLogitsLoss pos_weight (only used if loss_type=="bce").
    """
    model.train()
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0
    total_bce = 0
    total_mse = 0

    for batch_idx, batch in enumerate(dataloader):
        # frames shape: (B, T, C, H, W)
        # labels shape: (B, T) - 1 for boundary, 0 for non-boundary
        frames = batch['imgs'].to(device)
        labels = batch['labels'].to(device).float()
        soft_labels = smooth_labels(labels, sigma=label_sigma)

        optimizer.zero_grad()

        # Forward pass
        boundary_logits, anticipation_loss = model(frames)

        # Calculate Boundary Classification Loss
        if loss_type == "focal":
            bce_loss = focal_loss_with_logits(boundary_logits, soft_labels)
        else:
            bce_loss = bce_criterion(boundary_logits, soft_labels)

        # Total Loss = BCE/Focal + lambda * MSE
        loss = bce_loss + lambda_anticipation * anticipation_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_mse += anticipation_loss.item()

    print(f"Epoch Loss: {total_loss/len(dataloader):.4f} | "
          f"{'Focal' if loss_type == 'focal' else 'BCE'}: {total_bce/len(dataloader):.4f} | "
          f"Anticipation MSE: {total_mse/len(dataloader):.4f}")


@torch.no_grad()
def evaluate_boundaries(model, dataloader, device, prob_threshold=0.5, tolerance=2):
    """
    Streaming-appropriate evaluation: precision/recall/F1 with a tolerance
    window (standard for GEBD, since exact-frame matching is too strict
    given human annotation jitter). Uses model.step() to mirror real
    deployment (true causal, frame-by-frame decoding), not the batched
    forward() used for training.
    """
    model.eval()
    tp = fp = fn = 0

    for batch in dataloader:
        frames = batch['imgs'].to(device)
        labels = batch['labels'].to(device)
        B, T = labels.shape

        state = model.init_state(batch_size=B, device=device)
        probs = []
        for t in range(T):
            p, state = model.step(frames[:, t], state)
            probs.append(p)
        probs = torch.stack(probs, dim=1)  # (B, T)
        preds = (probs >= prob_threshold)

        for b in range(B):
            gt_idx = (labels[b] == 1).nonzero(as_tuple=True)[0].tolist()
            pred_idx = preds[b].nonzero(as_tuple=True)[0].tolist()
            matched_gt = set()
            for p_idx in pred_idx:
                match = next((g for g in gt_idx if g not in matched_gt and abs(g - p_idx) <= tolerance), None)
                if match is not None:
                    tp += 1
                    matched_gt.add(match)
                else:
                    fp += 1
            fn += len(gt_idx) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Eval (tolerance={tolerance}f) | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OnGEBD Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Create dummy args for EfficientGEBD's build_dataloader
    class DummyArgs:
        distributed = False
    dummy_args = DummyArgs()

    # Configure cfg for SEWING dataset
    cfg.merge_from_list([
        "DATASETS.TRAIN", ["SEWING_train"],
        "DATASETS.TEST", ["SEWING_val"],
        "SOLVER.BATCH_SIZE", args.batch_size,
        "INPUT.END_TO_END", True, # This forces the dataloader to return (B, T, C, H, W) instead of flattening
        "SOLVER.NUM_WORKERS", 4
    ])

    print("Building DataLoaders from EfficientGEBD...")
    train_dataloader = build_dataloader(cfg, dummy_args, cfg.DATASETS.TRAIN, is_train=True)
    val_dataloader = build_dataloader(cfg, dummy_args, cfg.DATASETS.TEST, is_train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OnGEBDModel(feature_dim=2048, hidden_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting OnGEBD Training for {args.epochs} epochs on device: {device}")
    
    best_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_one_epoch(model, train_dataloader, optimizer, device)
        
        precision, recall, f1 = evaluate_boundaries(model, val_dataloader, device)
        
        if f1 > best_f1:
            best_f1 = f1
            print(f"New best F1! Saving checkpoint...")
            torch.save(model.state_dict(), "ongebd_best.pth")
