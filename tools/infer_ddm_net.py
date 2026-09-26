#!/usr/bin/env python3
"""Inference and evaluation runner for DDM-Net in the AI Research Loop.

Loads a trained PyTorch Lightning checkpoint (.ckpt), runs inference on the
validation dataset, and exports:
  - <output_dir>/predictions.json  ({video_id: [timestamps]})
  - <output_dir>/metrics.json      (standardized metrics across 0.25s, 0.5s, 1.0s)
  - <output_dir>/infer.log
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DDM_DIR = REPO_ROOT / "src" / "step_segment" / "DDM-Net"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DDM_DIR))

import torch
import lightning as L
from omegaconf import OmegaConf

from config.config import get_system_defaults, load_file_to_omegaconf
from pl_ddm_datamodule import DDMDataModule
from train_sop_lightning import SOPLightningModule
from tools.eval_step_segment_predictions import load_ground_truth, evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("infer_ddm_net")


def main():
    parser = argparse.ArgumentParser(description="Run DDM-Net inference and export standardized metrics.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DDM_DIR / "config" / "ddm_train_config.yaml"),
        help="Path to ddm_train_config.yaml",
    )
    parser.add_argument("--anno-path", type=str, default=None, help="Override val annotation json path")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save predictions and metrics")
    parser.add_argument("--threshold", type=float, default=0.5, help="NMS threshold (default: 0.5)")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader workers (default: 1)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data/ containing GT step_segments.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = out_dir / "infer.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"=== Starting DDM-Net Inference ===")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output Directory: {out_dir}")

    # Load configuration
    cfg = get_system_defaults()
    if os.path.exists(args.config):
        user_cfg = load_file_to_omegaconf(args.config)
        cfg = OmegaConf.merge(cfg, user_cfg)
    config = OmegaConf.to_container(cfg, resolve=True)

    dataset_cfg = config["dataset_config"]
    if args.anno_path:
        dataset_cfg["val_config"]["anno_path"] = args.anno_path
    dataset_cfg["val_config"]["workers"] = args.num_workers

    # Initialize datamodule
    logger.info("Initializing DataModule...")
    datamodule = DDMDataModule(dataset_config=dataset_cfg)
    datamodule.setup(stage="validate")

    # Load model from checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    model = SOPLightningModule.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        strict=False,
    )
    model.eval()

    # Setup trainer for validation
    trainer = L.Trainer(
        accelerator="gpu" if "cuda" in args.device else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
    )

    logger.info("Running validation forward pass...")
    trainer.validate(model, datamodule=datamodule)

    # Extract NMS predictions
    predictions = getattr(model, "last_nms_result", None)
    if not predictions:
        logger.warning("Could not find last_nms_result on model; checking validation step outputs.")
        predictions = {}

    # Save predictions.json
    pred_path = out_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    logger.info(f"Saved boundary predictions for {len(predictions)} videos -> {pred_path}")

    # Run standardized evaluation against ground truth
    data_dir = Path(args.data_dir).resolve()
    if data_dir.exists():
        logger.info("Running standardized step segment evaluation against ground truth...")
        gt = load_ground_truth(data_dir)
        eval_gt = {k: v for k, v in gt.items() if k in predictions} or gt
        metrics_report = evaluate_predictions(eval_gt, predictions, thresholds=[0.25, 0.5, 1.0], primary_window=0.5)

        metrics_report["model"] = "DDM-Net"
        metrics_report["checkpoint"] = str(args.checkpoint)
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
        logger.info(f"Saved standardized metrics report -> {metrics_path}")

        primary = metrics_report["primary_metrics"]
        logger.info(f"Macro F1@0.5s: {primary.get('macro_f1', 0.0):.4f} | Recall: {primary.get('macro_recall', 0.0):.4f} | Precision: {primary.get('macro_precision', 0.0):.4f}")

    logger.info("=== DDM-Net Inference Complete ===")


if __name__ == "__main__":
    main()
