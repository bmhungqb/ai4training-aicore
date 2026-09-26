"""Standardized Step Segmentation Experiment Runner for the AI Research Loop.

Supports DDM-Net, DiffGEBD, and EfficientGEBD.
Ensures outputs follow the required structure:
  outputs/step_segment/<iter>/<exp_id>/
    ├── run.log
    ├── metrics.json        (Macro F1@0.5s, per-window F1, per-video breakdown)
    ├── predictions.json    ({video_id: [boundary_timestamps]})
    └── checkpoints/        (model_best.pth / best_model.ckpt)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step_segment_runner")


def parse_args():
    parser = argparse.ArgumentParser(description="Step Segment Experiment Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save artifacts and metrics",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "eval_only"],
        default="train",
        help="Execution mode (train or infer or eval_only)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    return parser.parse_args()


def run_ddm_net(config: dict, output_dir: Path, mode: str, num_gpus: int, env: dict):
    ddm_dir = REPO_ROOT / "src" / "step_segment" / "DDM-Net"
    model_cfg = config.get("model_params", {})
    train_cfg = config.get("training_params", {})

    if mode == "train":
        cmd = [
            sys.executable,
            str(ddm_dir / "train_sop_lightning.py"),
            "--output-dir", str(output_dir),
            "--num-gpus", str(num_gpus),
        ]
        # Append parameters from config
        if "backbone" in model_cfg:
            cmd.extend(["--backbone", str(model_cfg["backbone"])])
        if "epochs" in train_cfg:
            cmd.extend(["--epochs", str(train_cfg["epochs"])])
        if "learning_rate" in train_cfg:
            cmd.extend(["--learning-rate", str(train_cfg["learning_rate"])])
        if "aux_loss_weight" in train_cfg:
            cmd.extend(["--aux-loss-weight", str(train_cfg["aux_loss_weight"])])

        logger.info(f"Executing DDM-Net Training: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(ddm_dir), env=env, check=True)

    elif mode == "infer":
        ckpt = config.get("checkpoint") or str(output_dir / "best_model.ckpt")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "infer_ddm_net.py"),
            "--checkpoint", str(ckpt),
            "--output-dir", str(output_dir),
        ]
        logger.info(f"Executing DDM-Net Inference: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def run_diff_gebd(config: dict, output_dir: Path, mode: str, num_gpus: int, env: dict):
    diff_dir = REPO_ROOT / "src" / "step_segment" / "DiffGEBD"
    config_file = config.get("diffgebd_config_file") or (diff_dir / "config" / "sewing_diffgebd_resnet50_chunked.yaml")

    if mode == "train":
        cmd = [
            "torchrun", f"--nproc_per_node={num_gpus}",
            "train.py",
            "--config-file", str(config_file),
            "OUTPUT_DIR", str(output_dir),
        ]
        logger.info(f"Executing DiffGEBD Training: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(diff_dir), env=env, check=True)

        # Run inference on the resulting best model
        weights = output_dir / "model_best.pth"
        if weights.exists():
            infer_cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "infer_diffgebd.py"),
                "--config-file", str(config_file),
                "--weights", str(weights),
                "--out-dir", str(output_dir),
                "--num-gpus", str(num_gpus),
            ]
            logger.info(f"Exporting DiffGEBD predictions: {' '.join(infer_cmd)}")
            subprocess.run(infer_cmd, cwd=str(REPO_ROOT), env=env, check=True)

    elif mode == "infer":
        weights = config.get("checkpoint") or str(output_dir / "model_best.pth")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "infer_diffgebd.py"),
            "--config-file", str(config_file),
            "--weights", str(weights),
            "--out-dir", str(output_dir),
            "--num-gpus", str(num_gpus),
        ]
        logger.info(f"Executing DiffGEBD Inference: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def run_efficient_gebd(config: dict, output_dir: Path, mode: str, num_gpus: int, env: dict):
    eff_dir = REPO_ROOT / "src" / "step_segment" / "EfficientGEBD"
    config_file = config.get("efficientgebd_config_file") or (eff_dir / "config-files" / "sewing_resnet50.yaml")

    if mode == "train":
        cmd = [
            "torchrun", f"--nproc_per_node={num_gpus}",
            "train.py",
            "--config-file", str(config_file),
            "OUTPUT_DIR", str(output_dir),
        ]
        logger.info(f"Executing EfficientGEBD Training: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(eff_dir), env=env, check=True)

        # Run inference to export predictions and metrics
        weights = output_dir / "model_best.pth"
        if weights.exists():
            infer_cmd = [
                sys.executable,
                str(eff_dir / "infer.py"),
                "--config-file", str(config_file),
                "--checkpoint", str(weights),
                "--input", str(REPO_ROOT / "data" / "efficient_gebd_dataset" / "images" / "val"),
                "--gt", str(REPO_ROOT / "data" / "efficient_gebd_dataset" / "val_annotation.pkl"),
                "--output-dir", str(output_dir),
            ]
            logger.info(f"Exporting EfficientGEBD predictions: {' '.join(infer_cmd)}")
            subprocess.run(infer_cmd, cwd=str(eff_dir), env=env, check=True)

    elif mode == "infer":
        weights = config.get("checkpoint") or str(output_dir / "model_best.pth")
        cmd = [
            sys.executable,
            str(eff_dir / "infer.py"),
            "--config-file", str(config_file),
            "--checkpoint", str(weights),
            "--input", str(REPO_ROOT / "data" / "efficient_gebd_dataset" / "images" / "val"),
            "--gt", str(REPO_ROOT / "data" / "efficient_gebd_dataset" / "val_annotation.pkl"),
            "--output-dir", str(output_dir),
        ]
        logger.info(f"Executing EfficientGEBD Inference: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(eff_dir), env=env, check=True)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_id = config.get("experiment_id", "exp_unnamed")
    track = config.get("track", "step_segment")
    iter_id = config.get("iteration_id", "iter_01")
    model_type = config.get("model_type", "efficient_gebd").lower()

    output_dir = args.output_dir or (REPO_ROOT / "outputs" / track / iter_id / exp_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 65)
    logger.info(f" Starting Step Segmentation Experiment: {exp_id}")
    logger.info(f" Model: {model_type} | Mode: {args.mode}")
    logger.info(f" Output Directory: {output_dir}")
    logger.info("=" * 65)

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    if model_type in ["ddm_net", "ddm"]:
        run_ddm_net(config, output_dir, args.mode, args.num_gpus, env)
    elif model_type in ["diff_gebd", "diffgebd"]:
        run_diff_gebd(config, output_dir, args.mode, args.num_gpus, env)
    elif model_type in ["efficient_gebd", "efficientgebd"]:
        run_efficient_gebd(config, output_dir, args.mode, args.num_gpus, env)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected: ddm_net, diff_gebd, efficient_gebd.")

    elapsed = time.time() - start_time
    logger.info(f"Run completed in {elapsed:.2f}s")

    # Post-run check: verify metrics.json exists
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        primary = metrics_data.get("primary_metrics", {})
        logger.info("\n" + "=" * 65)
        logger.info(f" FINAL METRICS SUMMARY: {exp_id} ({model_type})")
        logger.info(f" Macro F1@0.5s:    {primary.get('macro_f1', 0.0) * 100:.2f}%")
        logger.info(f" Macro Recall@0.5s: {primary.get('macro_recall', 0.0) * 100:.2f}%")
        logger.info(f" Macro Prec@0.5s:   {primary.get('macro_precision', 0.0) * 100:.2f}%")
        logger.info(f" F1 @ 0.25s:       {primary.get('f1_at_0_25s', 0.0) * 100:.2f}%")
        logger.info(f" F1 @ 1.00s:       {primary.get('f1_at_1_0s', 0.0) * 100:.2f}%")
        logger.info("=" * 65)
    else:
        logger.warning(f"Notice: metrics.json was not generated in {output_dir}")


if __name__ == "__main__":
    main()
