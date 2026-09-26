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

def find_repo_root(path: Path) -> Path:
    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    return path.resolve().parents[2]


REPO_ROOT = find_repo_root(Path(__file__).resolve())
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, datasets, and preview commands without running heavy compute",
    )
    parser.add_argument(
        "--dry-run-iters",
        type=int,
        default=2,
        help="Number of iterations to probe data loading and train loop during dry-run (default: 2, set 0 to skip probe)",
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


def update_overview_tracker(config: dict, output_dir: Path, metrics_data: dict):
    tracker_path = REPO_ROOT / "experiments" / "step_segment" / "step_segment_overview.md"
    if not tracker_path.exists():
        return

    exp_id = config.get("experiment_id", output_dir.name)
    model_type = config.get("model_type", "efficient_gebd").lower()
    model_display = {
        "ddm_net": "DDM-Net",
        "diff_gebd": "DiffGEBD",
        "efficient_gebd": "EfficientGEBD",
    }.get(model_type, model_type)

    primary = metrics_data.get("primary_metrics", {})
    f1_05 = primary.get("macro_f1", 0.0) * 100
    rec_05 = primary.get("macro_recall", 0.0) * 100
    prec_05 = primary.get("macro_precision", 0.0) * 100
    f1_025 = primary.get("f1_at_0_25s", 0.0) * 100
    f1_10 = primary.get("f1_at_1_0s", 0.0) * 100

    now_str = time.strftime("%Y-%m-%d %H:%M")
    lines = tracker_path.read_text(encoding="utf-8").splitlines()
    updated = False

    new_lines = []
    for line in lines:
        # Check Leaderboard row
        if f"`{exp_id}`" in line and (line.strip().startswith("| - |") or line.strip().startswith("| 1 |") or line.strip().startswith("| 2 |") or line.strip().startswith("| 3 |")):
            backbone = config.get("model_params", {}).get("backbone", "ResNet-50")
            status = "Completed" if f1_05 > 0 else "Evaluated"
            new_lines.append(f"| - | **{model_display}** | [`{exp_id}`]({model_type}/{config.get('iteration_id', 'iter_01')}/{exp_id}/) | {backbone} | **{f1_05:.2f}%** | {rec_05:.2f}% | {prec_05:.2f}% | {f1_025:.2f}% | {f1_10:.2f}% | {status} |")
            updated = True
        # Check Chronological Results Log row
        elif f"`{exp_id}`" in line and ("Planned" in line or "Completed" in line or "Pending" in line):
            tested_var = config.get("description", "-")
            config_rel = f"{model_type}/{config.get('iteration_id', 'iter_01')}/{exp_id}/config.yaml"
            new_lines.append(f"| {now_str} | `{model_display}` | `{exp_id}` | [config.yaml]({config_rel}) | {tested_var} | **{f1_05:.2f}%** | {rec_05:.2f}% | {prec_05:.2f}% | Completed Run |")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        tested_var = config.get("description", "-")
        config_rel = f"{model_type}/{config.get('iteration_id', 'iter_01')}/{exp_id}/config.yaml"
        new_row = f"| {now_str} | `{model_display}` | `{exp_id}` | [config.yaml]({config_rel}) | {tested_var} | **{f1_05:.2f}%** | {rec_05:.2f}% | {prec_05:.2f}% | Completed Run |"
        sec3_idx = -1
        for idx, line in enumerate(new_lines):
            if "## 3. Model-Specific Experiment Tracks" in line:
                sec3_idx = idx
                break
        if sec3_idx != -1:
            new_lines.insert(sec3_idx - 1, new_row)
        else:
            new_lines.append(new_row)

    try:
        tracker_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info(f"Updated overview tracker: {tracker_path}")
    except Exception as e:
        logger.warning(f"Failed to write to overview tracker: {e}")


def probe_dataset_and_training(model_type: str, config: dict, dry_run_iters: int, mode: str):
    if dry_run_iters <= 0:
        return

    print("\n [2/4] Data Loading & Training Probe:")
    if model_type in ["efficient_gebd", "efficientgebd"]:
        eff_dir = REPO_ROOT / "src" / "step_segment" / "EfficientGEBD"
        dataset_dir = REPO_ROOT / "data" / "efficient_gebd_dataset"
        val_pkl = dataset_dir / "val_annotation.pkl"
        if not (dataset_dir.exists() and val_pkl.exists()):
            print("   ✗ EfficientGEBD dataset missing, skipping data/training probe.")
            return

        try:
            import torch
            if str(eff_dir) not in sys.path:
                sys.path.insert(0, str(eff_dir))

            from modeling.config import _C as eff_cfg_base
            from modeling import build_model
            from datasets import build_dataloader
            from train import make_inputs, make_targets
            from solver import build_optimizer

            config_file = config.get("efficientgebd_config_file") or (eff_dir / "config-files" / "sewing_resnet50.yaml")
            eff_cfg = eff_cfg_base.clone()
            eff_cfg.merge_from_file(str(config_file))
            eff_cfg.SOLVER.BATCH_SIZE = min(eff_cfg.SOLVER.BATCH_SIZE, 2)
            eff_cfg.SOLVER.NUM_WORKERS = 0
            eff_cfg.freeze()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dummy_args = argparse.Namespace(distributed=False, num_gpus=1, local_rank=0)

            split = 'SEWING_train' if mode == 'train' else 'SEWING_val'
            print(f"   → Initializing DataLoader for split '{split}' (device: {device})...")
            loader = build_dataloader(eff_cfg, dummy_args, [split], is_train=(mode == 'train'))
            print(f"   ✓ DataLoader initialized: {len(loader.dataset)} samples ({len(loader)} batches)")

            data_iter = iter(loader)
            batch = next(data_iter)
            samples = make_inputs(batch, device)
            targets = make_targets(eff_cfg, batch, device)

            print(f"   ✓ Data Probe: Successfully loaded 1 batch from disk:")
            print(f"     • Batch imgs shape:   {list(samples['imgs'].shape)} (B, T, C, H, W)")
            print(f"     • Batch labels shape: {list(targets.shape)} (B, T)")

            print(f"   → Instantiating model {eff_cfg.MODEL.NAME} ({eff_cfg.MODEL.BACKBONE.NAME})...")
            model = build_model(eff_cfg).to(device)

            if mode == 'train':
                model.train()
                optimizer = build_optimizer(eff_cfg, [p for p in model.parameters() if p.requires_grad])

                for step in range(1, dry_run_iters + 1):
                    if step > 1:
                        batch = next(data_iter)
                        samples = make_inputs(batch, device)
                        targets = make_targets(eff_cfg, batch, device)

                    optimizer.zero_grad()
                    losses = model(samples, targets)
                    total_loss = sum([w * l for w, l in zip(eff_cfg.MODEL.LOSS_WEIGHT, losses)])
                    total_loss.backward()
                    optimizer.step()
                    print(f"   ✓ Training Probe [Iter {step}/{dry_run_iters}]: Forward + Loss ({total_loss.item():.4f}) + Backward + Optimizer step PASSED")
            else:
                model.eval()
                with torch.no_grad():
                    outputs = model(samples)
                    print(f"   ✓ Inference Probe: Forward pass PASSED (output shape: {list(outputs.shape)})")

            print(f"   ✓ Probe verified successfully! Pipeline is ready.")

        except Exception as e:
            print(f"   ✗ Probe encountered an error: {e}")
            import traceback
            traceback.print_exc()

    elif model_type in ["diff_gebd", "diffgebd"]:
        dataset_dir = REPO_ROOT / "data" / "diff_gebd_dataset"
        chunk_train = dataset_dir / "chunked_train_annotation.json"
        if not (dataset_dir.exists() and chunk_train.exists()):
            print("   ✗ DiffGEBD dataset missing, skipping data/training probe.")
            return

    elif model_type in ["ddm_net", "ddm"]:
        dataset_dir = REPO_ROOT / "data" / "ddm_dataset"
        train_split = dataset_dir / "train_split.json"
        if not (dataset_dir.exists() and train_split.exists()):
            print("   ✗ DDM-Net dataset missing, skipping data/training probe.")
            return


def dry_run_check(config: dict, output_dir: Path, mode: str, num_gpus: int, env: dict, dry_run_iters: int = 2):
    exp_id = config.get("experiment_id", "exp_unnamed")
    model_type = config.get("model_type", "efficient_gebd").lower()
    iter_id = config.get("iteration_id", "iter_01")
    model_cfg = config.get("model_params", {})
    train_cfg = config.get("training_params", {})

    print("\n" + "=" * 70)
    print(" [DRY-RUN] STEP SEGMENTATION PRE-FLIGHT VERIFICATION")
    print("=" * 70)
    print(f" Experiment ID:      {exp_id}")
    print(f" Model Type:         {model_type}")
    print(f" Iteration:          {iter_id}")
    print(f" Mode:               {mode}")
    print(f" Target GPUs:        {num_gpus}")
    print(f" Target Output Dir:  {output_dir}")
    print("-" * 70)

    # 1. Dataset verification
    print(" [1/4] Dataset Verification:")
    if model_type in ["ddm_net", "ddm"]:
        dataset_dir = REPO_ROOT / "data" / "ddm_dataset"
        train_split = dataset_dir / "train_split.json"
        val_split = dataset_dir / "val_split.json"
        if dataset_dir.exists() and train_split.exists() and val_split.exists():
            print(f"   ✓ Found DDM dataset at: {dataset_dir}")
            print(f"   ✓ Train split: {train_split.name} | Val split: {val_split.name}")
        else:
            print(f"   ✗ DDM dataset missing or incomplete at {dataset_dir}")
            print(f"     Run: python tools/process_step_segments.py && python tools/prepare_ddm_dataset.py")

    elif model_type in ["diff_gebd", "diffgebd"]:
        dataset_dir = REPO_ROOT / "data" / "diff_gebd_dataset"
        chunk_train = dataset_dir / "chunked_train_annotation.json"
        chunk_val = dataset_dir / "chunked_val_annotation.json"
        if dataset_dir.exists() and (chunk_train.exists() or (dataset_dir / "train_annotation.json").exists()):
            print(f"   ✓ Found DiffGEBD dataset at: {dataset_dir}")
        else:
            print(f"   ✗ DiffGEBD dataset missing at {dataset_dir}")
            print(f"     Run: python tools/prepare_diff_gebd_dataset.py && python tools/chunk_diff_gebd_dataset.py")

    elif model_type in ["efficient_gebd", "efficientgebd"]:
        dataset_dir = REPO_ROOT / "data" / "efficient_gebd_dataset"
        val_pkl = dataset_dir / "val_annotation.pkl"
        if dataset_dir.exists() and val_pkl.exists():
            print(f"   ✓ Found EfficientGEBD dataset at: {dataset_dir}")
            print(f"   ✓ Found annotation pkl: {val_pkl.name}")
        else:
            print(f"   ✗ EfficientGEBD dataset missing at {dataset_dir}")
            print(f"     Run: python tools/prepare_efficient_gebd_dataset.py")

    # 2. Data Loading & Training Probe
    probe_dataset_and_training(model_type, config, dry_run_iters, mode)

    # 3. Command Preview
    print("\n [3/4] Command Preview:")
    if model_type in ["ddm_net", "ddm"]:
        ddm_dir = REPO_ROOT / "src" / "step_segment" / "DDM-Net"
        if mode == "train":
            cmd = [
                sys.executable,
                str(ddm_dir / "train_sop_lightning.py"),
                "--output-dir", str(output_dir),
                "--num-gpus", str(num_gpus),
            ]
            if "backbone" in model_cfg:
                cmd.extend(["--backbone", str(model_cfg["backbone"])])
            if "epochs" in train_cfg:
                cmd.extend(["--epochs", str(train_cfg["epochs"])])
            if "learning_rate" in train_cfg:
                cmd.extend(["--learning-rate", str(train_cfg["learning_rate"])])
            if "aux_loss_weight" in train_cfg:
                cmd.extend(["--aux-loss-weight", str(train_cfg["aux_loss_weight"])])
            print(f"   Working Directory: {ddm_dir}")
            print(f"   Execution Command: {' '.join(cmd)}")
        elif mode == "infer":
            ckpt = config.get("checkpoint") or str(output_dir / "best_model.ckpt")
            cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "infer_ddm_net.py"),
                "--checkpoint", str(ckpt),
                "--output-dir", str(output_dir),
            ]
            print(f"   Execution Command: {' '.join(cmd)}")

    elif model_type in ["diff_gebd", "diffgebd"]:
        diff_dir = REPO_ROOT / "src" / "step_segment" / "DiffGEBD"
        config_file = config.get("diffgebd_config_file") or (diff_dir / "config" / "sewing_diffgebd_resnet50_chunked.yaml")
        if mode == "train":
            cmd = [
                "torchrun", f"--nproc_per_node={num_gpus}",
                "train.py",
                "--config-file", str(config_file),
                "OUTPUT_DIR", str(output_dir),
            ]
            print(f"   Working Directory: {diff_dir}")
            print(f"   Execution Command: {' '.join(cmd)}")
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
            print(f"   Execution Command: {' '.join(cmd)}")

    elif model_type in ["efficient_gebd", "efficientgebd"]:
        eff_dir = REPO_ROOT / "src" / "step_segment" / "EfficientGEBD"
        config_file = config.get("efficientgebd_config_file") or (eff_dir / "config-files" / "sewing_resnet50.yaml")
        if mode == "train":
            cmd = [
                "torchrun", f"--nproc_per_node={num_gpus}",
                "train.py",
                "--config-file", str(config_file),
                "OUTPUT_DIR", str(output_dir),
            ]
            print(f"   Working Directory: {eff_dir}")
            print(f"   Execution Command: {' '.join(cmd)}")
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
            print(f"   Execution Command: {' '.join(cmd)}")

    # 4. Expected Artifacts
    print("\n [4/4] Expected Artifacts upon Completion:")
    print(f"   • {output_dir / 'run.log'}")
    print(f"   • {output_dir / 'train.log'}")
    print(f"   • {output_dir / ('best_model.ckpt' if model_type in ['ddm_net', 'ddm'] else 'model_best.pth')}")
    print(f"   • {output_dir / 'predictions.json'}")
    print(f"   • {output_dir / 'metrics.json'}")
    print(f"   • Auto-updates: experiments/step_segment/step_segment_overview.md")

    print("\n" + "=" * 70)
    print(" [DRY-RUN] Pre-flight verification & probe completed.")
    print(" Remove --dry-run flag to start full training run.")
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_id = config.get("experiment_id", "exp_unnamed")
    track = config.get("track", "step_segment")
    iter_id = config.get("iteration_id", "iter_01")
    model_type = config.get("model_type", "efficient_gebd").lower()

    output_dir = args.output_dir or (REPO_ROOT / "outputs" / track / model_type / iter_id / exp_id)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    if args.dry_run:
        dry_run_check(config, output_dir, args.mode, args.num_gpus, env, dry_run_iters=args.dry_run_iters)
        return

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

        # Automatically update central step_segment_overview.md
        update_overview_tracker(config, output_dir, metrics_data)
    else:
        logger.warning(f"Notice: metrics.json was not generated in {output_dir}")


if __name__ == "__main__":
    main()
