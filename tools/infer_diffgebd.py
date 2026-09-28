#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import yaml
import random
from pathlib import Path


def run_single_inference(
    split: str,
    config_path: Path,
    weights_path: Path,
    out_dir: Path,
    num_gpus: int,
    viz: bool,
    repo_root: Path,
    diffgebd_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    is_chunked = False
    test_ds = cfg_dict.get("DATASETS", {}).get("TEST", ["SEWING_CHUNKED_val"])
    if isinstance(test_ds, (list, tuple)) and len(test_ds) > 0:
        if "CHUNKED" in str(test_ds[0]):
            is_chunked = True

    if split == "train":
        target_dataset = "SEWING_CHUNKED_train" if is_chunked else "SEWING_train"
    else:
        target_dataset = "SEWING_CHUNKED_val" if is_chunked else "SEWING_val"

    print("\n" + "=" * 65)
    print(f"1/2: Running DiffGEBD native inference on split '{split}' ({target_dataset})...")
    print("=" * 65)

    port = str(random.randint(10220, 10290))
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        "--master_port",
        port,
        "train.py",
        "--config-file",
        str(config_path),
        "--resume",
        str(weights_path),
        "--test-only",
        "DATASETS.TEST",
        f"('{target_dataset}',)",
    ]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        subprocess.run(cmd, cwd=str(diffgebd_dir), env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during DiffGEBD inference on split {split}. Exit code: {e.returncode}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(f"2/2: Exporting predictions for split '{split}' to JSON format...")
    print("=" * 65)

    output_parent = diffgebd_dir / "output"
    pkl_files = list(output_parent.rglob("model_pred_dict_ep-1.pkl"))
    if not pkl_files:
        print(f"Could not find model_pred_dict_ep-1.pkl inside {output_parent}")
        sys.exit(1)

    pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_pkl = pkl_files[0]
    print(f"Found native prediction file: {latest_pkl}")

    export_cmd = [
        sys.executable,
        str(repo_root / "tools" / "export_diffgebd_predictions.py"),
        "--pred-pkl",
        str(latest_pkl),
        "--split",
        split,
        "--out-dir",
        str(out_dir),
    ]
    if viz:
        export_cmd.append("--viz")

    subprocess.run(export_cmd, cwd=str(repo_root), check=True)
    print(f"✓ Split '{split}' complete! Predictions saved to -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DiffGEBD inference and export predictions for train, val, or entire dataset."
    )
    parser.add_argument("--config-file", type=str, required=True, help="Path to the .yaml config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model_best.pth weights")
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "train", "test", "all"],
        default="all",
        help="Dataset split to evaluate ('val', 'train', or 'all' for both train and val)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/diffgebd_preds",
        help="Output directory (creates <out-dir>/train and <out-dir>/val when --split all)",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--viz", action="store_true", help="Render annotated.mp4 and score_curve.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    diffgebd_dir = repo_root / "src" / "step_segment" / "DiffGEBD"

    config_path = Path(args.config_file).resolve()
    weights_path = Path(args.weights).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class TeeLogger:
        def __init__(self, filepath, stream):
            self.file = open(filepath, "a", encoding="utf-8", buffering=1)
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            self.stream.flush()
            self.file.write(data)
            self.file.flush()

        def flush(self):
            self.stream.flush()
            self.file.flush()

    sys.stdout = TeeLogger(out_dir / "infer.log", sys.stdout)
    sys.stderr = TeeLogger(out_dir / "infer.log", sys.stderr)

    if args.split == "all":
        print(f"Running inference on FULL DATASET (both val and train)...")
        run_single_inference(
            "val",
            config_path,
            weights_path,
            out_dir / "val",
            args.num_gpus,
            args.viz,
            repo_root,
            diffgebd_dir,
        )
        run_single_inference(
            "train",
            config_path,
            weights_path,
            out_dir / "train",
            args.num_gpus,
            args.viz,
            repo_root,
            diffgebd_dir,
        )
    else:
        dest_dir = out_dir / args.split if out_dir.name != args.split else out_dir
        run_single_inference(
            args.split,
            config_path,
            weights_path,
            dest_dir,
            args.num_gpus,
            args.viz,
            repo_root,
            diffgebd_dir,
        )

    print("\n" + "=" * 65)
    print("SUCCESS! All requested inferences completed.")
    print(f"Predictions and metrics saved to: {out_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
