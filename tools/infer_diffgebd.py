#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run DiffGEBD inference and export predictions in one step.")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the .yaml config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model_best.pth weights")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--out-dir", type=str, default="experiments/diffgebd_preds", help="Output directory for JSON predictions")
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
    
    print("=" * 60)
    print("1/2: Running DiffGEBD native inference (--test-only)...")
    print("=" * 60)
    
    # We use torchrun to spawn the validation run
    cmd = [
        "torchrun", "--nproc_per_node", str(args.num_gpus), "--master_port", "10212",
        "train.py",
        "--config-file", str(config_path),
        "--resume", str(weights_path),
        "--test-only"
    ]
    
    # Run the native test inside DiffGEBD directory
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        subprocess.run(cmd, cwd=str(diffgebd_dir), env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during DiffGEBD inference. Exit code: {e.returncode}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("2/2: Exporting predictions to JSON format...")
    print("=" * 60)
    
    # train.py --test-only outputs model_pred_dict_ep-1.pkl into the output_dir defined in the config.
    # To find it, we can parse the config or just assume the user passes the standard structure.
    # Actually, we can just find the most recently created model_pred_dict_ep-1.pkl in the output dir!
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    output_dir_base = cfg.get("OUTPUT_DIR", "")
    if not output_dir_base:
        print("Could not parse OUTPUT_DIR from config.")
        sys.exit(1)
        
    # The actual output dir has exp_name appended as a string suffix (not a subdirectory).
    # e.g. output/sewing_diffgebd_resnet50_ann1_dim512_len150
    # Let's just search inside diffgebd_dir / "output" recursively.
    output_parent = diffgebd_dir / "output"
    pkl_files = list(output_parent.rglob("model_pred_dict_ep-1.pkl"))
    if not pkl_files:
        print(f"Could not find model_pred_dict_ep-1.pkl inside {output_parent}")
        sys.exit(1)
        
    # Sort by modification time to get the one we JUST generated
    pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_pkl = pkl_files[0]
    
    print(f"Found native prediction file: {latest_pkl}")
    
    export_cmd = [
        sys.executable, str(repo_root / "tools" / "export_diffgebd_predictions.py"),
        "--pred-pkl", str(latest_pkl),
        "--split", args.split,
        "--out-dir", str(Path(args.out_dir).resolve())
    ]
    if args.viz:
        export_cmd.append("--viz")
    
    subprocess.run(export_cmd, cwd=str(repo_root), check=True)
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! Inference complete.")
    print(f"Predictions saved to: {Path(args.out_dir).resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
