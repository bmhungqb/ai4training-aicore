"""Experiment Runner Template.

This script executes a single reproducible experiment in the AI Research Loop.
Outputs (metrics, logs, predictions) are written to outputs/<track>/<iter>/<exp_id>/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
import yaml

# Path anchors
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiment")
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
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_id = config.get("experiment_id", "exp_unnamed")
    track = config.get("track", "action_segment")
    iter_id = config.get("iteration_id", "iter_01")

    # Resolve output directory
    output_dir = args.output_dir or (REPO_ROOT / "outputs" / track / iter_id / exp_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Experiment: {exp_id} ({track} / {iter_id}) ===")
    logger.info(f"Artifacts will be stored at: {output_dir}")

    start_time = time.time()

    # --- EXPERIMENT IMPLEMENTATION HOOK ---
    # TODO: Implement the specific training/eval invocation here
    # e.g., calling pipeline.py or a model training function with parameters from config
    # -------------------------------------

    elapsed = time.time() - start_time
    logger.info(f"=== Completed in {elapsed:.2f}s ===")

    # Write execution metadata
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": exp_id,
                "track": track,
                "iteration_id": iter_id,
                "elapsed_seconds": elapsed,
                "config": config,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
