#!/usr/bin/env python3
"""exp_003_session_confound_profile (iter_02, Revision 2).

Descriptive, non-inferential per-video motion-energy / brightness / contrast
profiling for Hypothesis 3 (Session-Level Heterogeneity Dominates). Invokes
`tools/profile_diffgebd_video_stats.py` against each of the 9 val videos'
UNTOUCHED raw source frames at `data/diff_gebd_dataset/images/val/<video_id>/frame*.jpg`
(corrected path per Debate verdict FAIL -- NOT the non-existent
`outputs/.../<video_id>/` path, and NOT the stateful `__chunk<i>/` symlink
directories). No re-inference, no GPU. Processes ALL frames per video (no
subsampling, per Debate verdict WARN correction).

Revision 2 statistical framing: does NOT compute a Pearson correlation
coefficient (n=9 is not statistically meaningful). Presents a plain
descriptive table plus an explicit cd12_chuyen1 vs cd12_chuyen2 vs
rest-of-dataset-range comparison, labeled hypothesis-generating / non-inferential.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

def find_repo_root(p: Path) -> Path:
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return p.resolve().parents[4]

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(EXP_DIR)


def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    base_out = REPO_ROOT / "outputs" / "step_segment" / "diff_gebd" / "iter_02" / config["experiment_id"]
    base_out.mkdir(parents=True, exist_ok=True)

    dataset_dir = REPO_ROOT / config["dataset_dir"]
    metrics_json = REPO_ROOT / config["metrics_json"]

    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "profile_diffgebd_video_stats.py"),
        "--dataset-dir", str(dataset_dir),
        "--split", config["split"],
        "--metrics-json", str(metrics_json),
        "--named-pair", *config["named_pair"],
        "--out-dir", str(base_out),
    ]
    run(cmd)

    print(f"\nDone. Reports written under {base_out}/")
    print("  - video_stats_profile.json / .csv: full 9-video descriptive table.")
    print("  - named_pair_comparison.json: cd12_chuyen1 vs cd12_chuyen2 vs rest-of-dataset-range "
          "(hypothesis-generating, non-inferential -- NOT a Pearson correlation).")
    print("Next: check whether cd12_chuyen1 sits outside the range spanned by the other 8 videos "
          "on any of the 3 stats (motion_energy, brightness, contrast). If so, that stat becomes "
          "a concrete, named candidate confound for future, separately-designed, adequately-powered "
          "investigation -- explicitly not proof of causation.")


if __name__ == "__main__":
    main()
