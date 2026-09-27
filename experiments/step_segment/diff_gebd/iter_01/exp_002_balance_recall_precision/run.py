#!/usr/bin/env python3
"""Runner for exp_002_balance_recall_precision."""
import subprocess
import sys
from pathlib import Path

def find_repo_root(path: Path) -> Path:
    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    return path.resolve().parents[4]


REPO_ROOT = find_repo_root(Path(__file__).resolve())
RUNNER = REPO_ROOT / "experiments" / "templates" / "step_segment_runner.py"
CONFIG = Path(__file__).parent / "config.yaml"

if __name__ == "__main__":
    cmd = [sys.executable, str(RUNNER), "--config", str(CONFIG)] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
