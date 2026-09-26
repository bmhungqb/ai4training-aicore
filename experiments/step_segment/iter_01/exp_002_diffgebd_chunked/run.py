#!/usr/bin/env python3
"""Runner for exp_002_diffgebd_chunked."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "experiments" / "templates" / "step_segment_runner.py"
CONFIG = Path(__file__).parent / "config.yaml"

if __name__ == "__main__":
    cmd = [sys.executable, str(RUNNER), "--config", str(CONFIG)] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
