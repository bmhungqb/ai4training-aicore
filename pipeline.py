#!/usr/bin/env python3
"""Convenience root wrapper forwarding to src.action_segment.pipeline."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.action_segment.pipeline import main

if __name__ == "__main__":
    main()
