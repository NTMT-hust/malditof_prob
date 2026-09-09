#!/usr/bin/env python3
"""
Convenience entrypoint for running the pipeline from the project root.
Forwards execution to experiments/run_pipeline.py.
"""
import sys
import subprocess
from pathlib import Path

EXPERIMENTS_RUNNER = Path(__file__).resolve().parent / "experiments" / "run_pipeline.py"

if __name__ == "__main__":
    cmd = [sys.executable, str(EXPERIMENTS_RUNNER)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
