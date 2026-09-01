#!/usr/bin/env python3
"""Screen the pre-registered A1-b seeds without invoking policy training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.diagnostic_instances import A1BScreenConfig, screen_a1b_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/diagnostics/A1-b-screen")
    args = parser.parse_args()
    result = screen_a1b_candidates(args.output_dir, A1BScreenConfig())
    print(
        "A1-b candidates=",
        result["candidate_count"],
        "selected_seed=",
        result["selected_seed"],
        "output=",
        args.output_dir,
        sep="",
    )


if __name__ == "__main__":
    main()
