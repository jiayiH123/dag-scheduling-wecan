#!/usr/bin/env python3
"""Screen the predeclared active-wait A1-c fixture without policy training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.diagnostic_instances import screen_a1c_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/diagnostics/A1-c-screen")
    args = parser.parse_args()
    result = screen_a1c_candidates(args.output_dir)
    print(
        "A1-c fixtures=",
        result["fixture_count"],
        "selected_fixture=",
        result["selected_fixture"],
        "output=",
        args.output_dir,
        sep="",
    )


if __name__ == "__main__":
    main()
