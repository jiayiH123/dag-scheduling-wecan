#!/usr/bin/env python3
"""Reanalyse preserved A1 logs as A1-a without retraining or overwriting them."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.diagnostics import reanalyse_legacy_a1a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", default="results/diagnostics/A1")
    parser.add_argument("--output-dir", default="results/diagnostics/A1-a")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = reanalyse_legacy_a1a(args.legacy_dir, args.output_dir, seed=args.seed)
    print("A1-a passed=", result["passed"], "output=", Path(args.output_dir) / "report.json", sep="")


if __name__ == "__main__":
    main()
