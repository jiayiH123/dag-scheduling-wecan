#!/usr/bin/env python3
"""Audit the saved A1-b candidates without generating or screening new instances."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.a1b_audit import write_a1b_distribution_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/diagnostics/A1-b-screen/candidates.jsonl")
    parser.add_argument("--output-dir", default="results/diagnostics/A1-b-screen")
    args = parser.parse_args()
    result = write_a1b_distribution_audit(args.input, args.output_dir)
    print("A1-b audited candidates=", result["aggregate"]["candidate_count"], sep="")


if __name__ == "__main__":
    main()
