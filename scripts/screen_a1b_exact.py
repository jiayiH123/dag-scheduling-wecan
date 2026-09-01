#!/usr/bin/env python3
"""Run only the approved A1-b-exact 4000–4099 Oracle/heuristic screen."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.a1b_exact import A1BExactConfig, screen_a1b_exact_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/diagnostics/A1-b-exact-screen")
    args = parser.parse_args()
    result = screen_a1b_exact_candidates(args.output_dir, A1BExactConfig())
    print(
        "A1-b-exact candidates=", result["candidate_count"],
        " selected_seed=", result["selected_seed"],
        " output=", args.output_dir,
        sep="",
    )


if __name__ == "__main__":
    main()
