#!/usr/bin/env python3
"""Create a simple Phase-1 makespan comparison bar chart from evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/phase1_evaluation.json")
    parser.add_argument("--output", default="results/phase1_makespan.png")
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    labels = [record["algorithm"] for record in records]
    means = [record["makespan_mean"] for record in records]
    stds = [record["makespan_std"] for record in records]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, means, yerr=stds, capsize=4)
    axis.set_ylabel("Makespan")
    axis.set_title("Phase-1 evaluation (same test set)")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
