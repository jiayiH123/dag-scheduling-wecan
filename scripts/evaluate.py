#!/usr/bin/env python3
"""Evaluate all Phase-1 baselines on one immutable test dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instance import load_dataset
from environment.config import load_yaml
from evaluation.metrics import evaluate_wecan, standard_baselines, summarize
from models.wecan import WeCAN, WeCANConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--test-data", default="data/generated/test.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/phase1_evaluation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-count", type=int, default=0, help="Optional number of independent sampled WeCAN schedules per instance.")
    args = parser.parse_args()
    configuration = load_yaml(args.config)
    instances = load_dataset(args.test_data)
    random, greedy, ca_heft, standard_heft = standard_baselines(instances, configuration["seed"])
    records = [
        summarize(random, greedy, ca_heft),
        summarize(greedy, greedy, ca_heft),
        summarize(ca_heft, greedy, ca_heft),
        summarize(standard_heft, greedy, ca_heft),
    ]
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model = WeCAN(instances[0].resource_dims, WeCANConfig(**checkpoint["model_config"])).to(args.device)
        model.load_state_dict(checkpoint["model_state"])
        records.append(summarize(evaluate_wecan(instances, model, "greedy", configuration["seed"]), greedy, ca_heft))
        for sample in range(args.sample_count):
            records.append(summarize(evaluate_wecan(instances, model, "sample", configuration["seed"] + sample), greedy, ca_heft))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")
    for record in records:
        print(asdict(record))
    print(f"Saved metrics to {destination}")


if __name__ == "__main__":
    main()
