#!/usr/bin/env python3
"""Generate a deterministic CommBudget-WeCAN V1 JSON dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.comm_budget_synthetic import CommBudgetSyntheticConfig, generate_comm_budget_dataset
from data.instance import save_dataset
from environment.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/comm_budget_synthetic_v1.yaml")
    parser.add_argument("--output", default="data/generated/comm_budget_v1.json")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", default="comm-budget-v1")
    args = parser.parse_args()

    configuration = load_yaml(args.config)
    config = CommBudgetSyntheticConfig.from_mapping(configuration["synthetic"])
    count = int(configuration.get("sanity_count", 300)) if args.count is None else args.count
    seed = int(configuration["seed"]) if args.seed is None else args.seed
    instances = generate_comm_budget_dataset(config, count, seed, args.prefix)
    save_dataset(instances, args.output)
    print(f"Wrote {len(instances)} CommBudget V1 instances to {args.output}")


if __name__ == "__main__":
    main()
