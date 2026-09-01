#!/usr/bin/env python3
"""Generate deterministic train/validation/test datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instance import GeneratorConfig, generate_dataset, save_dataset
from environment.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    configuration = load_yaml(args.config)
    seed = configuration["seed"] if args.seed is None else args.seed
    generator_config = GeneratorConfig(**configuration["generator"])
    output_dir = Path(args.output_dir)
    for name, count, split_seed in (
        ("train", args.train_count, seed),
        ("validation", args.validation_count, seed + 1),
        ("test", args.test_count, seed + 2),
    ):
        instances = generate_dataset(generator_config, count, split_seed, name)
        destination = output_dir / f"{name}.json"
        save_dataset(instances, destination)
        print(f"Wrote {len(instances)} {name} instances to {destination}")


if __name__ == "__main__":
    main()
