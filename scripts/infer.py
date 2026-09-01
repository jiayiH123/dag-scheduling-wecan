#!/usr/bin/env python3
"""Run one model forward pass and construct a schedule for one JSON dataset item."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys
from time import perf_counter

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instance import load_dataset
from models.wecan import WeCAN, WeCANConfig
from scheduler.generator import SkipExtendedGenerator
from scheduler.types import validate_schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/generated/test.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="results/single_instance.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    instance = load_dataset(args.data)[args.index]
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = WeCAN(instance.resource_dims, WeCANConfig(**checkpoint["model_config"])).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.reset_forward_counter()
    with torch.no_grad():
        forward_start = perf_counter()
        output = model(instance)
        forward_seconds = perf_counter() - forward_start
        generation_start = perf_counter()
        trace = SkipExtendedGenerator().decode(instance, output.task_pool_scores, output.skip_parameters, mode="greedy")
        generation_seconds = perf_counter() - generation_start
    validation = validate_schedule(instance, trace.schedule)
    payload = {
        "instance": instance.name,
        "makespan": trace.schedule.makespan,
        "forward_calls": model.forward_calls,
        "forward_seconds": forward_seconds,
        "generation_seconds": generation_seconds,
        "feasible": validation.feasible,
        "violations": list(validation.violations),
        "placements": [asdict(placement) for placement in trace.schedule.ordered()],
        "decisions": list(trace.decisions),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if model.forward_calls != 1:
        raise RuntimeError("Single-instance inference called the model more than once.")
    if not validation.feasible:
        raise RuntimeError("Generated schedule violates Phase-1 constraints.")


if __name__ == "__main__":
    main()
