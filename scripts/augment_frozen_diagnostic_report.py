#!/usr/bin/env python3
"""Add deterministic final-trace detail to a completed frozen diagnostic without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.wecan import WeCAN, WeCANConfig
from scripts.run_diagnostics import load_frozen_instance
from scheduler.generator import SkipExtendedGenerator
from training.diagnostics import _trace_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--instance-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    gate_path = Path(args.gate)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    instance = load_frozen_instance(args.instance_json)
    model = WeCAN(instance.resource_dims, WeCANConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rollout_count = int(checkpoint["train_config"]["rollouts_per_instance"])
    final_update = int(checkpoint["update"]) - 1
    generator = SkipExtendedGenerator()
    with torch.no_grad():
        output = model(instance)
        traces = []
        for trajectory in range(rollout_count):
            random_generator = torch.Generator(device="cpu")
            random_generator.manual_seed(args.seed + 1_000_003 * final_update + trajectory)
            traces.append(generator.decode(
                instance, output.task_pool_scores, output.skip_parameters,
                mode="sample", generator=random_generator, track_log_probability=True,
            ))
    replay = _trace_summary(instance, traces)
    expected = gate["final_samples"]["makespans"]
    if replay["makespans"] != expected:
        raise RuntimeError(f"Replay diverged from stored final samples: {replay['makespans']} != {expected}")
    gate["final_samples"] = replay
    gate["final_trace_replay"] = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "final_update": final_update,
        "verified_against_stored_makespans": True,
    }
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("Augmented final trace report:", gate_path)


if __name__ == "__main__":
    main()
