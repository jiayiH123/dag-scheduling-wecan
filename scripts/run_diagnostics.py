#!/usr/bin/env python3
"""Run one explicit Phase 1.5 diagnostic; frozen A1-c never regenerates its instance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instance import DAGInstance
from environment.config import load_yaml
from models.wecan import WeCANConfig
from training.diagnostics import (
    build_integer_diagnostic_instance,
    run_frozen_instance_gate,
    run_single_instance_gate,
)
from training.reinforce import TrainConfig


PROTECTED_HISTORICAL_OUTPUTS = {
    (PROJECT_ROOT / "results/diagnostics/A1").resolve(),
    (PROJECT_ROOT / "results/diagnostics/A1-a").resolve(),
    (PROJECT_ROOT / "results/diagnostics/A1-b-screen").resolve(),
    (PROJECT_ROOT / "results/diagnostics/A1-b-exact-screen").resolve(),
    (PROJECT_ROOT / "results/diagnostics/A1-c-screen").resolve(),
    (PROJECT_ROOT / "results/diagnostics_gap/A1").resolve(),
}


def load_frozen_instance(path: str | Path) -> DAGInstance:
    """Load exactly the serialized instance; no seed-based generation occurs here."""
    import json

    source = Path(path)
    return DAGInstance.from_dict(json.loads(source.read_text(encoding="utf-8")))


def _assert_safe_output(directory: Path) -> None:
    if directory.resolve() in PROTECTED_HISTORICAL_OUTPUTS:
        raise ValueError(f"Refusing to write into a protected historical result directory: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("A1", "A1-b-exact", "A1-c", "A2"), required=True)
    parser.add_argument("--config", default="configs/phase15_diagnostics.yaml")
    parser.add_argument("--output-dir", default="results/diagnostics")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--instance-json")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    if args.stage in {"A1-c", "A1-b-exact"} and not args.instance_json:
        parser.error(f"{args.stage} requires --instance-json; it must never generate an instance from seed.")
    if args.stage not in {"A1-c", "A1-b-exact"} and args.instance_json:
        parser.error("--instance-json is supported only for frozen A1-c and A1-b-exact diagnostics.")
    config = load_yaml(args.config)
    seed = config["seed"] if args.seed is None else args.seed
    train_values = dict(config["training"])
    if args.max_updates is not None:
        if args.stage not in {"A1-c", "A1-b-exact"} or args.max_updates > 300:
            parser.error("Only frozen A1-c/A1-b-exact diagnostics accept --max-updates, capped at 300.")
        train_values["max_updates"] = args.max_updates
    allowed = {
        "learning_rate", "batch_size", "epochs", "max_updates", "rollouts_per_instance",
        "baseline_mode", "snapshot_sync_interval_updates", "seed", "checkpoint_dir",
    }

    output_root = Path(args.output_dir)
    if args.stage in {"A1-c", "A1-b-exact"}:
        _assert_safe_output(output_root)
        instance = load_frozen_instance(args.instance_json)
        train_values.update(
            seed=seed,
            checkpoint_dir=str(output_root / "checkpoint"),
            epochs=300,
        )
        result = run_frozen_instance_gate(
            instance,
            WeCANConfig(**config["model"]),
            TrainConfig(**{key: value for key, value in train_values.items() if key in allowed}),
            output_root,
            stage=args.stage,
        )
        print(result["stage"], "passed=" + str(result["passed"]), "output=" + str(output_root / "gate.json"))
        return

    stage_directory = output_root / args.stage
    _assert_safe_output(stage_directory)
    instance = build_integer_diagnostic_instance(args.stage, seed)
    train_values.update(seed=seed, checkpoint_dir=str(stage_directory / "checkpoint"), epochs=300)
    result = run_single_instance_gate(
        instance,
        WeCANConfig(**config["model"]),
        TrainConfig(**{key: value for key, value in train_values.items() if key in allowed}),
        stage_directory,
        stage=args.stage,
    )
    print(result["stage"], "passed=" + str(result["passed"]), "output=" + str(stage_directory / "gate.json"))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
