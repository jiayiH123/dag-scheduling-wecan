#!/usr/bin/env python3
"""Run hand-crafted and random MILP-versus-exhaustive Oracle cross-checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.algorithms import greedy_schedule, heft_schedule, random_schedule
from oracle.exhaustive_oracle import solve_exhaustive_oracle
from oracle.milp_oracle import solve_milp_oracle
from scheduler.validator import validate_schedule
from tests.fixtures import handcrafted_instances, random_tiny_instances

EPS = 1e-6


def check_instance(instance) -> dict:
    milp = solve_milp_oracle(instance)
    exhaustive = solve_exhaustive_oracle(instance)
    heuristic_rows = []
    for name, schedule in (
        ("random", random_schedule(instance, 17)),
        ("greedy", greedy_schedule(instance)),
        ("ca_heft", heft_schedule(instance)),
    ):
        validator = validate_schedule(instance, schedule)
        heuristic_rows.append({
            "algorithm": name,
            "makespan": schedule.makespan,
            "validator": validator.to_dict(),
            "invalidly_better_than_oracle": (
                milp.makespan is not None and schedule.makespan < milp.makespan - EPS
            ),
        })
    passed = (
        milp.status == "optimal"
        and exhaustive.status == "optimal"
        and milp.makespan is not None
        and exhaustive.makespan is not None
        and abs(milp.makespan - exhaustive.makespan) <= EPS
        and milp.validator is not None and milp.validator.feasible
        and exhaustive.validator is not None and exhaustive.validator.feasible
        and all(item["validator"]["feasible"] and not item["invalidly_better_than_oracle"] for item in heuristic_rows)
    )
    return {
        "instance": instance.name,
        "tasks": instance.num_tasks,
        "pools": instance.num_pools,
        "time_scale": milp.time_scale,
        "milp_status": milp.status,
        "milp_makespan": milp.makespan,
        "exhaustive_status": exhaustive.status,
        "exhaustive_makespan": exhaustive.makespan,
        "absolute_difference": None if milp.makespan is None or exhaustive.makespan is None else abs(milp.makespan - exhaustive.makespan),
        "milp_seconds": milp.wall_seconds,
        "exhaustive_seconds": exhaustive.wall_seconds,
        "milp_validator": None if milp.validator is None else milp.validator.to_dict(),
        "exhaustive_validator": None if exhaustive.validator is None else exhaustive.validator.to_dict(),
        "heuristics": heuristic_rows,
        "crosscheck_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="results/oracle/crosscheck.jsonl")
    args = parser.parse_args()
    instances = [*handcrafted_instances().values(), *random_tiny_instances(args.random_count, args.seed)]
    rows = [check_instance(instance) for instance in instances]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    failures = [row for row in rows if not row["crosscheck_passed"]]
    print(f"Cross-checked {len(rows)} instances; failures={len(failures)}; report={destination}")
    if failures:
        print(json.dumps(failures[0], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
