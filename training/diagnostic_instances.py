"""Pre-registered A1-b/A1-c diagnostic instance screening utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json

from baselines.algorithms import ca_heft_schedule, greedy_schedule
from data.instance import DAGInstance, GeneratorConfig, RandomDAGGenerator
from oracle.exhaustive_oracle import solve_exhaustive_with_wait, solve_exhaustive_without_wait
from oracle.fixtures import handcrafted_instances
from oracle.milp_oracle import solve_milp_oracle
from scheduler.types import Schedule
from scheduler.validator import validate_schedule


@dataclass(frozen=True)
class A1BScreenConfig:
    seed_start: int = 3000
    seed_end: int = 3099
    num_tasks_min: int = 8
    num_tasks_max: int = 10
    num_pools: int = 3
    resource_dims: int = 2
    max_width: int = 4
    edge_probability: float = 0.35
    duration_low: int = 1
    duration_high: int = 5
    demand_fraction_low: float = 0.15
    demand_fraction_high: float = 0.5
    capacity_low: int = 6
    capacity_high: int = 12
    incompatibility_probability: float = 0.15
    compatibility_low: int = 1
    compatibility_high: int = 1
    required_relative_greedy_gap: float = 0.05
    required_absolute_greedy_gap_ticks: int = 2
    preferred_relative_gap_upper: float = 0.30

    def __post_init__(self) -> None:
        if self.seed_end < self.seed_start:
            raise ValueError("seed_end must be at least seed_start")
        if self.seed_end - self.seed_start + 1 < 100:
            raise ValueError("A1-b pre-registration requires at least 100 candidates")


def _schedule_dict(schedule: Schedule) -> dict[str, Any]:
    return {
        "makespan": schedule.makespan,
        "placements": [asdict(item) for item in schedule.ordered()],
    }


def a1b_candidate_instance(seed: int, config: A1BScreenConfig) -> DAGInstance:
    """Generate exactly one deterministic, integer-tick candidate for a predeclared seed."""
    task_count = config.num_tasks_min + (seed - config.seed_start) % (
        config.num_tasks_max - config.num_tasks_min + 1
    )
    raw = RandomDAGGenerator(
        GeneratorConfig(
            num_tasks_min=task_count,
            num_tasks_max=task_count,
            num_pools=config.num_pools,
            resource_dims=config.resource_dims,
            max_width=config.max_width,
            edge_probability=config.edge_probability,
            duration_low=config.duration_low,
            duration_high=config.duration_high,
            demand_fraction_low=config.demand_fraction_low,
            demand_fraction_high=config.demand_fraction_high,
            capacity_low=config.capacity_low,
            capacity_high=config.capacity_high,
            incompatibility_probability=config.incompatibility_probability,
            compatibility_low=config.compatibility_low,
            compatibility_high=config.compatibility_high,
        ),
        seed,
    ).generate(f"a1b-seed-{seed}")
    # The random generator produces real-valued demand/capacity values. Their explicit
    # conversion is part of this fixed screen specification, never Oracle rounding.
    instance = DAGInstance(
        name=raw.name,
        task_durations=tuple(float(max(1, round(value))) for value in raw.task_durations),
        task_demands=tuple(
            tuple(float(max(1, round(value))) for value in row) for row in raw.task_demands
        ),
        pool_capacities=tuple(
            tuple(float(max(1, round(value))) for value in row) for row in raw.pool_capacities
        ),
        compatibility=raw.compatibility,
        edges=raw.edges,
    )
    instance.validate()
    return instance


def audit_a1b_candidate(seed: int, config: A1BScreenConfig) -> dict[str, Any]:
    """Evaluate one candidate exclusively with allowed Oracle/heuristic inputs."""
    instance = a1b_candidate_instance(seed, config)
    milp = solve_milp_oracle(instance)
    greedy = greedy_schedule(instance)
    ca_heft = ca_heft_schedule(instance)
    greedy_validation = validate_schedule(instance, greedy).to_dict()
    ca_heft_validation = validate_schedule(instance, ca_heft).to_dict()
    reasons: list[str] = []
    relative_gap = None
    absolute_gap = None
    if milp.status != "optimal" or milp.makespan is None or milp.validator is None or not milp.validator.feasible:
        reasons.append("milp_not_proven_optimal_or_invalid")
    if not greedy_validation["feasible"]:
        reasons.append("external_greedy_invalid")
    if not ca_heft_validation["feasible"]:
        reasons.append("ca_heft_invalid")
    if milp.makespan is not None:
        absolute_gap = greedy.makespan - milp.makespan
        relative_gap = absolute_gap / milp.makespan
        if relative_gap < config.required_relative_greedy_gap:
            reasons.append("external_greedy_relative_gap_below_5_percent")
        if absolute_gap < config.required_absolute_greedy_gap_ticks:
            reasons.append("external_greedy_absolute_gap_below_2_ticks")
    else:
        reasons.append("missing_oracle_makespan")
    mandatory_accepted = not reasons
    preferred_tier = (
        mandatory_accepted
        and relative_gap is not None
        and config.required_relative_greedy_gap <= relative_gap <= config.preferred_relative_gap_upper
    )
    return {
        "seed": seed,
        "instance": instance.to_dict(),
        "milp": milp.to_dict(),
        "external_greedy": {
            "schedule": _schedule_dict(greedy),
            "validator": greedy_validation,
        },
        "ca_heft": {
            "schedule": _schedule_dict(ca_heft),
            "validator": ca_heft_validation,
        },
        "external_greedy_relative_gap": relative_gap,
        "external_greedy_absolute_gap_ticks": absolute_gap,
        "mandatory_accepted": mandatory_accepted,
        "preferred_gap_tier": preferred_tier,
        "rejection_reasons": reasons,
    }


def screen_a1b_candidates(output_directory: str | Path, config: A1BScreenConfig = A1BScreenConfig()) -> dict[str, Any]:
    """Audit every pre-registered seed then select the first allowed candidate.

    The selection uses no training metric. Preferred 5–30% gaps take precedence; when
    that bucket is empty, the first >30% candidate satisfying mandatory conditions is
    selected. The range is never expanded by this function.
    """
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [audit_a1b_candidate(seed, config) for seed in range(config.seed_start, config.seed_end + 1)]
    preferred = next((row for row in rows if row["preferred_gap_tier"]), None)
    fallback = next((row for row in rows if row["mandatory_accepted"] and not row["preferred_gap_tier"]), None)
    selected = preferred or fallback
    selection_reason = None
    if selected is not None:
        selection_reason = (
            "first_seed_in_preferred_5_to_30_percent_gap_tier"
            if selected is preferred
            else "first_seed_above_preferred_gap_tier_after_no_preferred_candidate"
        )
        selected["selection_reason"] = selection_reason
    (destination / "candidates.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    config_dict = asdict(config)
    summary = {
        "stage": "A1-b-screen",
        "screening_only": True,
        "config": config_dict,
        "candidate_count": len(rows),
        "mandatory_accepted_count": sum(row["mandatory_accepted"] for row in rows),
        "preferred_tier_count": sum(row["preferred_gap_tier"] for row in rows),
        "selected_seed": None if selected is None else selected["seed"],
        "selection_reason": selection_reason,
        "stopped_without_selection": selected is None,
        "audit_path": str(destination / "candidates.jsonl"),
    }
    (destination / "screening_config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if selected is not None:
        (destination / "selected_instance.json").write_text(json.dumps(selected["instance"], indent=2), encoding="utf-8")
        (destination / "selected_result.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return {**summary, "selected": selected}


def a1c_fixture_candidates() -> Iterable[tuple[str, DAGInstance, str]]:
    fixture = handcrafted_instances()["a1c_active_wait_six_task"]
    yield (
        "a1c_active_wait_six_task",
        fixture,
        (
            "At t=0 the root consumes y while an independent long x blocker is feasible. "
            "A no-wait policy starts the blocker and delays the critical x child; active waiting "
            "until the root completes allows that child first, then overlaps blocker and the long y descendant. "
            "The x/y dependent tails preserve the strict makespan gap."
        ),
    )


def screen_a1c_candidates(output_directory: str | Path) -> dict[str, Any]:
    """Screen predeclared hand-crafted A1-c fixtures with two independent enumerations."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, instance, rationale in a1c_fixture_candidates():
        milp = solve_milp_oracle(instance)
        with_wait = solve_exhaustive_with_wait(instance)
        without_wait = solve_exhaustive_without_wait(instance)
        greedy = greedy_schedule(instance)
        ca_heft = ca_heft_schedule(instance)
        greedy_validation = validate_schedule(instance, greedy).to_dict()
        ca_heft_validation = validate_schedule(instance, ca_heft).to_dict()
        reasons: list[str] = []
        if milp.status != "optimal" or milp.makespan is None or milp.validator is None or not milp.validator.feasible:
            reasons.append("milp_not_proven_optimal_or_invalid")
        for label, result in (("with_wait", with_wait), ("without_wait", without_wait)):
            if result.status != "optimal" or result.makespan is None or result.validator is None or not result.validator.feasible:
                reasons.append(f"{label}_search_not_complete_or_invalid")
        if with_wait.makespan is None or without_wait.makespan is None or not with_wait.makespan < without_wait.makespan:
            reasons.append("active_wait_not_strictly_better")
        if not with_wait.detail.get("contains_active_wait", False):
            reasons.append("optimal_with_wait_history_lacks_active_wait")
        if milp.makespan is None or not greedy.makespan > milp.makespan:
            reasons.append("external_greedy_is_not_strictly_suboptimal")
        rows.append({
            "fixture": name,
            "instance": instance.to_dict(),
            "construction_rationale": rationale,
            "milp": milp.to_dict(),
            "exhaustive_with_wait": with_wait.to_dict(),
            "exhaustive_without_wait": without_wait.to_dict(),
            "external_greedy": {"schedule": _schedule_dict(greedy), "validator": greedy_validation},
            "ca_heft": {"schedule": _schedule_dict(ca_heft), "validator": ca_heft_validation},
            "accepted": not reasons,
            "rejection_reasons": reasons,
        })
    selected = next((row for row in rows if row["accepted"]), None)
    (destination / "candidates.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    summary = {
        "stage": "A1-c-screen",
        "screening_only": True,
        "fixture_count": len(rows),
        "selected_fixture": None if selected is None else selected["fixture"],
        "stopped_without_selection": selected is None,
        "audit_path": str(destination / "candidates.jsonl"),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if selected is not None:
        (destination / "selected_instance.json").write_text(json.dumps(selected["instance"], indent=2), encoding="utf-8")
        (destination / "selected_result.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return {**summary, "selected": selected}
