"""Pre-registered exact A1-b structured candidate generator and Oracle screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from baselines.algorithms import ca_heft_schedule, greedy_schedule
from data.instance import DAGInstance
from oracle.exhaustive_oracle import solve_exhaustive_with_wait, solve_exhaustive_without_wait
from oracle.milp_oracle import solve_milp_oracle
from scheduler.types import Schedule
from scheduler.validator import validate_schedule


@dataclass(frozen=True)
class A1BExactConfig:
    name: str = "a1b_exact_structured_v1"
    seed_start: int = 4000
    seed_end: int = 4099
    fast_tight_capacity: tuple[int, int] = (7, 7)
    slow_loose_capacity: tuple[int, int] = (14, 14)
    required_relative_greedy_gap_min: float = 0.05
    required_relative_greedy_gap_max: float = 0.30
    required_absolute_greedy_gap_ticks: int = 2

    def __post_init__(self) -> None:
        if self.seed_end - self.seed_start + 1 != 100:
            raise ValueError("A1-b-exact requires exactly 100 pre-registered candidates.")


def _schedule_dict(schedule: Schedule) -> dict[str, Any]:
    return {
        "makespan": schedule.makespan,
        "placements": [asdict(placement) for placement in schedule.ordered()],
    }


def exact_config_payload(config: A1BExactConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "task_range": [6, 7],
        "pools": 2,
        "resource_dims": 2,
        "exact_integer_ticks": True,
        "selection": "first accepted candidate in ascending seed order",
        "active_wait_rule": "MILP == exhaustive_with_wait == exhaustive_without_wait; without-wait optimum is the no-active-wait witness",
    }


def exact_config_hash(config: A1BExactConfig) -> str:
    encoded = json.dumps(exact_config_payload(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def generate_a1b_exact_instance(seed: int, config: A1BExactConfig = A1BExactConfig()) -> DAGInstance:
    """Generate a deterministic 6/7 task resource-allocation conflict without active-wait need.

    Task 0 unlocks fast-only critical task 2, then the long fast-only y task 3.
    Independent task 1 is attracted to the fast x capacity but can instead use the slow
    loose pool. Greedy dispatches it fast at t=0 and delays the spine; the optimum starts
    it slow at t=0 while retaining fast x for task 2. Task 4 is a slow-only branch tail,
    task 5 closes the critical spine, and the optional task 6 is another slow-only branch
    tail. Both exact enumeration spaces retain the same non-delay optimum.
    """
    if not config.seed_start <= seed <= config.seed_end:
        raise ValueError("seed is outside the pre-registered exact A1-b range.")
    offset = seed - config.seed_start
    task_count = 6 + offset % 2
    branch_duration = 2 + (offset // 2) % 2  # Greedy gap: 1/8 or 2/8, always within 5–30%.
    critical_y_duration = 5 + (offset // 4) % 2
    durations = [1.0, float(branch_duration), 1.0, float(critical_y_duration), 1.0, 1.0]
    demands: list[tuple[float, float]] = [
        (0.0, 6.0),  # root: fast-only y
        (6.0, 0.0),  # branch: fast preferred, slow fallback
        (6.0, 0.0),  # critical x: fast-only
        (0.0, 6.0),  # critical y: fast-only
        (6.0, 0.0),  # branch tail: slow-only
        (0.0, 6.0),  # critical tail: fast-only
    ]
    compatibility: list[tuple[float, float]] = [
        (1.0, 0.0),
        (1.0, 0.5),  # pool 1 is exactly 2x slower for task 1.
        (1.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ]
    edges: list[tuple[int, int]] = [(0, 2), (2, 3), (3, 5), (1, 4)]
    if task_count == 7:
        durations.append(1.0)
        demands.append((0.0, 6.0))
        compatibility.append((0.0, 1.0))
        edges.append((4, 6))
    instance = DAGInstance(
        name=f"a1b-exact-seed-{seed}",
        task_durations=tuple(durations),
        task_demands=tuple(demands),
        pool_capacities=(
            tuple(float(value) for value in config.fast_tight_capacity),
            tuple(float(value) for value in config.slow_loose_capacity),
        ),
        compatibility=tuple(compatibility),
        edges=tuple(edges),
    )
    instance.validate()
    return instance


def audit_a1b_exact_candidate(seed: int, config: A1BExactConfig = A1BExactConfig()) -> dict[str, Any]:
    """Audit one candidate strictly through exact solvers, heuristics, and Validator."""
    instance = generate_a1b_exact_instance(seed, config)
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
    for label, result in (("exhaustive_with_wait", with_wait), ("exhaustive_without_wait", without_wait)):
        if (
            result.status != "optimal"
            or result.makespan is None
            or result.validator is None
            or not result.validator.feasible
            or not result.detail.get("complete_search", False)
        ):
            reasons.append(f"{label}_not_complete_or_invalid")
    exact_makespans = [result.makespan for result in (milp, with_wait, without_wait)]
    if None in exact_makespans or max(exact_makespans) - min(exact_makespans) > 1e-6:
        reasons.append("with_wait_and_without_wait_optima_differ_from_exact_optimum")
    if without_wait.detail.get("contains_active_wait", True):
        reasons.append("without_wait_optimal_witness_contains_active_wait")
    if not greedy_validation["feasible"]:
        reasons.append("external_greedy_invalid")
    if not ca_heft_validation["feasible"]:
        reasons.append("ca_heft_invalid")
    oracle_makespan = milp.makespan
    relative_gap = None
    absolute_gap = None
    if oracle_makespan is None:
        reasons.append("missing_oracle_makespan")
    else:
        absolute_gap = greedy.makespan - oracle_makespan
        relative_gap = absolute_gap / oracle_makespan
        if greedy.makespan < oracle_makespan - 1e-6:
            reasons.append("external_greedy_invalidly_better_than_oracle")
        if not config.required_relative_greedy_gap_min <= relative_gap <= config.required_relative_greedy_gap_max:
            reasons.append("external_greedy_relative_gap_outside_5_to_30_percent")
        if absolute_gap < config.required_absolute_greedy_gap_ticks:
            reasons.append("external_greedy_absolute_gap_below_2_ticks")
    return {
        "seed": seed,
        "config_hash": exact_config_hash(config),
        "instance": instance.to_dict(),
        "milp": milp.to_dict(),
        "exhaustive_with_wait": with_wait.to_dict(),
        "exhaustive_without_wait": without_wait.to_dict(),
        "external_greedy": {"schedule": _schedule_dict(greedy), "validator": greedy_validation},
        "ca_heft": {"schedule": _schedule_dict(ca_heft), "validator": ca_heft_validation},
        "external_greedy_relative_gap": relative_gap,
        "external_greedy_absolute_gap_ticks": absolute_gap,
        "accepted": not reasons,
        "rejection_reasons": reasons,
    }


def screen_a1b_exact_candidates(output_directory: str | Path, config: A1BExactConfig = A1BExactConfig()) -> dict[str, Any]:
    """Run all 100 pre-registered exact audits, then select the first accepted seed."""
    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty exact-screen directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    rows = [audit_a1b_exact_candidate(seed, config) for seed in range(config.seed_start, config.seed_end + 1)]
    selected = next((row for row in rows if row["accepted"]), None)
    payload = exact_config_payload(config)
    (destination / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (destination / "screening_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (destination / "screening_config.sha256").write_text(exact_config_hash(config) + "\n", encoding="utf-8")
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in row["rejection_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    exact_equal_count = sum(
        row["milp"]["makespan"] == row["exhaustive_with_wait"]["makespan"] == row["exhaustive_without_wait"]["makespan"]
        and row["exhaustive_with_wait"]["detail"].get("complete_search")
        and row["exhaustive_without_wait"]["detail"].get("complete_search")
        for row in rows
    )
    greedy_gap_count = sum(
        row["external_greedy_relative_gap"] is not None
        and config.required_relative_greedy_gap_min <= row["external_greedy_relative_gap"] <= config.required_relative_greedy_gap_max
        and row["external_greedy_absolute_gap_ticks"] >= config.required_absolute_greedy_gap_ticks
        for row in rows
    )
    summary = {
        "stage": "A1-b-exact-screen",
        "screening_only": True,
        "config_hash": exact_config_hash(config),
        "candidate_count": len(rows),
        "greedy_gap_condition_count": greedy_gap_count,
        "with_wait_equals_without_wait_count": exact_equal_count,
        "accepted_count": sum(row["accepted"] for row in rows),
        "rejection_reason_counts": reason_counts,
        "selected_seed": None if selected is None else selected["seed"],
        "selection_reason": None if selected is None else "first_accepted_seed_in_4000_to_4099",
        "stopped_without_selection": selected is None,
        "audit_path": str(destination / "candidates.jsonl"),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if selected is not None:
        (destination / "selected_instance.json").write_text(json.dumps(selected["instance"], indent=2), encoding="utf-8")
        (destination / "selected_result.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return {**summary, "selected": selected}
