"""Canonical schedule representation and independent feasibility validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from data.instance import DAGInstance

EPS = 1e-7


@dataclass(frozen=True)
class TaskPlacement:
    task: int
    pool: int
    start: float
    end: float


@dataclass(frozen=True)
class Schedule:
    placements: tuple[TaskPlacement, ...]

    @property
    def makespan(self) -> float:
        return max((placement.end for placement in self.placements), default=0.0)

    def placement_for(self, task: int) -> TaskPlacement:
        for placement in self.placements:
            if placement.task == task:
                return placement
        raise KeyError(f"No placement for task {task}.")

    def ordered(self) -> tuple[TaskPlacement, ...]:
        return tuple(sorted(self.placements, key=lambda item: item.task))


@dataclass(frozen=True)
class ValidationResult:
    feasible: bool
    violations: tuple[str, ...]


def validate_schedule(instance: DAGInstance, schedule: Schedule) -> ValidationResult:
    """Validate all Phase-1 WeCAN constraints independently of any generator."""
    violations: list[str] = []
    placements = {placement.task: placement for placement in schedule.placements}
    if len(placements) != instance.num_tasks or set(placements) != set(range(instance.num_tasks)):
        violations.append("Schedule must contain exactly one placement for every task.")
        return ValidationResult(False, tuple(violations))

    for task, placement in placements.items():
        if not 0 <= placement.pool < instance.num_pools:
            violations.append(f"Task {task} has invalid pool {placement.pool}.")
            continue
        expected_duration = instance.actual_duration(task, placement.pool) if instance.compatibility[task][placement.pool] > 0 else None
        if expected_duration is None:
            violations.append(f"Task {task} is assigned to an incompatible pool.")
        elif abs((placement.end - placement.start) - expected_duration) > EPS:
            violations.append(f"Task {task} duration does not match its pool compatibility.")
        if placement.start < -EPS:
            violations.append(f"Task {task} begins before time zero.")

    for source, destination in instance.edges:
        if source in placements and destination in placements:
            if placements[source].end > placements[destination].start + EPS:
                violations.append(f"Dependency {source}->{destination} is violated.")

    event_times = sorted({
        time
        for placement in schedule.placements
        for time in (placement.start, placement.end)
    })
    for pool in range(instance.num_pools):
        for left, right in zip(event_times, event_times[1:]):
            if right - left <= EPS:
                continue
            midpoint = (left + right) / 2.0
            usage = [0.0] * instance.resource_dims
            for placement in schedule.placements:
                if placement.pool != pool or not (placement.start <= midpoint < placement.end):
                    continue
                for dimension, demand in enumerate(instance.task_demands[placement.task]):
                    usage[dimension] += demand
            for dimension, value in enumerate(usage):
                if value > instance.pool_capacities[pool][dimension] + EPS:
                    violations.append(
                        f"Pool {pool} exceeds capacity in dimension {dimension} during ({left}, {right})."
                    )
    return ValidationResult(not violations, tuple(violations))
