"""Independent event-timeline schedule validator for Phase 1.5 and Phase 2.1a.

This module intentionally does not call the scheduling generator, heuristic logic, or
Oracle solvers. It validates only the concrete schedule (and optionally an action trace).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

from data.instance import DAGInstance
from scheduler.action_bounds import max_decode_actions
from scheduler.types import EPS, Schedule, TaskPlacement


@dataclass(frozen=True)
class IntervalUsage:
    pool: int
    start: float
    end: float
    usage: tuple[float, ...]
    capacity: tuple[float, ...]


@dataclass(frozen=True)
class ScheduleValidationResult:
    feasible: bool
    violations: tuple[str, ...]
    interval_usages: tuple[IntervalUsage, ...]
    trace_checked: bool
    communication_checked: bool = False
    cross_pool_edge_count: int = 0
    total_communication_delay: int = 0
    passive_communication_advance_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "violations": list(self.violations),
            "interval_usages": [asdict(usage) for usage in self.interval_usages],
            "trace_checked": self.trace_checked,
            "communication_checked": self.communication_checked,
            "cross_pool_edge_count": self.cross_pool_edge_count,
            "total_communication_delay": self.total_communication_delay,
            "passive_communication_advance_count": self.passive_communication_advance_count,
        }


def validate_schedule(
    instance: DAGInstance,
    schedule: Schedule,
    *,
    trace: Sequence[str] | None = None,
    eps: float = EPS,
) -> ScheduleValidationResult:
    """Independently validate all scheduling constraints and optional trace events."""
    violations: list[str] = []
    placements_by_task: dict[int, TaskPlacement] = {}
    duplicate_tasks: set[int] = set()
    for placement in schedule.placements:
        if placement.task in placements_by_task:
            duplicate_tasks.add(placement.task)
        else:
            placements_by_task[placement.task] = placement
    if duplicate_tasks:
        violations.append(f"Duplicate task placements: {sorted(duplicate_tasks)}.")
    expected_tasks = set(range(instance.num_tasks))
    actual_tasks = set(placements_by_task)
    missing = sorted(expected_tasks - actual_tasks)
    unexpected = sorted(actual_tasks - expected_tasks)
    if missing:
        violations.append(f"Missing task placements: {missing}.")
    if unexpected:
        violations.append(f"Out-of-range task placements: {unexpected}.")

    for task in sorted(expected_tasks & actual_tasks):
        placement = placements_by_task[task]
        if not 0 <= placement.pool < instance.num_pools:
            violations.append(f"Task {task} has invalid pool {placement.pool}.")
            continue
        if not all(math.isfinite(value) for value in (placement.start, placement.end)):
            violations.append(f"Task {task} has a non-finite start or end time.")
            continue
        if placement.start < -eps:
            violations.append(f"Task {task} starts before time zero.")
        if placement.end < placement.start - eps:
            violations.append(f"Task {task} completes before it starts.")
        compatibility = instance.compatibility[task][placement.pool]
        if compatibility <= 0:
            violations.append(f"Task {task} is assigned to incompatible pool {placement.pool}.")
            continue
        expected_duration = instance.actual_duration(task, placement.pool)
        actual_duration = placement.end - placement.start
        if abs(actual_duration - expected_duration) > eps:
            violations.append(
                f"Task {task} has duration {actual_duration}, expected {expected_duration} on pool {placement.pool}."
            )

    cross_pool_edge_count, total_communication_delay = _validate_precedence_and_communication(
        instance, placements_by_task, violations, eps,
    )
    interval_usages = _validate_capacity(instance, tuple(placements_by_task.values()), violations, eps)
    expected_makespan = max((placement.end for placement in placements_by_task.values()), default=0.0)
    if abs(schedule.makespan - expected_makespan) > eps:
        violations.append(f"Schedule makespan {schedule.makespan} differs from maximum completion {expected_makespan}.")

    trace_checked = trace is not None
    passive_communication_advances = 0
    if trace is not None:
        passive_communication_advances = _validate_trace(instance, schedule, trace, violations, eps)
    return ScheduleValidationResult(
        not violations,
        tuple(violations),
        tuple(interval_usages),
        trace_checked,
        communication_checked=instance.communication_enabled,
        cross_pool_edge_count=cross_pool_edge_count,
        total_communication_delay=total_communication_delay,
        passive_communication_advance_count=passive_communication_advances,
    )


def _validate_precedence_and_communication(
    instance: DAGInstance,
    placements_by_task: dict[int, TaskPlacement],
    violations: list[str],
    eps: float,
) -> tuple[int, int]:
    cross_pool_edge_count = 0
    total_communication_delay = 0
    for parent, child in instance.edges:
        parent_placement = placements_by_task.get(parent)
        child_placement = placements_by_task.get(child)
        if parent_placement is None or child_placement is None:
            continue
        delay = instance.communication_delay_ticks(parent, child, parent_placement.pool, child_placement.pool)
        if parent_placement.pool != child_placement.pool:
            cross_pool_edge_count += 1
            total_communication_delay += delay
        release = parent_placement.end + delay
        if release > child_placement.start + eps:
            violations.append(
                f"Dependency {parent}->{child} is violated: child starts at {child_placement.start}, "
                f"but communication release is {release}."
            )
    return cross_pool_edge_count, total_communication_delay


def _validate_capacity(
    instance: DAGInstance,
    placements: tuple[TaskPlacement, ...],
    violations: list[str],
    eps: float,
) -> list[IntervalUsage]:
    event_times = sorted({time for placement in placements for time in (placement.start, placement.end) if math.isfinite(time)})
    usages: list[IntervalUsage] = []
    for pool in range(instance.num_pools):
        for left, right in zip(event_times, event_times[1:]):
            if right <= left + eps:
                continue
            midpoint = (left + right) / 2.0
            usage = [0.0] * instance.resource_dims
            for placement in placements:
                if placement.pool != pool or not (placement.start <= midpoint < placement.end):
                    continue
                for dimension, demand in enumerate(instance.task_demands[placement.task]):
                    usage[dimension] += demand
            capacity = instance.pool_capacities[pool]
            usage_record = IntervalUsage(pool, left, right, tuple(usage), tuple(capacity))
            usages.append(usage_record)
            for dimension, amount in enumerate(usage):
                if amount > capacity[dimension] + eps:
                    violations.append(
                        f"Pool {pool} capacity dimension {dimension} is exceeded on [{left}, {right}): "
                        f"{amount} > {capacity[dimension]}."
                    )
    return usages


def _ready_time(instance: DAGInstance, task: int, pool: int, placements: dict[int, TaskPlacement]) -> float:
    return max(
        (
            placements[parent].end
            + instance.communication_delay_ticks(parent, task, placements[parent].pool, pool)
            for parent in instance.parents[task]
        ),
        default=0.0,
    )


def _dispatch_mask(
    instance: DAGInstance,
    unscheduled: set[int],
    completed: set[int],
    available: list[list[float]],
    placements: dict[int, TaskPlacement],
    current_time: float,
    eps: float,
) -> list[bool]:
    mask: list[bool] = []
    for task in range(instance.num_tasks):
        parents_done = set(instance.parents[task]).issubset(completed)
        for pool in range(instance.num_pools):
            capacity_ok = all(
                instance.task_demands[task][dimension] <= available[pool][dimension] + eps
                for dimension in range(instance.resource_dims)
            )
            ready = parents_done and current_time + eps >= _ready_time(instance, task, pool, placements)
            mask.append(task in unscheduled and ready and instance.compatibility[task][pool] > 0 and capacity_ok)
    return mask


def _next_event(
    instance: DAGInstance,
    unscheduled: set[int],
    completed: set[int],
    running: dict[int, TaskPlacement],
    placements: dict[int, TaskPlacement],
    current_time: float,
    eps: float,
) -> tuple[float | None, str | None]:
    completion_times = [placement.end for placement in running.values() if placement.end > current_time + eps]
    release_times: list[float] = []
    for task in unscheduled:
        if not set(instance.parents[task]).issubset(completed):
            continue
        for pool in range(instance.num_pools):
            if instance.compatibility[task][pool] <= 0:
                continue
            ready = _ready_time(instance, task, pool, placements)
            if ready > current_time + eps:
                release_times.append(ready)
    next_completion = min(completion_times) if completion_times else None
    next_release = min(release_times) if release_times else None
    if next_completion is None and next_release is None:
        return None, None
    if next_completion is not None and (next_release is None or next_completion <= next_release + eps):
        return next_completion, "running_completion"
    assert next_release is not None
    return next_release, "communication_release"


def _validate_trace(
    instance: DAGInstance,
    schedule: Schedule,
    trace: Sequence[str],
    violations: list[str],
    eps: float,
) -> int:
    """Independently validate a compact dispatch/skip trace without generator code."""
    placements = {placement.task: placement for placement in schedule.placements}
    current_time = 0.0
    unscheduled = set(range(instance.num_tasks))
    completed: set[int] = set()
    running: dict[int, TaskPlacement] = {}
    available = [list(capacity) for capacity in instance.pool_capacities]
    dispatches = 0
    passive_communication_advances = 0
    for index, event in enumerate(trace):
        mask = _dispatch_mask(instance, unscheduled, completed, available, placements, current_time, eps)
        feasible_dispatch = any(mask)
        if event == "skip":
            next_time, reason = _next_event(instance, unscheduled, completed, running, placements, current_time, eps)
            if next_time is None:
                if not running:
                    violations.append(f"Trace event {index} skips while no task is running and no communication release is pending.")
                else:
                    violations.append(f"Trace event {index} skips while no completion or communication release is pending.")
                continue
            if next_time <= current_time + eps:
                violations.append(f"Trace event {index} does not advance time.")
                continue
            current_time = next_time
            finished = [task for task, placement in running.items() if abs(placement.end - current_time) <= eps]
            for task in finished:
                placement = running.pop(task)
                completed.add(task)
                for dimension, demand in enumerate(instance.task_demands[task]):
                    available[placement.pool][dimension] += demand
            if not feasible_dispatch and reason == "communication_release":
                passive_communication_advances += 1
            continue
        parts = event.split(":")
        if len(parts) != 3 or parts[0] != "dispatch":
            violations.append(f"Trace event {index} has invalid format: {event!r}.")
            continue
        dispatches += 1
        try:
            task, pool = int(parts[1]), int(parts[2])
        except ValueError:
            violations.append(f"Trace event {index} has non-integer task/pool: {event!r}.")
            continue
        placement = placements.get(task)
        if task not in unscheduled:
            violations.append(f"Trace event {index} dispatches task {task} more than once or outside range.")
            continue
        if placement is None or placement.pool != pool:
            violations.append(f"Trace event {index} does not match schedule placement for task {task}.")
            continue
        if abs(placement.start - current_time) > eps:
            violations.append(
                f"Trace event {index} dispatches task {task} at time {current_time}, schedule starts it at {placement.start}."
            )
        if not set(instance.parents[task]).issubset(completed):
            violations.append(f"Trace event {index} dispatches task {task} before every parent completes.")
        elif current_time + eps < _ready_time(instance, task, pool, placements):
            violations.append(f"Trace event {index} dispatches task {task} before its communication release time.")
        if not 0 <= pool < instance.num_pools or instance.compatibility[task][pool] <= 0:
            violations.append(f"Trace event {index} dispatches task {task} to an invalid/incompatible pool.")
            continue
        for dimension, demand in enumerate(instance.task_demands[task]):
            if demand > available[pool][dimension] + eps:
                violations.append(f"Trace event {index} exceeds pool {pool} capacity for task {task}.")
            available[pool][dimension] -= demand
        running[task] = placement
        unscheduled.remove(task)
    if dispatches != instance.num_tasks:
        violations.append(f"Trace contains {dispatches} dispatches, expected {instance.num_tasks}.")
    action_bound = max_decode_actions(instance)
    if len(trace) > action_bound:
        violations.append(f"Trace contains {len(trace)} actions, exceeding event-driven bound {action_bound}.")
    if unscheduled:
        violations.append(f"Trace leaves tasks unscheduled: {sorted(unscheduled)}.")
    return passive_communication_advances
