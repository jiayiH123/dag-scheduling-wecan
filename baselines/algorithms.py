"""Random, Greedy, and HEFT baselines evaluated under the same Phase-1 constraints."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from data.instance import DAGInstance
from scheduler.generator import DecodeTrace, SkipExtendedGenerator
from scheduler.types import EPS, Schedule, TaskPlacement


def random_schedule(instance: DAGInstance, seed: int) -> Schedule:
    """Uniform random static score table decoded through the common feasible generator."""
    rng = np.random.default_rng(seed)
    scores = torch.tensor(rng.normal(size=(instance.num_tasks, instance.num_pools)), dtype=torch.float32)
    skip_parameters = torch.tensor([0.02, 1e-4, 1.0], dtype=torch.float32)
    return SkipExtendedGenerator().decode(instance, scores, skip_parameters, mode="greedy").schedule


COMMUNICATION_AWARE_MYOPIC_GREEDY_V1 = "communication-aware-myopic-greedy-v1"


def _duration_only_precommitted_scores(instance: DAGInstance) -> torch.Tensor:
    scores = torch.full((instance.num_tasks, instance.num_pools), -1e9, dtype=torch.float32)
    for task in range(instance.num_tasks):
        compatible = [
            (instance.actual_duration(task, pool), pool)
            for pool in range(instance.num_pools)
            if instance.compatibility[task][pool] > 0
        ]
        _, selected_pool = min(compatible, key=lambda item: (item[0], item[1]))
        scores[task, selected_pool] = 0.0
    return scores


def duration_only_greedy_trace(instance: DAGInstance) -> DecodeTrace:
    """Decode the fixed fastest-pool assignment under real communication releases."""
    skip_parameters = torch.tensor([1e-5, 1e-6, 1.0], dtype=torch.float32)
    return SkipExtendedGenerator().decode(instance, _duration_only_precommitted_scores(instance), skip_parameters, mode="greedy")


def duration_only_greedy_schedule(instance: DAGInstance) -> Schedule:
    """Choose each compatible pair by execution duration only, ignoring communication.

    The task--pool assignment is precommitted before decode. The shared generator only
    realizes that assignment under actual communication releases and capacity; it cannot
    repair the choice by selecting a communication-aware alternative pool.
    """
    return duration_only_greedy_trace(instance).schedule


def communication_aware_myopic_greedy_trace(instance: DAGInstance) -> DecodeTrace:
    """Run the fixed non-look-ahead communication-aware myopic greedy baseline.

    Version ``communication-aware-myopic-greedy-v1`` ranks every currently legal pair
    by ``(actual_duration, task_id, pool_id)``. The common generator supplies the exact
    communication-ready/capacity mask and is forbidden from choosing active waits.
    """
    scores = torch.empty((instance.num_tasks, instance.num_pools), dtype=torch.float32)
    for task in range(instance.num_tasks):
        for pool in range(instance.num_pools):
            if instance.compatibility[task][pool] <= 0:
                scores[task, pool] = -1e9
                continue
            # Scores are the negative lexicographic rank: duration dominates, then the
            # flattened generator action order resolves task ID, then pool ID ties.
            scores[task, pool] = -instance.actual_duration(task, pool)
    skip_parameters = torch.tensor([1e-5, 1e-6, 1.0], dtype=torch.float32)
    return SkipExtendedGenerator().decode(
        instance, scores, skip_parameters, mode="greedy", allow_active_wait=False,
    )


def communication_aware_myopic_greedy_schedule(instance: DAGInstance) -> Schedule:
    """Return the schedule for ``communication-aware-myopic-greedy-v1``."""
    return communication_aware_myopic_greedy_trace(instance).schedule


def greedy_schedule(instance: DAGInstance) -> Schedule:
    """Critical-path-first list policy with earliest-finish resource preference."""
    ranks = _upward_ranks(instance)
    scores = torch.empty((instance.num_tasks, instance.num_pools), dtype=torch.float32)
    for task in range(instance.num_tasks):
        for pool in range(instance.num_pools):
            if instance.compatibility[task][pool] <= 0:
                scores[task, pool] = -1e9
            else:
                # Larger rank first; for equal rank, prefer a shorter actual execution time.
                scores[task, pool] = ranks[task] - 1e-3 * instance.actual_duration(task, pool)
    skip_parameters = torch.tensor([1e-5, 1e-6, 1.0], dtype=torch.float32)
    return SkipExtendedGenerator().decode(instance, scores, skip_parameters, mode="greedy").schedule


def ca_heft_schedule(instance: DAGInstance) -> Schedule:
    """Capacity-aware HEFT (CA-HEFT) for multi-resource pools with parallel tasks."""
    ranks = _upward_ranks(instance)
    task_order = sorted(range(instance.num_tasks), key=lambda task: (-ranks[task], task))
    placements: dict[int, TaskPlacement] = {}
    for task in task_order:
        predecessor_finish = max((placements[parent].end for parent in instance.parents[task]), default=0.0)
        best: TaskPlacement | None = None
        for pool in range(instance.num_pools):
            if instance.compatibility[task][pool] <= 0:
                continue
            if any(
                instance.task_demands[task][dimension] > instance.pool_capacities[pool][dimension] + EPS
                for dimension in range(instance.resource_dims)
            ):
                continue
            duration = instance.actual_duration(task, pool)
            start = _earliest_capacity_feasible_start(instance, task, pool, predecessor_finish, duration, placements.values())
            candidate = TaskPlacement(task=task, pool=pool, start=start, end=start + duration)
            if best is None or (candidate.end, candidate.start, candidate.pool) < (best.end, best.start, best.pool):
                best = candidate
        if best is None:
            raise RuntimeError(f"HEFT found no feasible pool for task {task}.")
        placements[task] = best
    return Schedule(tuple(placements[task] for task in range(instance.num_tasks)))


def heft_schedule(instance: DAGInstance) -> Schedule:
    """Deprecated compatibility alias for CA-HEFT; use ca_heft_schedule."""
    return ca_heft_schedule(instance)


def standard_heft_schedule(instance: DAGInstance) -> Schedule:
    """Classical HEFT-style single-task-per-pool timeline baseline.

    Unlike CA-HEFT, this baseline deliberately treats every resource pool as a single
    exclusive processor. It is therefore conservative but feasible under cumulative
    capacity constraints when each task fits the selected pool.
    """
    ranks = _upward_ranks(instance)
    task_order = sorted(range(instance.num_tasks), key=lambda task: (-ranks[task], task))
    placements: dict[int, TaskPlacement] = {}
    for task in task_order:
        ready = max((placements[parent].end for parent in instance.parents[task]), default=0.0)
        best: TaskPlacement | None = None
        for pool in range(instance.num_pools):
            if instance.compatibility[task][pool] <= 0:
                continue
            if any(instance.task_demands[task][d] > instance.pool_capacities[pool][d] + EPS for d in range(instance.resource_dims)):
                continue
            duration = instance.actual_duration(task, pool)
            pool_jobs = sorted((placement for placement in placements.values() if placement.pool == pool), key=lambda item: item.start)
            start = _first_exclusive_gap(ready, duration, pool_jobs)
            candidate = TaskPlacement(task, pool, start, start + duration)
            if best is None or (candidate.end, candidate.start, candidate.pool) < (best.end, best.start, best.pool):
                best = candidate
        if best is None:
            raise RuntimeError(f"Standard HEFT found no feasible pool for task {task}.")
        placements[task] = best
    return Schedule(tuple(placements[task] for task in range(instance.num_tasks)))


def _first_exclusive_gap(ready: float, duration: float, placements: list[TaskPlacement]) -> float:
    cursor = ready
    for placement in placements:
        if cursor + duration <= placement.start + EPS:
            return cursor
        cursor = max(cursor, placement.end)
    return cursor
def _upward_ranks(instance: DAGInstance) -> list[float]:
    mean_duration = []
    for task in range(instance.num_tasks):
        feasible_durations = [
            instance.actual_duration(task, pool)
            for pool in range(instance.num_pools)
            if instance.compatibility[task][pool] > 0
        ]
        mean_duration.append(float(np.mean(feasible_durations)))
    ranks = [0.0] * instance.num_tasks
    for task in reversed(instance.topological_order()):
        ranks[task] = mean_duration[task] + max((ranks[child] for child in instance.children[task]), default=0.0)
    return ranks


def _earliest_capacity_feasible_start(
    instance: DAGInstance,
    task: int,
    pool: int,
    lower_bound: float,
    duration: float,
    existing: Iterable[TaskPlacement],
) -> float:
    pool_placements = [placement for placement in existing if placement.pool == pool]
    events = sorted({lower_bound, *(placement.start for placement in pool_placements), *(placement.end for placement in pool_placements)})
    candidate_times = [time for time in events if time >= lower_bound - EPS]
    if lower_bound not in candidate_times:
        candidate_times.insert(0, lower_bound)
    # A feasible insertion only changes truth values at resource-release/start events.
    for start in candidate_times:
        end = start + duration
        if _has_capacity_for_interval(instance, task, pool, start, end, pool_placements):
            return start
    # After every existing task on this pool ends, capacity is certainly available.
    return max([lower_bound, *(placement.end for placement in pool_placements)])


def _has_capacity_for_interval(
    instance: DAGInstance,
    task: int,
    pool: int,
    start: float,
    end: float,
    existing: list[TaskPlacement],
) -> bool:
    relevant = [placement for placement in existing if placement.start < end - EPS and placement.end > start + EPS]
    events = sorted({start, end, *(max(start, placement.start) for placement in relevant), *(min(end, placement.end) for placement in relevant)})
    for left, right in zip(events, events[1:]):
        if right - left <= EPS:
            continue
        midpoint = (left + right) / 2
        usage = [instance.task_demands[task][dimension] for dimension in range(instance.resource_dims)]
        for placement in relevant:
            if placement.start <= midpoint < placement.end:
                for dimension, demand in enumerate(instance.task_demands[placement.task]):
                    usage[dimension] += demand
        if any(usage[dimension] > instance.pool_capacities[pool][dimension] + EPS for dimension in range(instance.resource_dims)):
            return False
    return True
