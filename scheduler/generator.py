"""WeCAN Algorithm-2-style skip-extended schedule realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from data.instance import DAGInstance
from scheduler.action_bounds import max_decode_actions
from scheduler.types import EPS, Schedule, TaskPlacement

DecodeMode = Literal["greedy", "sample"]


@dataclass(frozen=True)
class DecodeTrace:
    schedule: Schedule
    log_probability: torch.Tensor | None
    entropy_sum: torch.Tensor | None
    decisions: tuple[str, ...]
    skip_count: int
    active_wait_count: int
    passive_time_advance_count: int
    action_count: int
    forward_calls: int = 0
    passive_communication_advance_count: int = 0
    time_advance_reasons: tuple[str, ...] = ()
    spent_budget: float = 0.0
    remaining_budget: float | None = None


def skip_log_score(skip_parameters: torch.Tensor, decision_count: int, num_tasks: int) -> torch.Tensor:
    """Paper Eq. (7): log(alpha * exp(-gamma*k/(2n)) + beta)."""
    alpha, beta, gamma = skip_parameters.unbind(dim=-1)
    return torch.log(alpha * torch.exp(-gamma * decision_count / (2.0 * num_tasks)) + beta)


@dataclass(frozen=True)
class _TopologyCache:
    """Per-instance topology built once at the start of decode().

    Avoids the O(E) edge scan that `DAGInstance.parents` / `.children`
    perform on every property access.  The cache holds the same data as
    those properties but in frozenset form so that ``issubset`` checks
    need not construct a temporary ``set`` on each call.
    """
    parents: tuple[frozenset[int], ...]   # parents[t] = frozenset of parent task ids
    children: tuple[frozenset[int], ...]  # children[t] = frozenset of child task ids

    @staticmethod
    def build(instance: DAGInstance) -> "_TopologyCache":
        n = instance.num_tasks
        par: list[set[int]] = [set() for _ in range(n)]
        chi: list[set[int]] = [set() for _ in range(n)]
        for src, dst in instance.edges:
            par[dst].add(src)
            chi[src].add(dst)
        return _TopologyCache(
            parents=tuple(frozenset(s) for s in par),
            children=tuple(frozenset(s) for s in chi),
        )


class SkipExtendedGenerator:
    """Dynamic feasibility mask + static scores; it never invokes a neural model."""

    def decode(
        self,
        instance: DAGInstance,
        task_pool_scores: torch.Tensor,
        skip_parameters: torch.Tensor,
        mode: DecodeMode,
        generator: torch.Generator | None = None,
        track_log_probability: bool = False,
        allow_active_wait: bool = True,
    ) -> DecodeTrace:
        if task_pool_scores.shape != (instance.num_tasks, instance.num_pools):
            raise ValueError("Score tensor has the wrong [tasks, pools] shape.")
        if skip_parameters.shape != (3,):
            raise ValueError("skip_parameters must contain alpha, beta, gamma.")
        device = task_pool_scores.device
        topo = _TopologyCache.build(instance)  # build once; O(E) total instead of O(E) per access
        current_time = 0.0
        unscheduled = set(range(instance.num_tasks))
        completed: set[int] = set()
        running: list[TaskPlacement] = []
        available = [list(capacity) for capacity in instance.pool_capacities]
        placements: list[TaskPlacement] = []
        placements_by_task: dict[int, TaskPlacement] = {}
        decisions: list[str] = []
        log_probabilities: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        skip_count = 0
        active_wait_count = 0
        passive_time_advance_count = 0
        passive_communication_advance_count = 0
        time_advance_reasons: list[str] = []
        decision_count = 0
        spent_budget = 0.0

        while unscheduled:
            mask = self._dispatch_mask(
                instance, unscheduled, completed, available, placements_by_task, current_time,
                topo=topo, spent_budget=spent_budget,
            )
            flat_scores = task_pool_scores.reshape(-1)
            mask_tensor = torch.tensor(mask, device=device, dtype=torch.bool)
            feasible_dispatch = bool(mask_tensor.any().item())
            next_time, next_reason = self._next_event(
                instance, unscheduled, completed, running, placements_by_task, current_time,
                topo=topo,
            )
            active_wait_available = next_time is not None and feasible_dispatch and allow_active_wait
            passive_time_advance_available = next_time is not None and not feasible_dispatch
            skip_available = active_wait_available or passive_time_advance_available
            if not feasible_dispatch and not skip_available:
                if instance.comm_budget_enabled:
                    raise RuntimeError(
                        "No action can satisfy the CommBudget hard constraint; the remaining instance is budget-infeasible."
                    )
                raise RuntimeError("No feasible dispatch and no completion or communication-release event: input instance is invalid.")

            skip_score = skip_log_score(skip_parameters, decision_count, instance.num_tasks)
            action_scores = torch.cat((flat_scores, skip_score.reshape(1)))
            full_mask = torch.cat((mask_tensor, torch.tensor([skip_available], device=device)))
            masked_scores = action_scores.masked_fill(~full_mask, -torch.inf)
            log_distribution = torch.log_softmax(masked_scores, dim=0)
            probabilities = torch.softmax(masked_scores, dim=0)
            if track_log_probability:
                entropies.append(-(probabilities * log_distribution.masked_fill(~full_mask, 0.0)).sum())
            if mode == "greedy":
                action = int(torch.argmax(masked_scores).item())
            elif mode == "sample":
                action = int(torch.multinomial(probabilities, 1, generator=generator).item())
            else:
                raise ValueError(f"Unknown decode mode: {mode}")
            if track_log_probability:
                log_probabilities.append(log_distribution[action])

            if action == instance.num_tasks * instance.num_pools:
                assert next_time is not None and next_reason is not None
                current_time, newly_completed = self._advance_time(running, next_time)
                for task in newly_completed:
                    completed.add(task)
                self._release(instance, running, available, newly_completed)
                decisions.append("skip")
                skip_count += 1
                if feasible_dispatch:
                    active_wait_count += 1
                    time_advance_reasons.append("active_wait")
                else:
                    passive_time_advance_count += 1
                    time_advance_reasons.append(next_reason)
                    if next_reason == "communication_release":
                        passive_communication_advance_count += 1
            else:
                task = action // instance.num_pools
                pool = action % instance.num_pools
                duration = instance.actual_duration(task, pool)
                placement = TaskPlacement(task=task, pool=pool, start=current_time, end=current_time + duration)
                placements.append(placement)
                placements_by_task[task] = placement
                running.append(placement)
                unscheduled.remove(task)
                for dimension, demand in enumerate(instance.task_demands[task]):
                    available[pool][dimension] -= demand
                spent_budget += instance.task_pool_cost(task, pool)
                decisions.append(f"dispatch:{task}:{pool}")
            decision_count += 1
            if decision_count > max_decode_actions(instance):
                raise RuntimeError("The generator exceeded its event-driven decode action bound.")

        log_probability = torch.stack(log_probabilities).sum() if log_probabilities else None
        entropy_sum = torch.stack(entropies).sum() if entropies else None
        return DecodeTrace(
            schedule=Schedule(tuple(sorted(placements, key=lambda placement: placement.task))),
            log_probability=log_probability,
            entropy_sum=entropy_sum,
            decisions=tuple(decisions),
            skip_count=skip_count,
            active_wait_count=active_wait_count,
            passive_time_advance_count=passive_time_advance_count,
            action_count=decision_count,
            passive_communication_advance_count=passive_communication_advance_count,
            time_advance_reasons=tuple(time_advance_reasons),
            spent_budget=spent_budget,
            remaining_budget=None if instance.budget is None else instance.budget - spent_budget,
        )

    @staticmethod
    def _ready_time(
        instance: DAGInstance,
        task: int,
        pool: int,
        placements_by_task: dict[int, TaskPlacement],
        topo: _TopologyCache | None = None,
    ) -> float:
        parent_ids = topo.parents[task] if topo is not None else instance.parents[task]
        return max(
            (
                placements_by_task[parent].end
                + instance.communication_delay_ticks(parent, task, placements_by_task[parent].pool, pool)
                for parent in parent_ids
            ),
            default=0.0,
        )

    @classmethod
    def _dispatch_mask(
        cls,
        instance: DAGInstance,
        unscheduled: set[int],
        completed: set[int],
        available: list[list[float]],
        placements_by_task: dict[int, TaskPlacement] | None = None,
        current_time: float = 0.0,
        topo: _TopologyCache | None = None,
        spent_budget: float = 0.0,
    ) -> list[bool]:
        placements_by_task = {} if placements_by_task is None else placements_by_task
        mask: list[bool] = []
        remaining_lower_bound = instance.minimum_cost_lower_bound(unscheduled)
        for task in range(instance.num_tasks):
            if task not in unscheduled:
                mask.extend([False] * instance.num_pools)
                continue

            if topo is not None:
                parents_done = topo.parents[task].issubset(completed)
            else:
                parents_done = set(instance.parents[task]).issubset(completed)

            if not parents_done:
                mask.extend([False] * instance.num_pools)
                continue

            for pool in range(instance.num_pools):
                if instance.compatibility[task][pool] <= 0:
                    mask.append(False)
                    continue

                capacity_ok = all(
                    instance.task_demands[task][dimension] <= available[pool][dimension] + EPS
                    for dimension in range(instance.resource_dims)
                )
                if not capacity_ok:
                    mask.append(False)
                    continue

                if not cls._budget_action_feasible(
                    instance,
                    task,
                    pool,
                    spent_budget,
                    remaining_lower_bound,
                ):
                    mask.append(False)
                    continue

                ready = current_time + EPS >= cls._ready_time(
                    instance, task, pool, placements_by_task, topo=topo,
                )
                mask.append(ready)
        return mask

    @staticmethod
    def _budget_action_feasible(
        instance: DAGInstance,
        task: int,
        pool: int,
        spent_budget: float,
        remaining_lower_bound: float,
    ) -> bool:
        if not instance.comm_budget_enabled:
            return True
        assert instance.budget is not None
        action_cost = instance.task_pool_cost(task, pool)
        remaining_budget = instance.budget - spent_budget
        if action_cost > remaining_budget + EPS:
            return False
        future_lower_bound = remaining_lower_bound - instance.minimum_task_cost(task)
        return spent_budget + action_cost + future_lower_bound <= instance.budget + EPS

    @classmethod
    def _next_event(
        cls,
        instance: DAGInstance,
        unscheduled: set[int],
        completed: set[int],
        running: list[TaskPlacement],
        placements_by_task: dict[int, TaskPlacement],
        current_time: float,
        topo: _TopologyCache | None = None,
    ) -> tuple[float | None, str | None]:
        completion_times = [placement.end for placement in running if placement.end > current_time + EPS]
        release_times: list[float] = []
        for task in unscheduled:
            if topo is not None:
                parents_done = topo.parents[task].issubset(completed)
            else:
                parents_done = set(instance.parents[task]).issubset(completed)
            if not parents_done:
                continue
            for pool in range(instance.num_pools):
                if instance.compatibility[task][pool] <= 0:
                    continue
                ready = cls._ready_time(instance, task, pool, placements_by_task, topo=topo)
                if ready > current_time + EPS:
                    release_times.append(ready)
        next_completion = min(completion_times) if completion_times else None
        next_release = min(release_times) if release_times else None
        if next_completion is None and next_release is None:
            return None, None
        if next_completion is not None and (next_release is None or next_completion <= next_release + EPS):
            return next_completion, "running_completion"
        assert next_release is not None
        return next_release, "communication_release"

    @staticmethod
    def _advance_time(running: list[TaskPlacement], next_time: float | None = None) -> tuple[float, list[int]]:
        if next_time is None:
            next_time = min(placement.end for placement in running)
        completed = [placement.task for placement in running if abs(placement.end - next_time) <= EPS]
        return next_time, completed

    @staticmethod
    def _release(
        instance: DAGInstance,
        running: list[TaskPlacement],
        available: list[list[float]],
        completed: list[int],
    ) -> None:
        completed_set = set(completed)
        to_release = [placement for placement in running if placement.task in completed_set]
        running[:] = [placement for placement in running if placement.task not in completed_set]
        for placement in to_release:
            for dimension, demand in enumerate(instance.task_demands[placement.task]):
                available[placement.pool][dimension] += demand
