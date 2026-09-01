"""Independent complete enumerator for 4–7 task exact-tick instances."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from data.instance import DAGInstance
from oracle.common import ORACLE_EPS, OracleResult, TickInstance, make_tick_instance
from scheduler.types import Schedule, TaskPlacement
from scheduler.validator import validate_schedule


@dataclass(frozen=True)
class ExhaustiveConfig:
    max_time_scale: int = 10_000
    node_limit: int | None = None
    time_limit_seconds: float | None = None
    symmetry_breaking: bool = True
    # True covers the full serial-SGS action space: a wait branch is retained even
    # when dispatches exist. False is the independently searched restricted non-delay
    # space, which may advance time only once no dispatch is feasible.
    allow_active_wait: bool = True


@dataclass(frozen=True)
class _Running:
    task: int
    pool: int
    end_tick: int


class _Search:
    def __init__(self, tick: TickInstance, config: ExhaustiveConfig) -> None:
        self.tick = tick
        self.instance = tick.instance
        self.config = config
        self.started = perf_counter()
        self.best_ticks = float("inf")
        self.best_schedule: Schedule | None = None
        self.best_history: tuple[str, ...] = ()
        self.best_active_wait_indices: tuple[int, ...] = ()
        self.best_passive_time_advance_indices: tuple[int, ...] = ()
        self.best_passive_communication_indices: tuple[int, ...] = ()
        self.nodes = 0
        self.pruned = 0
        self.complete = True
        self.seen: dict[tuple[Any, ...], int] = {}

    def solve(self) -> OracleResult:
        self._dfs(0, frozenset(range(self.instance.num_tasks)), frozenset(), (), (), ())
        elapsed = perf_counter() - self.started
        if self.best_schedule is None:
            return OracleResult("incomplete" if not self.complete else "infeasible", None, None, self.tick.time_scale, elapsed, None, self._detail())
        validation = validate_schedule(self.instance, self.best_schedule, trace=self.best_history, eps=ORACLE_EPS)
        status = "optimal" if self.complete and validation.feasible else ("incomplete" if not self.complete else "invalid_solution")
        detail = self._detail()
        detail["history"] = list(self.best_history)
        detail["active_wait_indices"] = list(self.best_active_wait_indices)
        detail["passive_time_advance_indices"] = list(self.best_passive_time_advance_indices)
        detail["passive_communication_advance_indices"] = list(self.best_passive_communication_indices)
        detail["contains_active_wait"] = bool(self.best_active_wait_indices)
        detail["passive_time_advance_count"] = len(self.best_passive_time_advance_indices)
        detail["passive_communication_advance_count"] = len(self.best_passive_communication_indices)
        detail["assignment"] = {str(item.task): item.pool for item in self.best_schedule.placements}
        return OracleResult(status, self.best_schedule.makespan, self.best_schedule, self.tick.time_scale, elapsed, validation, detail)

    def _dfs(
        self,
        current_tick: int,
        unscheduled: frozenset[int],
        completed: frozenset[int],
        running: tuple[_Running, ...],
        placements: tuple[TaskPlacement, ...],
        history: tuple[str, ...],
    ) -> None:
        if self._stopped():
            self.complete = False
            return
        self.nodes += 1
        if not unscheduled:
            makespan_ticks = max((round(placement.end * self.tick.time_scale) for placement in placements), default=0)
            if makespan_ticks < self.best_ticks:
                self.best_ticks = makespan_ticks
                self.best_schedule = Schedule(tuple(sorted(placements, key=lambda placement: placement.task)))
                self.best_history = history
                (
                    self.best_active_wait_indices,
                    self.best_passive_time_advance_indices,
                    self.best_passive_communication_indices,
                ) = self._time_advance_indices(history)
            return
        lower_bound = max(current_tick, self._critical_path_lower_bound(unscheduled))
        if lower_bound >= self.best_ticks:
            self.pruned += 1
            return
        usage = self._usage(running)
        key = self._state_key(current_tick, unscheduled, completed, running, usage, placements)
        previous = self.seen.get(key)
        if previous is not None and previous <= lower_bound:
            self.pruned += 1
            return
        self.seen[key] = lower_bound

        placements_by_task = {placement.task: placement for placement in placements}
        dispatched_any = False
        for task in sorted(unscheduled):
            if not set(self.instance.parents[task]).issubset(completed):
                continue
            for pool in self._candidate_pools(task, usage):
                if current_tick < self._ready_tick(task, pool, placements_by_task):
                    continue
                dispatched_any = True
                duration = self.tick.duration_ticks[task][pool]
                assert duration is not None
                placement = TaskPlacement(
                    task=task,
                    pool=pool,
                    start=current_tick / self.tick.time_scale,
                    end=(current_tick + duration) / self.tick.time_scale,
                )
                self._dfs(
                    current_tick,
                    unscheduled - {task},
                    completed,
                    tuple(sorted((*running, _Running(task, pool, current_tick + duration)), key=lambda item: (item.end_tick, item.task))),
                    (*placements, placement),
                    (*history, f"dispatch:{task}:{pool}"),
                )
        # With active waiting, retain the wait branch even while non-delay dispatches
        # exist. In the restricted no-wait variant, this branch is permitted only
        # when capacity, precedence, and communication release leave no dispatch action.
        next_tick, _ = self._next_event(current_tick, unscheduled, completed, running, placements_by_task)
        if next_tick is not None and (self.config.allow_active_wait or not dispatched_any):
            newly_completed = frozenset(item.task for item in running if item.end_tick == next_tick)
            remaining = tuple(item for item in running if item.end_tick != next_tick)
            self._dfs(
                next_tick,
                unscheduled,
                completed | newly_completed,
                remaining,
                placements,
                (*history, "skip"),
            )
        elif not dispatched_any:
            self.pruned += 1

    def _ready_tick(self, task: int, pool: int, placements: dict[int, TaskPlacement]) -> int:
        return max(
            (
                round(placements[parent].end * self.tick.time_scale)
                + self.instance.communication_delay_ticks(parent, task, placements[parent].pool, pool)
                for parent in self.instance.parents[task]
            ),
            default=0,
        )

    def _next_event(
        self,
        current_tick: int,
        unscheduled: frozenset[int],
        completed: frozenset[int],
        running: tuple[_Running, ...],
        placements: dict[int, TaskPlacement],
    ) -> tuple[int | None, str | None]:
        completion = min((item.end_tick for item in running if item.end_tick > current_tick), default=None)
        releases: list[int] = []
        for task in unscheduled:
            if not set(self.instance.parents[task]).issubset(completed):
                continue
            for pool in range(self.instance.num_pools):
                if self.tick.duration_ticks[task][pool] is None:
                    continue
                ready = self._ready_tick(task, pool, placements)
                if ready > current_tick:
                    releases.append(ready)
        release = min(releases, default=None)
        if completion is None and release is None:
            return None, None
        if completion is not None and (release is None or completion <= release):
            return completion, "running_completion"
        return release, "communication_release"

    def _candidate_pools(self, task: int, usage: tuple[tuple[float, ...], ...]) -> list[int]:
        pools: list[int] = []
        seen_signatures: set[tuple[Any, ...]] = set()
        for pool in range(self.instance.num_pools):
            if self.tick.duration_ticks[task][pool] is None:
                continue
            if any(
                usage[pool][dimension] + self.tick.resource_demands[task][dimension]
                > self.tick.resource_capacities[pool][dimension] + ORACLE_EPS
                for dimension in range(self.instance.resource_dims)
            ):
                continue
            signature = (
                tuple(self.tick.resource_capacities[pool]),
                tuple(self.instance.compatibility[item][pool] for item in range(self.instance.num_tasks)),
                usage[pool],
                None if self.instance.bandwidth is None else tuple(self.instance.bandwidth[pool]),
                None if self.instance.bandwidth is None else tuple(row[pool] for row in self.instance.bandwidth),
                None if self.instance.latency_ticks is None else tuple(self.instance.latency_ticks[pool]),
                None if self.instance.latency_ticks is None else tuple(row[pool] for row in self.instance.latency_ticks),
            )
            if self.config.symmetry_breaking and signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            pools.append(pool)
        return pools

    def _usage(self, running: tuple[_Running, ...]) -> tuple[tuple[float, ...], ...]:
        usage = [[0.0] * self.instance.resource_dims for _ in range(self.instance.num_pools)]
        for item in running:
            for dimension, demand in enumerate(self.tick.resource_demands[item.task]):
                usage[item.pool][dimension] += demand
        return tuple(tuple(row) for row in usage)

    def _state_key(
        self,
        current_tick: int,
        unscheduled: frozenset[int],
        completed: frozenset[int],
        running: tuple[_Running, ...],
        usage: tuple[tuple[float, ...], ...],
        placements: tuple[TaskPlacement, ...],
    ) -> tuple[Any, ...]:
        # Completed parent pool/end assignments affect all future communication releases.
        return (
            current_tick,
            tuple(sorted(unscheduled)),
            tuple(sorted(completed)),
            tuple((item.task, item.pool, item.end_tick) for item in running),
            usage,
            tuple((item.task, item.pool, round(item.end * self.tick.time_scale)) for item in sorted(placements, key=lambda item: item.task)),
        )

    def _time_advance_indices(self, history: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Classify advances as active, passive completion, or passive communication release."""
        current_tick = 0
        unscheduled = set(range(self.instance.num_tasks))
        completed: set[int] = set()
        running: tuple[_Running, ...] = ()
        placements: dict[int, TaskPlacement] = {}
        active_waits: list[int] = []
        passive_advances: list[int] = []
        passive_communications: list[int] = []
        for index, event in enumerate(history):
            usage = self._usage(running)
            dispatch_exists = any(
                set(self.instance.parents[task]).issubset(completed)
                and current_tick >= self._ready_tick(task, pool, placements)
                for task in unscheduled
                for pool in self._candidate_pools(task, usage)
            )
            if event == "skip":
                next_tick, reason = self._next_event(
                    current_tick, frozenset(unscheduled), frozenset(completed), running, placements,
                )
                assert next_tick is not None and reason is not None
                if dispatch_exists:
                    active_waits.append(index)
                else:
                    passive_advances.append(index)
                    if reason == "communication_release":
                        passive_communications.append(index)
                newly_completed = {item.task for item in running if item.end_tick == next_tick}
                running = tuple(item for item in running if item.end_tick != next_tick)
                completed.update(newly_completed)
                current_tick = next_tick
                continue
            _, task_text, pool_text = event.split(":")
            task, pool = int(task_text), int(pool_text)
            duration = self.tick.duration_ticks[task][pool]
            assert duration is not None
            placement = TaskPlacement(task, pool, current_tick / self.tick.time_scale, (current_tick + duration) / self.tick.time_scale)
            placements[task] = placement
            unscheduled.remove(task)
            running = tuple(sorted(
                (*running, _Running(task, pool, current_tick + duration)),
                key=lambda item: (item.end_tick, item.task),
            ))
        return tuple(active_waits), tuple(passive_advances), tuple(passive_communications)

    def _critical_path_lower_bound(self, unscheduled: frozenset[int]) -> int:
        durations = {
            task: min(duration for duration in self.tick.duration_ticks[task] if duration is not None)
            for task in unscheduled
        }
        rank: dict[int, int] = {}
        for task in reversed(self.instance.topological_order()):
            if task not in unscheduled:
                continue
            rank[task] = durations[task] + max((rank[child] for child in self.instance.children[task] if child in rank), default=0)
        return max(rank.values(), default=0)

    def _stopped(self) -> bool:
        if self.config.node_limit is not None and self.nodes >= self.config.node_limit:
            return True
        return self.config.time_limit_seconds is not None and perf_counter() - self.started >= self.config.time_limit_seconds

    def _detail(self) -> dict[str, Any]:
        return {
            "search_nodes": self.nodes,
            "pruned_nodes": self.pruned,
            "complete_search": self.complete,
            "unique_canonical_states": len(self.seen),
            "allow_active_wait": self.config.allow_active_wait,
            "communication_enabled": self.instance.communication_enabled,
        }


def solve_exhaustive_oracle(instance: DAGInstance, config: ExhaustiveConfig = ExhaustiveConfig()) -> OracleResult:
    """Solve using the complete action space selected by ``config.allow_active_wait``."""
    tick = make_tick_instance(instance, max_time_scale=config.max_time_scale)
    if tick is None:
        return OracleResult("unsupported_time_scale", None, None, None, 0.0, None, {})
    if instance.num_tasks > 7:
        return OracleResult("incomplete", None, None, tick.time_scale, 0.0, None, {"reason": "exhaustive oracle is limited to 7 tasks"})
    return _Search(tick, config).solve()


def solve_exhaustive_with_active_wait(instance: DAGInstance, config: ExhaustiveConfig = ExhaustiveConfig()) -> OracleResult:
    """Independently enumerate the full space including optional active waits."""
    return solve_exhaustive_oracle(instance, ExhaustiveConfig(
        max_time_scale=config.max_time_scale,
        node_limit=config.node_limit,
        time_limit_seconds=config.time_limit_seconds,
        symmetry_breaking=config.symmetry_breaking,
        allow_active_wait=True,
    ))


def solve_exhaustive_without_active_wait(instance: DAGInstance, config: ExhaustiveConfig = ExhaustiveConfig()) -> OracleResult:
    """Enumerate the complete non-active-wait space, retaining passive time advances."""
    return solve_exhaustive_oracle(instance, ExhaustiveConfig(
        max_time_scale=config.max_time_scale,
        node_limit=config.node_limit,
        time_limit_seconds=config.time_limit_seconds,
        symmetry_breaking=config.symmetry_breaking,
        allow_active_wait=False,
    ))


# Backwards-compatible names. These retain passive time advances; only active waits
# are included or excluded by the corresponding explicit names above.
def solve_exhaustive_with_wait(instance: DAGInstance, config: ExhaustiveConfig = ExhaustiveConfig()) -> OracleResult:
    return solve_exhaustive_with_active_wait(instance, config)


def solve_exhaustive_without_wait(instance: DAGInstance, config: ExhaustiveConfig = ExhaustiveConfig()) -> OracleResult:
    return solve_exhaustive_without_active_wait(instance, config)
