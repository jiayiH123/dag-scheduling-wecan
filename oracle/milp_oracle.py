"""PuLP/CBC discrete-time Oracle for integer or exactly rational-scaled small instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import re
import tempfile

import pulp

from data.instance import DAGInstance
from oracle.common import ORACLE_EPS, OracleResult, TickInstance, make_tick_instance
from scheduler.types import Schedule, TaskPlacement
from scheduler.validator import validate_schedule


@dataclass(frozen=True)
class MilpConfig:
    max_time_scale: int = 10_000
    time_limit_seconds: float | None = None
    max_nodes: int | None = None
    optimality_tolerance: float = 1e-6
    log_dir: str | None = None


def solve_milp_oracle(instance: DAGInstance, config: MilpConfig = MilpConfig()) -> OracleResult:
    started = perf_counter()
    tick = make_tick_instance(instance, max_time_scale=config.max_time_scale)
    if tick is None:
        return OracleResult("unsupported_time_scale", None, None, None, perf_counter() - started, None, {})
    try:
        horizon, serial_schedule = _serial_upper_bound(tick)
    except ValueError as error:
        return OracleResult("infeasible", None, None, tick.time_scale, perf_counter() - started, None, {"error": str(error)})

    model = pulp.LpProblem(f"oracle_{instance.name}", pulp.LpMinimize)
    starts: dict[tuple[int, int, int], pulp.LpVariable] = {}
    for task in range(instance.num_tasks):
        for pool in range(instance.num_pools):
            duration = tick.duration_ticks[task][pool]
            if duration is None:
                continue
            for start in range(0, horizon - duration + 1):
                starts[task, start, pool] = pulp.LpVariable(f"y_{task}_{start}_{pool}", cat="Binary")
    cmax = pulp.LpVariable("Cmax_ticks", lowBound=0)
    model += cmax

    for task in range(instance.num_tasks):
        choices = [variable for (item, _, _), variable in starts.items() if item == task]
        if not choices:
            return OracleResult("infeasible", None, None, tick.time_scale, perf_counter() - started, None, {"error": f"task {task} has no tick choices"})
        model += pulp.lpSum(choices) == 1, f"assign_{task}"

    start_expression: dict[int, pulp.LpAffineExpression] = {}
    end_expression: dict[int, pulp.LpAffineExpression] = {}
    for task in range(instance.num_tasks):
        start_expression[task] = pulp.lpSum(start * variable for (item, start, _), variable in starts.items() if item == task)
        end_expression[task] = pulp.lpSum(
            (start + tick.duration_ticks[task][pool]) * variable
            for (item, start, pool), variable in starts.items()
            if item == task
        )
        model += cmax >= end_expression[task], f"cmax_{task}"
    for parent, child in instance.edges:
        if not instance.communication_enabled:
            model += start_expression[child] >= end_expression[parent], f"precedence_{parent}_{child}"
            continue
        big_m = horizon + max(
            instance.communication_delay_ticks(parent, child, source_pool, target_pool)
            for source_pool in range(instance.num_pools)
            for target_pool in range(instance.num_pools)
        )
        for source_pool in range(instance.num_pools):
            parent_assignment = pulp.lpSum(
                variable for (item, _, pool), variable in starts.items() if item == parent and pool == source_pool
            )
            if not parent_assignment:
                continue
            for target_pool in range(instance.num_pools):
                child_assignment = pulp.lpSum(
                    variable for (item, _, pool), variable in starts.items() if item == child and pool == target_pool
                )
                if not child_assignment:
                    continue
                delay = instance.communication_delay_ticks(parent, child, source_pool, target_pool)
                model += (
                    start_expression[child]
                    >= end_expression[parent] + delay - big_m * (2 - parent_assignment - child_assignment)
                ), f"communication_{parent}_{child}_{source_pool}_{target_pool}"

    for pool in range(instance.num_pools):
        for current_tick in range(horizon):
            for dimension in range(instance.resource_dims):
                active = []
                for task in range(instance.num_tasks):
                    demand = tick.resource_demands[task][dimension]
                    if demand == 0:
                        continue
                    for start in range(horizon + 1):
                        variable = starts.get((task, start, pool))
                        duration = tick.duration_ticks[task][pool]
                        if variable is not None and duration is not None and start <= current_tick < start + duration:
                            active.append(demand * variable)
                if active:
                    model += pulp.lpSum(active) <= tick.resource_capacities[pool][dimension], f"capacity_{pool}_{current_tick}_{dimension}"

    log_path = _log_path(instance.name, config.log_dir)
    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=config.time_limit_seconds,
        maxNodes=config.max_nodes,
        logPath=str(log_path),
        keepFiles=False,
    )
    try:
        pulp_status_code = model.solve(solver)
    except Exception as error:  # solver invocation errors must never be relabeled optimal.
        return OracleResult(
            "solver_error", None, None, tick.time_scale, perf_counter() - started, None,
            {"error": repr(error), "log_path": str(log_path), "horizon_ticks": horizon},
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    audit = _audit_cbc(log_text, pulp.LpStatus.get(pulp_status_code, "Unknown"), config)
    detail: dict[str, Any] = {
        "pulp_status": pulp.LpStatus.get(pulp_status_code, "Unknown"),
        "horizon_ticks": horizon,
        "time_scale": tick.time_scale,
        "log_path": str(log_path),
        "cbc_audit": audit,
        "serial_upper_bound_ticks": horizon,
    }
    schedule = _recover_schedule(tick, starts)
    if schedule is None:
        status = _status_without_schedule(audit, pulp_status_code)
        return OracleResult(status, None, None, tick.time_scale, perf_counter() - started, None, detail)
    validation = validate_schedule(instance, schedule, eps=ORACLE_EPS)
    objective = float(pulp.value(cmax)) / tick.time_scale
    detail["objective_ticks"] = float(pulp.value(cmax))
    detail["recovered_assignment"] = {str(placement.task): placement.pool for placement in schedule.placements}
    if not validation.feasible:
        return OracleResult("invalid_solution", objective, schedule, tick.time_scale, perf_counter() - started, validation, detail)
    status = "optimal" if _is_proven_optimal(audit, pulp_status_code, config) else "feasible_not_proven_optimal"
    return OracleResult(status, objective, schedule, tick.time_scale, perf_counter() - started, validation, detail)


def _serial_upper_bound(tick: TickInstance) -> tuple[int, Schedule]:
    """Independent guaranteed serial fallback: topological order, one task at a time."""
    placements: list[TaskPlacement] = []
    placements_by_task: dict[int, TaskPlacement] = {}
    current_tick = 0
    for task in tick.instance.topological_order():
        choices = [
            (duration, pool)
            for pool, duration in enumerate(tick.duration_ticks[task])
            if duration is not None and all(
                tick.resource_demands[task][dimension] <= tick.resource_capacities[pool][dimension] + ORACLE_EPS
                for dimension in range(tick.instance.resource_dims)
            )
        ]
        if not choices:
            raise ValueError(f"task {task} has no compatible individually-capable pool")
        duration, pool = min(choices, key=lambda item: (item[0], item[1]))
        ready_tick = max(
            (
                round(placements_by_task[parent].end * tick.time_scale)
                + tick.instance.communication_delay_ticks(parent, task, placements_by_task[parent].pool, pool)
                for parent in tick.instance.parents[task]
            ),
            default=0,
        )
        current_tick = max(current_tick, ready_tick)
        start = current_tick / tick.time_scale
        end = (current_tick + duration) / tick.time_scale
        placement = TaskPlacement(task, pool, start, end)
        placements.append(placement)
        placements_by_task[task] = placement
        current_tick += duration
    schedule = Schedule(tuple(placements))
    validation = validate_schedule(tick.instance, schedule, eps=ORACLE_EPS)
    if not validation.feasible:
        raise ValueError(f"serial fallback is unexpectedly infeasible: {validation.violations}")
    return current_tick, schedule


def _recover_schedule(tick: TickInstance, starts: dict[tuple[int, int, int], pulp.LpVariable]) -> Schedule | None:
    placements: list[TaskPlacement] = []
    for task in range(tick.instance.num_tasks):
        selected = [
            (start, pool)
            for (item, start, pool), variable in starts.items()
            if item == task and variable.value() is not None and variable.value() > 0.5
        ]
        if len(selected) != 1:
            return None
        start_tick, pool = selected[0]
        duration = tick.duration_ticks[task][pool]
        assert duration is not None
        placements.append(
            TaskPlacement(task, pool, start_tick / tick.time_scale, (start_tick + duration) / tick.time_scale)
        )
    return Schedule(tuple(sorted(placements, key=lambda placement: placement.task)))


def _log_path(instance_name: str, log_dir: str | None) -> Path:
    if log_dir is None:
        directory = Path(tempfile.mkdtemp(prefix="phase15_cbc_"))
    else:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{instance_name}.cbc.log"


def _audit_cbc(log: str, pulp_status: str, config: MilpConfig) -> dict[str, Any]:
    lower = log.lower()
    objective = _first_number(log, r"Objective value:\s*([-+0-9.eE]+)")
    bound = _first_number(log, r"Lower bound:\s*([-+0-9.eE]+)")
    gap = _first_number(log, r"Gap:\s*([-+0-9.eE]+)%?")
    terminated = any(
        marker in lower
        for marker in ("stopped on time limit", "stopped on node limit", "stopped on solution limit", "maximum time", "maximum nodes")
    )
    proven = (
        "result - optimal solution found" in lower
        or "optimal - objective value" in lower
        or "search completed - best objective" in lower
    ) and not terminated
    abnormal = any(marker in lower for marker in ("abandoned", "unrecoverable", "exiting on user event", "fatal"))
    gap_ok = gap is None or abs(gap) <= config.optimality_tolerance * 100.0
    bound_ok = objective is None or bound is None or abs(objective - bound) <= config.optimality_tolerance
    return {
        "proven_optimal": proven,
        "time_limit_reached": "time limit" in lower or "maximum time" in lower,
        "node_limit_reached": "node limit" in lower or "maximum nodes" in lower,
        "solution_limit_reached": "solution limit" in lower,
        "abnormal_termination": abnormal,
        "incumbent": objective,
        "best_bound": bound,
        "gap": gap,
        "gap_ok": gap_ok,
        "bound_ok": bound_ok,
        "log_present": bool(log),
        "pulp_status": pulp_status,
    }


def _first_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _is_proven_optimal(audit: dict[str, Any], pulp_status_code: int, config: MilpConfig) -> bool:
    return bool(
        audit["proven_optimal"]
        and not audit["time_limit_reached"]
        and not audit["node_limit_reached"]
        and not audit["solution_limit_reached"]
        and not audit["abnormal_termination"]
        and audit["incumbent"] is not None
        and audit["gap_ok"]
        and audit["bound_ok"]
        and pulp.LpStatus.get(pulp_status_code) == "Optimal"
    )


def _status_without_schedule(audit: dict[str, Any], pulp_status_code: int) -> str:
    pulp_status = pulp.LpStatus.get(pulp_status_code, "Unknown")
    if pulp_status == "Infeasible":
        return "infeasible"
    if pulp_status == "Unbounded":
        return "unbounded"
    if audit["incumbent"] is not None or pulp_status in {"Not Solved", "Undefined", "Integer Feasible"}:
        return "feasible_not_proven_optimal"
    return "solver_error"
