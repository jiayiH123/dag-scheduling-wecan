"""Shared immutable result types and exact rational tick conversions for Oracle modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any, Iterable, Literal

from data.instance import DAGInstance
from scheduler.types import Schedule
from scheduler.validator import ScheduleValidationResult

ORACLE_EPS = 1e-6
OracleStatus = Literal[
    "optimal",
    "feasible_not_proven_optimal",
    "infeasible",
    "unbounded",
    "unsupported_time_scale",
    "solver_error",
    "invalid_solution",
    "incomplete",
]


@dataclass(frozen=True)
class TickInstance:
    instance: DAGInstance
    time_scale: int
    duration_ticks: tuple[tuple[int | None, ...], ...]
    resource_capacities: tuple[tuple[float, ...], ...]
    resource_demands: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class OracleResult:
    status: OracleStatus
    makespan: float | None
    schedule: Schedule | None
    time_scale: int | None
    wall_seconds: float
    validator: ScheduleValidationResult | None
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.validator is not None:
            result["validator"] = self.validator.to_dict()
        if self.schedule is not None:
            result["schedule"] = {
                "makespan": self.schedule.makespan,
                "placements": [asdict(placement) for placement in self.schedule.ordered()],
            }
        return result


def make_tick_instance(instance: DAGInstance, *, time_scale: int | None = None, max_time_scale: int = 10_000) -> TickInstance | None:
    """Return exact integer tick data, or None when input cannot be exactly represented.

    Communication-enabled Phase-2.1a instances intentionally use a stricter regime:
    their validated compatible durations and every communication delay are already
    integral, and their Oracle time scale is exactly one. Phase-1 instances retain the
    existing rational-scale behavior below.
    """
    if time_scale is not None and (not isinstance(time_scale, int) or time_scale < 1):
        raise ValueError("time_scale must be a positive integer when supplied.")
    if instance.communication_enabled:
        if time_scale not in {None, 1}:
            return None
        try:
            instance.validate()
        except ValueError:
            return None
        durations: list[tuple[int | None, ...]] = []
        for task in range(instance.num_tasks):
            row: list[int | None] = []
            for pool in range(instance.num_pools):
                if instance.compatibility[task][pool] <= 0:
                    row.append(None)
                    continue
                duration = instance.actual_duration(task, pool)
                if not float(duration).is_integer() or duration <= 0:
                    return None
                row.append(int(duration))
            durations.append(tuple(row))
        return TickInstance(
            instance=instance,
            time_scale=1,
            duration_ticks=tuple(durations),
            resource_capacities=tuple(tuple(float(value) for value in row) for row in instance.pool_capacities),
            resource_demands=tuple(tuple(float(value) for value in row) for row in instance.task_demands),
        )
    values = [*instance.task_durations]
    values.extend(value for row in instance.compatibility for value in row if value > 0)
    values.extend(value for row in instance.task_demands for value in row)
    values.extend(value for row in instance.pool_capacities for value in row)
    duration_fractions: list[Fraction] = []
    for task in range(instance.num_tasks):
        for pool in range(instance.num_pools):
            coefficient = instance.compatibility[task][pool]
            if coefficient <= 0:
                continue
            duration_fractions.append(_exact_fraction(instance.task_durations[task]) / _exact_fraction(coefficient))
    fractions: list[Fraction] = []
    for value in values:
        fraction = _exact_fraction(value)
        if fraction is None:
            return None
        fractions.append(fraction)
    fractions.extend(duration_fractions)
    required_scale = 1
    for fraction in fractions:
        required_scale = _lcm(required_scale, fraction.denominator)
        if required_scale > max_time_scale:
            return None
    scale = time_scale if time_scale is not None else required_scale
    if scale > max_time_scale or scale % required_scale != 0:
        return None
    durations: list[tuple[int | None, ...]] = []
    for task in range(instance.num_tasks):
        row: list[int | None] = []
        for pool in range(instance.num_pools):
            coefficient = instance.compatibility[task][pool]
            if coefficient <= 0:
                row.append(None)
                continue
            duration_fraction = _exact_fraction(instance.task_durations[task]) / _exact_fraction(coefficient)
            ticks = duration_fraction * scale
            if ticks.denominator != 1 or ticks.numerator <= 0:
                return None
            row.append(int(ticks))
        durations.append(tuple(row))
    demand_ticks = _scaled_rows(instance.task_demands, scale)
    capacity_ticks = _scaled_rows(instance.pool_capacities, scale)
    if demand_ticks is None or capacity_ticks is None:
        return None
    return TickInstance(
        instance=instance,
        time_scale=scale,
        duration_ticks=tuple(durations),
        resource_capacities=tuple(tuple(float(value) for value in row) for row in instance.pool_capacities),
        resource_demands=tuple(tuple(float(value) for value in row) for row in instance.task_demands),
    )


def _scaled_rows(rows: Iterable[Iterable[float]], scale: int) -> tuple[tuple[int, ...], ...] | None:
    output: list[tuple[int, ...]] = []
    for row in rows:
        scaled: list[int] = []
        for value in row:
            ticks = _exact_fraction(value) * scale
            if ticks.denominator != 1:
                return None
            scaled.append(int(ticks))
        output.append(tuple(scaled))
    return tuple(output)


def _exact_fraction(value: float) -> Fraction | None:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not decimal.is_finite():
        return None
    return Fraction(decimal)


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right)
