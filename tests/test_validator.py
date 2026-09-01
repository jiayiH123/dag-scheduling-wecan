from __future__ import annotations

from scheduler.types import Schedule, TaskPlacement
from scheduler.validator import validate_schedule
from tests.fixtures import handcrafted_instances


def test_validator_accepts_half_open_boundary_handoff() -> None:
    instance = handcrafted_instances()["single_chain"]
    schedule = Schedule((
        TaskPlacement(0, 0, 0.0, 2.0),
        TaskPlacement(1, 0, 2.0, 5.0),
        TaskPlacement(2, 0, 5.0, 7.0),
    ))
    result = validate_schedule(instance, schedule)
    assert result.feasible, result.violations


def test_validator_rejects_missing_duplicate_and_capacity_interval_peak() -> None:
    instance = handcrafted_instances()["capacity_forces_delay"]
    schedule = Schedule((
        TaskPlacement(0, 0, 0.0, 4.0),
        TaskPlacement(0, 0, 0.0, 4.0),
        TaskPlacement(1, 0, 0.0, 3.0),
    ))
    result = validate_schedule(instance, schedule)
    assert not result.feasible
    assert any("Duplicate" in item for item in result.violations)
    assert any("Missing" in item for item in result.violations)
    assert any("capacity" in item for item in result.violations)


def test_validator_rejects_invalid_trace_skip_and_time_regression() -> None:
    instance = handcrafted_instances()["single_chain"]
    schedule = Schedule((
        TaskPlacement(0, 0, 0.0, 2.0),
        TaskPlacement(1, 0, 2.0, 5.0),
        TaskPlacement(2, 0, 5.0, 7.0),
    ))
    result = validate_schedule(instance, schedule, trace=("skip", "dispatch:0:0", "skip", "dispatch:1:0", "skip", "dispatch:2:0"))
    assert not result.feasible
    assert any("skips while no task" in item for item in result.violations)
