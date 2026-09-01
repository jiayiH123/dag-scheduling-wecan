from __future__ import annotations

from oracle.exhaustive_oracle import (
    solve_exhaustive_oracle,
    solve_exhaustive_with_active_wait,
    solve_exhaustive_with_wait,
    solve_exhaustive_without_active_wait,
    solve_exhaustive_without_wait,
)
from scheduler.validator import validate_schedule
from tests.fixtures import handcrafted_instances


def test_exhaustive_oracle_completes_handcrafted_tiny_instances() -> None:
    for name, instance in handcrafted_instances().items():
        result = solve_exhaustive_oracle(instance)
        assert result.status == "optimal", (name, result.to_dict())
        assert result.schedule is not None
        assert result.validator is not None and result.validator.feasible
        assert result.detail["complete_search"]
        assert result.detail["search_nodes"] >= 1
        assert validate_schedule(instance, result.schedule, trace=result.detail["history"]).feasible


def test_exhaustive_active_wait_fixture_explores_wait_action() -> None:
    result = solve_exhaustive_oracle(handcrafted_instances()["active_wait_counterexample"])
    assert result.status == "optimal"
    assert "skip" in result.detail["history"]


def test_exhaustive_refuses_unbounded_problem_size() -> None:
    instance = handcrafted_instances()["wide_parallel"]
    result = solve_exhaustive_oracle(instance)
    assert result.detail["complete_search"]


def test_a1c_active_wait_is_strictly_required_in_two_complete_search_spaces() -> None:
    instance = handcrafted_instances()["a1c_active_wait_six_task"]
    with_wait = solve_exhaustive_with_wait(instance)
    without_wait = solve_exhaustive_without_wait(instance)
    assert with_wait.status == "optimal", with_wait.to_dict()
    assert without_wait.status == "optimal", without_wait.to_dict()
    assert with_wait.validator is not None and with_wait.validator.feasible
    assert without_wait.validator is not None and without_wait.validator.feasible
    assert with_wait.detail["complete_search"]
    assert without_wait.detail["complete_search"]
    assert with_wait.makespan == 13.0
    assert without_wait.makespan == 22.0
    assert with_wait.makespan < without_wait.makespan
    assert with_wait.detail["contains_active_wait"]
    assert with_wait.detail["active_wait_indices"] == [1]
    assert with_wait.detail["passive_time_advance_count"] > 0


def test_without_active_wait_still_permits_passive_time_advances() -> None:
    result = solve_exhaustive_without_active_wait(handcrafted_instances()["a1c_active_wait_six_task"])
    assert result.status == "optimal"
    assert result.detail["contains_active_wait"] is False
    assert result.detail["passive_time_advance_count"] > 0
    assert solve_exhaustive_without_wait(handcrafted_instances()["a1c_active_wait_six_task"]).makespan == result.makespan
    assert solve_exhaustive_with_active_wait(handcrafted_instances()["a1c_active_wait_six_task"]).makespan < result.makespan
