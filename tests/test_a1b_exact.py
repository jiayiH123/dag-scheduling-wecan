from __future__ import annotations

from training.a1b_exact import A1BExactConfig, audit_a1b_exact_candidate, generate_a1b_exact_instance
from oracle.exhaustive_oracle import solve_exhaustive_with_wait, solve_exhaustive_without_wait


def test_a1b_exact_generator_is_deterministic_and_structured() -> None:
    config = A1BExactConfig()
    first = generate_a1b_exact_instance(4000, config)
    repeated = generate_a1b_exact_instance(4000, config)
    assert first == repeated
    assert first.num_tasks == 6
    assert first.pool_capacities == ((7.0, 7.0), (14.0, 14.0))
    assert all(value.is_integer() for value in first.task_durations)
    assert all(value.is_integer() for row in first.task_demands for value in row)
    assert first.compatibility[1] == (1.0, 0.5)
    assert first.compatibility[2] == (1.0, 0.0)
    assert first.task_demands[1][0] / first.pool_capacities[0][0] >= 0.55
    assert {(0, 2), (2, 3), (3, 5), (1, 4)}.issubset(set(first.edges))


def test_a1b_exact_representative_has_equal_wait_optima_and_greedy_gap() -> None:
    instance = generate_a1b_exact_instance(4000)
    with_wait = solve_exhaustive_with_wait(instance)
    without_wait = solve_exhaustive_without_wait(instance)
    assert with_wait.status == without_wait.status == "optimal"
    assert with_wait.detail["complete_search"] and without_wait.detail["complete_search"]
    assert with_wait.makespan == without_wait.makespan == 8.0
    assert without_wait.detail["contains_active_wait"] is False
    assert without_wait.detail["passive_time_advance_count"] > 0
    row = audit_a1b_exact_candidate(4002)
    assert row["accepted"], row["rejection_reasons"]
    assert row["external_greedy_absolute_gap_ticks"] == 2.0
    assert 0.05 <= row["external_greedy_relative_gap"] <= 0.30
