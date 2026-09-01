from __future__ import annotations

from training.diagnostics import a1_metrics
from training.diagnostic_instances import A1BScreenConfig, a1b_candidate_instance


def test_a1_metrics_keep_policy_and_external_greedy_separate() -> None:
    metrics = a1_metrics(
        oracle_makespan=18.0,
        external_greedy_makespan=18.0,
        ca_heft_makespan=18.0,
        initial_policy_greedy_makespan=38.0,
        final_policy_greedy_makespan=18.0,
        initial_sample_mean_makespan=33.375,
        final_sample_mean_makespan=18.0,
    )
    assert metrics["external_greedy_makespan"] == 18.0
    assert metrics["initial_policy_greedy_makespan"] == 38.0
    assert metrics["final_policy_greedy_makespan"] == 18.0
    assert metrics["policy_improvement_ratio"] == 20.0 / 38.0
    assert metrics["oracle_gap_closure_ratio"] == 1.0
    assert metrics["improvement_over_external_greedy"] == 0.0


def test_a1_metric_oracle_closure_is_undefined_without_initial_gap() -> None:
    metrics = a1_metrics(
        oracle_makespan=18.0,
        external_greedy_makespan=18.0,
        ca_heft_makespan=18.0,
        initial_policy_greedy_makespan=18.0,
        final_policy_greedy_makespan=18.0,
        initial_sample_mean_makespan=18.0,
        final_sample_mean_makespan=18.0,
    )
    assert metrics["oracle_gap_closure_ratio"] is None


def test_a1b_candidate_generation_is_fixed_by_seed_and_integer_tick() -> None:
    config = A1BScreenConfig()
    first = a1b_candidate_instance(3000, config)
    repeated = a1b_candidate_instance(3000, config)
    assert first == repeated
    assert first.num_tasks == 8
    assert all(value.is_integer() for value in first.task_durations)
    assert all(value.is_integer() for row in first.task_demands for value in row)
    assert all(value.is_integer() for row in first.pool_capacities for value in row)
