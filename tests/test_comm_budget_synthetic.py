from __future__ import annotations

import math

from data.comm_budget_synthetic import CommBudgetSyntheticConfig, generate_comm_budget_dataset
from environment.config import load_yaml


def _config() -> CommBudgetSyntheticConfig:
    values = load_yaml("configs/comm_budget_synthetic_v1.yaml")
    return CommBudgetSyntheticConfig.from_mapping(values["synthetic"])


def test_same_seed_generates_identical_instances() -> None:
    config = _config()
    first = generate_comm_budget_dataset(config, 6, 2026, "deterministic")
    second = generate_comm_budget_dataset(config, 6, 2026, "deterministic")
    assert [instance.to_dict() for instance in first] == [instance.to_dict() for instance in second]


def test_d_size_is_global_across_the_dataset() -> None:
    config = _config()
    instances = generate_comm_budget_dataset(config, 12, 7, "global-d-size")
    assert {instance.d_size for instance in instances} == {config.d_size}


def test_bandwidth_diagonal_and_cross_pool_delay_semantics() -> None:
    instance = generate_comm_budget_dataset(_config(), 1, 11, "bandwidth")[0]
    assert instance.bandwidth is not None and instance.d_size is not None
    for source in range(instance.num_pools):
        assert instance.bandwidth[source][source] == 0.0
        assert instance.communication_delay_ticks(0, 0, source, source) == 0.0
        for target in range(instance.num_pools):
            if source == target:
                continue
            assert instance.bandwidth[source][target] > 0
            assert instance.communication_delay_ticks(0, 0, source, target) == (
                instance.d_size / instance.bandwidth[source][target]
            )


def test_c_min_matches_v1_cost_formula() -> None:
    instance = generate_comm_budget_dataset(_config(), 1, 17, "c-min")[0]
    assert instance.task_workloads is not None
    assert instance.pool_frequencies is not None
    assert instance.pool_powers is not None and instance.pool_costs is not None
    manual = sum(
        min(
            instance.task_workloads[task] / instance.pool_frequencies[pool]
            * instance.pool_powers[pool]
            * instance.pool_costs[pool]
            for pool in range(instance.num_pools)
        )
        for task in range(instance.num_tasks)
    )
    assert math.isclose(instance.minimum_cost_lower_bound(range(instance.num_tasks)), manual, rel_tol=1e-12)


def test_budget_ratio_equals_configured_alpha() -> None:
    config = _config()
    for instance in generate_comm_budget_dataset(config, 9, 23, "budget-alpha"):
        assert instance.budget is not None
        c_min = instance.minimum_cost_lower_bound(range(instance.num_tasks))
        assert math.isclose(instance.budget / c_min, config.budget_alpha, rel_tol=1e-12)


def test_generated_instances_cover_all_topologies_and_pass_structure_checks() -> None:
    config = _config()
    instances = generate_comm_budget_dataset(config, 6, 29, "structure")
    assert [instance.name.split("-")[1] for instance in instances] == [
        "layered", "erdos_renyi", "stochastic_block",
        "layered", "erdos_renyi", "stochastic_block",
    ]
    for instance in instances:
        instance.validate()
        assert instance.num_tasks == 30
        assert instance.num_pools == 3
        assert len(instance.topological_order()) == instance.num_tasks
        assert all(len(demand) == 2 for demand in instance.task_demands)
