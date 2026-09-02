from __future__ import annotations

import numpy as np
import pytest
import torch

from data.instance import DAGInstance
from data.paper_computation_graph import generate_single_graph
from environment.config import load_yaml
from models.wecan import PoolNetworkEncoder, WeCAN, WeCANConfig
from scheduler.generator import SkipExtendedGenerator
from scheduler.types import Schedule, TaskPlacement, validate_schedule as validate_schedule_basic
from scheduler.validator import validate_schedule


def test_comm_budget_config_explicitly_enables_v1_without_training_changes() -> None:
    configuration = load_yaml("configs/comm_budget_wecan_v1.yaml")
    assert WeCANConfig(**configuration["model"]).comm_budget_enabled
    assert "training" not in configuration


def _two_task_budget_instance(*, budget: float) -> DAGInstance:
    instance = DAGInstance(
        name="comm-budget-two-task",
        task_durations=(4.0, 4.0),  # legacy field retained but ignored in V1
        task_demands=((1.0,), (1.0,)),
        pool_capacities=((2.0,), (2.0,)),
        compatibility=((1.0, 1.0), (1.0, 1.0)),
        edges=(),
        bandwidth=((0.0, 4.0), (4.0, 0.0)),
        task_workloads=(4.0, 4.0),
        pool_frequencies=(1.0, 2.0),
        pool_powers=(1.0, 1.0),
        pool_costs=(1.0, 1.0),
        d_size=8.0,
        budget=budget,
    )
    instance.validate()
    return instance


def _communication_instance() -> DAGInstance:
    instance = DAGInstance(
        name="comm-budget-multi-parent",
        task_durations=(100.0, 100.0, 100.0),  # ignored by V1 execution semantics
        task_demands=((1.0,), (1.0,), (1.0,)),
        pool_capacities=((2.0,), (2.0,)),
        compatibility=((1.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
        edges=((0, 2), (1, 2)),
        bandwidth=((0.0, 6.0), (3.0, 0.0)),
        task_workloads=(6.0, 28.0, 4.0),
        pool_frequencies=(2.0, 4.0),
        pool_powers=(2.0, 4.0),
        pool_costs=(3.0, 5.0),
        d_size=12.0,
        budget=1000.0,
    )
    instance.validate()
    return instance


def test_same_pool_and_cross_pool_communication_follow_v1_formula() -> None:
    instance = _communication_instance()
    assert instance.communication_delay_ticks(0, 2, 0, 0) == 0.0
    assert instance.communication_delay_ticks(0, 2, 0, 1) == 2.0
    assert instance.communication_delay_ticks(1, 2, 1, 0) == 4.0


def test_multi_parent_ready_time_uses_maximum_release_and_validator_enforces_it() -> None:
    instance = _communication_instance()
    parents = {
        0: TaskPlacement(0, 0, 0.0, 3.0),
        1: TaskPlacement(1, 1, 0.0, 7.0),
    }
    assert SkipExtendedGenerator._ready_time(instance, 2, 0, parents) == 11.0

    too_early = Schedule((*parents.values(), TaskPlacement(2, 0, 10.0, 12.0)))
    result = validate_schedule(instance, too_early)
    assert not result.feasible
    assert any("communication release is 11.0" in violation for violation in result.violations)
    assert not validate_schedule_basic(instance, too_early).feasible

    on_release = Schedule((*parents.values(), TaskPlacement(2, 0, 11.0, 13.0)))
    assert validate_schedule(instance, on_release).feasible

    scores = torch.tensor([[30.0, 0.0], [0.0, 20.0], [10.0, -100.0]])
    trace = SkipExtendedGenerator().decode(
        instance,
        scores,
        torch.tensor([0.1, 0.1, 1.0]),
        mode="greedy",
        allow_active_wait=False,
    )
    assert trace.schedule.placement_for(2).start == 11.0
    assert validate_schedule(instance, trace.schedule, trace=trace.decisions).feasible


def test_workload_frequency_duration_cost_and_expanded_pool_features() -> None:
    instance = _communication_instance()
    assert DAGInstance.from_dict(instance.to_dict()) == instance
    assert instance.actual_duration(0, 0) == 3.0
    assert instance.actual_duration(0, 1) == 1.5
    assert instance.task_pool_cost(0, 0) == 18.0
    assert instance.task_pool_cost(0, 1) == 30.0
    assert instance.pool_features(0) == (2.0, 2.0, 2.0, 3.0, 3.0)


def test_pool_network_encoder_feeds_expanded_pool_embeddings_into_weca() -> None:
    instance = _communication_instance()
    config = WeCANConfig(
        profile="smoke",
        high_dim=16,
        low_dim=16,
        weca_heads=4,
        ldd_heads=4,
        ldd_layers=1,
        alternating_weca_layers=1,
        comm_budget_enabled=True,
    )
    model = WeCAN(instance.resource_dims, config)
    output = model(instance)
    assert model.pool_embedder.layers[0].in_features == instance.resource_dims + 4
    assert isinstance(model.pool_network_encoder, PoolNetworkEncoder)
    assert output.task_pool_scores.shape == (instance.num_tasks, instance.num_pools)
    assert torch.isfinite(output.task_pool_scores).all()


def test_budget_exactly_feasible_is_decoded_and_validated() -> None:
    instance = _two_task_budget_instance(budget=4.0)
    scores = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
    trace = SkipExtendedGenerator().decode(
        instance,
        scores,
        torch.tensor([0.1, 0.1, 1.0]),
        mode="greedy",
        allow_active_wait=False,
    )
    result = validate_schedule(instance, trace.schedule, trace=trace.decisions)
    assert result.feasible, result.violations
    assert result.budget_checked and result.total_cost == pytest.approx(4.0)
    assert trace.spent_budget == pytest.approx(4.0)
    assert trace.remaining_budget == pytest.approx(0.0)


def test_budget_infeasible_instance_has_no_decode_and_schedule_is_rejected() -> None:
    instance = _two_task_budget_instance(budget=3.0)
    scores = torch.zeros((2, 2))
    with pytest.raises(RuntimeError, match="budget-infeasible"):
        SkipExtendedGenerator().decode(
            instance,
            scores,
            torch.tensor([0.1, 0.1, 1.0]),
            mode="greedy",
            allow_active_wait=False,
        )
    over_budget = Schedule((
        TaskPlacement(0, 1, 0.0, 2.0),
        TaskPlacement(1, 1, 0.0, 2.0),
    ))
    result = validate_schedule(instance, over_budget)
    assert not result.feasible
    assert any("exceeds budget" in violation for violation in result.violations)
    assert not validate_schedule_basic(instance, over_budget).feasible


def test_budget_lower_bound_masks_an_individually_affordable_dead_end_action() -> None:
    instance = _two_task_budget_instance(budget=5.0)
    mask = SkipExtendedGenerator._dispatch_mask(
        instance,
        unscheduled={0, 1},
        completed=set(),
        available=[list(capacity) for capacity in instance.pool_capacities],
    )
    # Each pool-0 action costs 4 <= remaining budget 5, but leaves only 1 for
    # the other task whose minimum cost is 2. The lower bound masks that dead end.
    assert mask == [False, True, False, True]


def test_legacy_paper_instance_serialization_model_shape_and_semantics_are_unchanged() -> None:
    instance = generate_single_graph("layered", np.random.default_rng(2026), "legacy-paper", n=4)
    payload = instance.to_dict()
    assert not instance.comm_budget_enabled
    assert "task_workloads" not in payload and "budget" not in payload and "bandwidth" not in payload
    assert DAGInstance.from_dict(payload) == instance
    compatible_pool = next(pool for pool, value in enumerate(instance.compatibility[0]) if value > 0)
    assert instance.actual_duration(0, compatible_pool) == pytest.approx(
        instance.task_durations[0] / instance.compatibility[0][compatible_pool]
    )
    assert instance.communication_delay_ticks(0, 0, compatible_pool, compatible_pool) == 0.0
    model = WeCAN(
        instance.resource_dims,
        WeCANConfig(
            profile="smoke", high_dim=16, low_dim=16, weca_heads=4,
            ldd_heads=4, ldd_layers=1, alternating_weca_layers=1,
        ),
    )
    assert model.pool_embedder.layers[0].in_features == instance.resource_dims
    assert model.pool_network_encoder is None
    assert model(instance).task_pool_scores.shape == (instance.num_tasks, instance.num_pools)
