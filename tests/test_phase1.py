from __future__ import annotations

import torch

from baselines.algorithms import greedy_schedule, heft_schedule, random_schedule
from data.instance import DAGInstance, GeneratorConfig, generate_dataset
from models.wecan import WeCAN, WeCANConfig
from scheduler.generator import SkipExtendedGenerator, skip_log_score
from scheduler.types import validate_schedule


def _instance() -> DAGInstance:
    instance = DAGInstance(
        name="handcrafted",
        task_durations=(4.0, 5.0, 3.0, 2.0),
        task_demands=((2.0, 1.0), (3.0, 1.0), (2.0, 2.0), (1.0, 2.0)),
        pool_capacities=((4.0, 3.0), (5.0, 4.0)),
        compatibility=((1.0, 0.8), (1.2, 1.0), (0.0, 1.1), (1.0, 1.0)),
        edges=((0, 2), (1, 2), (2, 3)),
    )
    instance.validate()
    return instance


def test_random_generator_creates_valid_dags() -> None:
    instances = generate_dataset(GeneratorConfig(num_tasks_min=8, num_tasks_max=8), count=10, seed=7, prefix="test")
    for instance in instances:
        instance.validate()
        assert len(instance.topological_order()) == instance.num_tasks


def test_random_greedy_and_heft_schedules_are_feasible() -> None:
    instance = _instance()
    for schedule in (random_schedule(instance, 1), greedy_schedule(instance), heft_schedule(instance)):
        result = validate_schedule(instance, schedule)
        assert result.feasible, result.violations


def test_skip_extended_generator_honours_dependencies_capacity_and_terminates() -> None:
    instance = _instance()
    # High root scores dispatch roots at t=0; this also tests parallel capacity handling.
    scores = torch.tensor([[5.0, 1.0], [4.0, 2.0], [3.0, 6.0], [2.0, 2.0]])
    trace = SkipExtendedGenerator().decode(
        instance, scores, torch.tensor([0.5, 0.1, 1.0]), mode="greedy", track_log_probability=True
    )
    result = validate_schedule(instance, trace.schedule)
    assert result.feasible, result.violations
    assert len(trace.decisions) <= 2 * instance.num_tasks
    assert trace.log_probability is not None


def test_model_inference_is_single_forward_and_generator_does_not_recall_it() -> None:
    instance = _instance()
    model = WeCAN(resource_dims=2, config=WeCANConfig(hidden_dim=32, heads=4, ldd_layers=1, alternating_weca_layers=1))
    model.reset_forward_counter()
    output = model(instance)
    trace = SkipExtendedGenerator().decode(instance, output.task_pool_scores, output.skip_parameters, mode="greedy")
    assert model.forward_calls == 1
    assert validate_schedule(instance, trace.schedule).feasible


def test_skip_score_is_strictly_decreasing() -> None:
    parameters = torch.tensor([1.0, 0.1, 2.0])
    scores = [float(skip_log_score(parameters, k, 10)) for k in range(5)]
    assert all(left > right for left, right in zip(scores, scores[1:]))
