#!/usr/bin/env python3
"""Generate and summarize the 300-instance CommBudget V1 sanity dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.comm_budget_synthetic import CommBudgetSyntheticConfig, generate_comm_budget_dataset
from data.instance import DAGInstance, save_dataset
from environment.config import load_yaml
from scheduler.types import Schedule, TaskPlacement
from scheduler.validator import validate_schedule


def distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        raise ValueError("Cannot summarize an empty distribution.")
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def minimum_cost_sequential_schedule(instance: DAGInstance) -> Schedule:
    """Construct a capacity-safe witness using each task's cheapest pool."""
    placements: dict[int, TaskPlacement] = {}
    current_time = 0.0
    for task in instance.topological_order():
        pool = min(range(instance.num_pools), key=lambda candidate: instance.task_pool_cost(task, candidate))
        ready = max(
            (
                placements[parent].end
                + instance.communication_delay_ticks(parent, task, placements[parent].pool, pool)
                for parent in instance.parents[task]
            ),
            default=0.0,
        )
        start = max(current_time, ready)
        end = start + instance.actual_duration(task, pool)
        placements[task] = TaskPlacement(task, pool, start, end)
        current_time = end
    return Schedule(tuple(placements[task] for task in range(instance.num_tasks)))


def _upgrade_fraction(instance: DAGInstance, target_pool: int) -> float:
    assert instance.budget is not None
    tasks = range(instance.num_tasks)
    c_min = instance.minimum_cost_lower_bound(tasks)
    premiums = sorted(
        instance.task_pool_cost(task, target_pool) - instance.minimum_task_cost(task)
        for task in tasks
    )
    spent = 0.0
    upgraded = 0
    for premium in premiums:
        if c_min + spent + premium > instance.budget + 1e-9:
            break
        spent += premium
        upgraded += 1
    return upgraded / instance.num_tasks


def analyze(
    instances: list[DAGInstance],
    config: CommBudgetSyntheticConfig,
    *,
    random_assignments_per_instance: int,
    dataset_seed: int,
    assignment_seed: int,
) -> dict:
    execution_times: list[float] = []
    communication_times: list[float] = []
    c_min_values: list[float] = []
    budgets: list[float] = []
    budget_ratios: list[float] = []
    absolute_slack: list[float] = []
    fastest_assignment_ratios: list[float] = []
    balanced_assignment_ratios: list[float] = []
    economy_assignment_ratios: list[float] = []
    balanced_upgrade_fractions: list[float] = []
    performance_upgrade_fractions: list[float] = []
    frequency_ratios: list[float] = []
    power_ratios: list[float] = []
    cost_ratios: list[float] = []
    unit_cost_ratios: list[float] = []
    bandwidth_ratios: list[float] = []
    per_role: list[dict[str, list[float]]] = [
        {key: [] for key in (
            "frequency", "power", "cost", "unit_cost", "cpu_capacity", "memory_capacity",
            "mean_outgoing_bandwidth", "mean_incoming_bandwidth",
        )}
        for _ in range(config.num_pools)
    ]
    edges_by_topology: dict[str, list[float]] = {topology: [] for topology in config.topologies}
    topology_counts: Counter[str] = Counter()
    structure_failures = 0
    budget_infeasible_instances = 0
    random_assignment_count = 0
    random_assignment_feasible = 0
    rng = np.random.default_rng(assignment_seed)

    for index, instance in enumerate(instances):
        topology = config.topologies[index % len(config.topologies)]
        topology_counts[topology] += 1
        edges_by_topology[topology].append(float(len(instance.edges)))
        try:
            instance.validate()
            if len(instance.topological_order()) != instance.num_tasks:
                structure_failures += 1
        except ValueError:
            structure_failures += 1

        for task in range(instance.num_tasks):
            for pool in range(instance.num_pools):
                execution_times.append(instance.actual_duration(task, pool))
        cross_bandwidth: list[float] = []
        for source in range(instance.num_pools):
            for target in range(instance.num_pools):
                if source == target:
                    continue
                communication_times.append(instance.communication_delay_ticks(0, 0, source, target))
                assert instance.bandwidth is not None
                cross_bandwidth.append(instance.bandwidth[source][target])

        assert instance.budget is not None
        assert instance.pool_frequencies is not None
        c_min = instance.minimum_cost_lower_bound(range(instance.num_tasks))
        c_min_values.append(c_min)
        budgets.append(instance.budget)
        budget_ratios.append(instance.budget / c_min)
        absolute_slack.append(instance.budget - c_min)
        pool_assignment_costs = [
            sum(instance.task_pool_cost(task, pool) for task in range(instance.num_tasks))
            for pool in range(instance.num_pools)
        ]
        economy_assignment_ratios.append(pool_assignment_costs[0] / instance.budget)
        balanced_assignment_ratios.append(pool_assignment_costs[1] / instance.budget)
        fastest_pool = max(range(instance.num_pools), key=lambda pool: instance.pool_frequencies[pool])
        fastest_assignment_ratios.append(pool_assignment_costs[fastest_pool] / instance.budget)
        balanced_upgrade_fractions.append(_upgrade_fraction(instance, 1))
        performance_upgrade_fractions.append(_upgrade_fraction(instance, 2))

        witness = minimum_cost_sequential_schedule(instance)
        if not validate_schedule(instance, witness).feasible:
            budget_infeasible_instances += 1

        for _ in range(random_assignments_per_instance):
            assignment = rng.integers(0, instance.num_pools, size=instance.num_tasks)
            assignment_cost = sum(
                instance.task_pool_cost(task, int(assignment[task])) for task in range(instance.num_tasks)
            )
            random_assignment_count += 1
            random_assignment_feasible += assignment_cost <= instance.budget + 1e-9

        assert instance.pool_powers is not None and instance.pool_costs is not None
        unit_costs = instance.pool_unit_costs
        frequency_ratios.append(max(instance.pool_frequencies) / min(instance.pool_frequencies))
        power_ratios.append(max(instance.pool_powers) / min(instance.pool_powers))
        cost_ratios.append(max(instance.pool_costs) / min(instance.pool_costs))
        unit_cost_ratios.append(max(unit_costs) / min(unit_costs))
        bandwidth_ratios.append(max(cross_bandwidth) / min(cross_bandwidth))
        for pool in range(instance.num_pools):
            per_role[pool]["frequency"].append(instance.pool_frequencies[pool])
            per_role[pool]["power"].append(instance.pool_powers[pool])
            per_role[pool]["cost"].append(instance.pool_costs[pool])
            per_role[pool]["unit_cost"].append(unit_costs[pool])
            per_role[pool]["cpu_capacity"].append(instance.pool_capacities[pool][0])
            per_role[pool]["memory_capacity"].append(instance.pool_capacities[pool][1])
            assert instance.bandwidth is not None
            per_role[pool]["mean_outgoing_bandwidth"].append(float(np.mean([
                instance.bandwidth[pool][target]
                for target in range(instance.num_pools)
                if target != pool
            ])))
            per_role[pool]["mean_incoming_bandwidth"].append(float(np.mean([
                instance.bandwidth[source][pool]
                for source in range(instance.num_pools)
                if source != pool
            ])))

    compute = distribution(execution_times)
    communication = distribution(communication_times)
    comm_compute_ratio = communication["median"] / compute["median"]
    all_speed_ordered = all(
        instance.pool_frequencies is not None
        and instance.pool_frequencies[0] < instance.pool_frequencies[1] < instance.pool_frequencies[2]
        for instance in instances
    )
    all_unit_cost_ordered = all(
        instance.pool_unit_costs[0] < instance.pool_unit_costs[1] < instance.pool_unit_costs[2]
        for instance in instances
    )
    random_feasible_rate = random_assignment_feasible / random_assignment_count
    fastest_over_budget_rate = float(np.mean(np.asarray(fastest_assignment_ratios) > 1.0))
    balanced_over_budget_rate = float(np.mean(np.asarray(balanced_assignment_ratios) > 1.0))
    return {
        "instance_count": len(instances),
        "dataset_seed": dataset_seed,
        "assignment_sampling_seed": assignment_seed,
        "unique_d_size_values": sorted({instance.d_size for instance in instances}),
        "config": config.to_dict(),
        "topology_counts": dict(topology_counts),
        "edges_by_topology": {key: distribution(values) for key, values in edges_by_topology.items()},
        "execution_time": compute,
        "cross_pool_communication_time": communication,
        "median_communication_over_median_compute": comm_compute_ratio,
        "pool_roles": {
            role: {key: distribution(values) for key, values in per_role[index].items()}
            for index, role in enumerate(("economy", "balanced_network", "performance"))
        },
        "within_instance_heterogeneity": {
            "frequency_max_over_min": distribution(frequency_ratios),
            "power_max_over_min": distribution(power_ratios),
            "cost_max_over_min": distribution(cost_ratios),
            "unit_cost_max_over_min": distribution(unit_cost_ratios),
            "cross_bandwidth_max_over_min": distribution(bandwidth_ratios),
            "speed_ordered_instance_rate": float(all_speed_ordered),
            "unit_cost_ordered_instance_rate": float(all_unit_cost_ordered),
        },
        "budget": {
            "c_min": distribution(c_min_values),
            "budget": distribution(budgets),
            "budget_over_c_min": distribution(budget_ratios),
            "absolute_slack": distribution(absolute_slack),
            "economy_all_cost_over_budget": distribution(economy_assignment_ratios),
            "balanced_all_cost_over_budget": distribution(balanced_assignment_ratios),
            "fastest_all_cost_over_budget": distribution(fastest_assignment_ratios),
            "balanced_upgrade_task_fraction": distribution(balanced_upgrade_fractions),
            "performance_upgrade_task_fraction": distribution(performance_upgrade_fractions),
            "random_assignment_samples": random_assignment_count,
            "random_assignment_feasible_rate": random_feasible_rate,
            "balanced_all_over_budget_instance_rate": balanced_over_budget_rate,
            "fastest_all_over_budget_instance_rate": fastest_over_budget_rate,
        },
        "feasibility": {
            "structure_failure_count": structure_failures,
            "budget_infeasible_count": budget_infeasible_instances,
            "budget_infeasible_rate": budget_infeasible_instances / len(instances),
        },
        "sanity_checks": {
            "communication_non_negligible": comm_compute_ratio >= 0.10,
            "communication_not_compute_dominant": comm_compute_ratio <= 1.00,
            "budget_not_randomly_loose": random_feasible_rate < 0.50,
            "fastest_pool_not_always_affordable": fastest_over_budget_rate > 0.50,
            "no_widespread_infeasibility": budget_infeasible_instances / len(instances) <= 0.01,
            "ordered_speed_cost_tradeoff": all_speed_ordered and all_unit_cost_ordered,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/comm_budget_synthetic_v1.yaml")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-output", default="_runs/comm_budget_v1_sanity/instances_300.json")
    parser.add_argument("--summary-output", default="results/comm_budget_v1_sanity/summary.json")
    parser.add_argument("--random-assignments", type=int, default=256)
    args = parser.parse_args()

    configuration = load_yaml(args.config)
    config = CommBudgetSyntheticConfig.from_mapping(configuration["synthetic"])
    count = int(configuration.get("sanity_count", 300)) if args.count is None else args.count
    seed = int(configuration["seed"]) if args.seed is None else args.seed
    instances = generate_comm_budget_dataset(config, count, seed, "sanity")
    save_dataset(instances, args.dataset_output)
    summary = analyze(
        instances,
        config,
        random_assignments_per_instance=args.random_assignments,
        dataset_seed=seed,
        assignment_seed=seed + 10_000,
    )
    destination = Path(args.summary_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "instances": count,
        "summary": str(destination),
        "dataset": args.dataset_output,
        "comm_compute_ratio": summary["median_communication_over_median_compute"],
        "budget_infeasible_rate": summary["feasibility"]["budget_infeasible_rate"],
        "sanity_checks": summary["sanity_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
