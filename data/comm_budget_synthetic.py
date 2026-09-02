"""Deterministic synthetic instances for CommBudget-WeCAN V1.

The topology families reuse the paper computation-graph generator. Compute,
network, and budget attributes are sampled independently of any model or training
result, with three explicit pool roles: economy, balanced-network, performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from data.instance import DAGInstance
from data.paper_computation_graph import TopologyType, _topology_edges

SUPPORTED_TOPOLOGIES: tuple[TopologyType, ...] = (
    "layered",
    "erdos_renyi",
    "stochastic_block",
)


@dataclass(frozen=True)
class CommBudgetSyntheticConfig:
    """Configuration for 30-task, three-pool CommBudget V1 instances."""

    num_tasks: int = 30
    num_pools: int = 3
    topologies: tuple[TopologyType, ...] = SUPPORTED_TOPOLOGIES
    d_size: float = 96.0
    budget_alpha: float = 1.25

    workload_median: float = 24.0
    workload_log_sigma: float = 0.45
    workload_min: float = 8.0
    workload_max: float = 64.0

    processor_demand_values: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0)
    processor_demand_probabilities: tuple[float, ...] = (0.15, 0.25, 0.30, 0.20, 0.10)
    memory_demand_values: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0)
    memory_demand_probabilities: tuple[float, ...] = (0.15, 0.25, 0.30, 0.20, 0.10)

    capacity_bases: tuple[tuple[float, float], ...] = (
        (12.0, 12.0),
        (20.0, 16.0),
        (36.0, 24.0),
    )
    capacity_jitter_fraction: float = 0.10
    frequency_ranges: tuple[tuple[float, float], ...] = (
        (0.85, 1.00),
        (1.30, 1.55),
        (1.90, 2.25),
    )
    power_ranges: tuple[tuple[float, float], ...] = (
        (0.80, 1.00),
        (1.30, 1.65),
        (2.20, 2.80),
    )
    # Cost_i is derived from sampled target unit cost, frequency, and power so
    # that the intended speed/cost trade-off is guaranteed rather than accidental.
    unit_cost_ranges: tuple[tuple[float, float], ...] = (
        (0.65, 0.85),
        (0.95, 1.20),
        (1.35, 1.75),
    )
    bandwidth_bases: tuple[tuple[float, ...], ...] = (
        (0.0, 24.0, 12.0),
        (18.0, 0.0, 30.0),
        (14.0, 22.0, 0.0),
    )
    bandwidth_jitter_fraction: float = 0.15

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CommBudgetSyntheticConfig":
        payload = dict(values)
        for key in (
            "topologies",
            "processor_demand_values",
            "processor_demand_probabilities",
            "memory_demand_values",
            "memory_demand_probabilities",
        ):
            if key in payload:
                payload[key] = tuple(payload[key])
        for key in (
            "capacity_bases",
            "frequency_ranges",
            "power_ranges",
            "unit_cost_ranges",
            "bandwidth_bases",
        ):
            if key in payload:
                payload[key] = tuple(tuple(row) for row in payload[key])
        config = cls(**payload)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be positive.")
        if self.num_pools != 3:
            raise ValueError("CommBudget synthetic V1 currently supports exactly three pool roles.")
        if not self.topologies or any(topology not in SUPPORTED_TOPOLOGIES for topology in self.topologies):
            raise ValueError(f"topologies must be drawn from {SUPPORTED_TOPOLOGIES}.")
        if not math.isfinite(self.d_size) or self.d_size <= 0:
            raise ValueError("d_size must be finite and positive.")
        if not math.isfinite(self.budget_alpha) or self.budget_alpha < 1.0:
            raise ValueError("budget_alpha must be finite and at least 1.0.")
        if not (0 <= self.capacity_jitter_fraction < 1 and 0 <= self.bandwidth_jitter_fraction < 1):
            raise ValueError("Jitter fractions must be in [0, 1).")
        if not (
            0 < self.workload_min <= self.workload_median <= self.workload_max
            and self.workload_log_sigma >= 0
        ):
            raise ValueError("Workload distribution parameters are invalid.")
        self._validate_discrete_distribution(
            "processor demand", self.processor_demand_values, self.processor_demand_probabilities
        )
        self._validate_discrete_distribution(
            "memory demand", self.memory_demand_values, self.memory_demand_probabilities
        )
        for name, ranges in (
            ("frequency_ranges", self.frequency_ranges),
            ("power_ranges", self.power_ranges),
            ("unit_cost_ranges", self.unit_cost_ranges),
        ):
            if len(ranges) != self.num_pools:
                raise ValueError(f"{name} must have one range per pool.")
            for lower, upper in ranges:
                if not (0 < lower <= upper and math.isfinite(lower) and math.isfinite(upper)):
                    raise ValueError(f"Every {name} range must be finite, positive, and ordered.")
        if len(self.capacity_bases) != self.num_pools or any(len(row) != 2 for row in self.capacity_bases):
            raise ValueError("capacity_bases must have shape [num_pools, 2].")
        maximum_demands = (max(self.processor_demand_values), max(self.memory_demand_values))
        for capacities in self.capacity_bases:
            if any(value * (1 - self.capacity_jitter_fraction) < demand for value, demand in zip(capacities, maximum_demands)):
                raise ValueError("Every task demand must fit every pool at the minimum sampled capacity.")
        if len(self.bandwidth_bases) != self.num_pools:
            raise ValueError("bandwidth_bases must be square [num_pools, num_pools].")
        for source, row in enumerate(self.bandwidth_bases):
            if len(row) != self.num_pools:
                raise ValueError("bandwidth_bases must be square [num_pools, num_pools].")
            for target, value in enumerate(row):
                if source == target and value != 0:
                    raise ValueError("Bandwidth diagonal must be zero.")
                if source != target and value <= 0:
                    raise ValueError("Cross-pool base bandwidth must be positive.")

    @staticmethod
    def _validate_discrete_distribution(
        name: str,
        values: Sequence[float],
        probabilities: Sequence[float],
    ) -> None:
        if len(values) != len(probabilities) or not values:
            raise ValueError(f"{name} values and probabilities must have equal non-zero length.")
        if any(value <= 0 for value in values) or any(probability < 0 for probability in probabilities):
            raise ValueError(f"{name} values must be positive and probabilities non-negative.")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name} probabilities must sum to one.")


class CommBudgetSyntheticGenerator:
    """Seeded generator with a deterministic sequential RNG stream."""

    def __init__(self, config: CommBudgetSyntheticConfig, seed: int) -> None:
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate(self, topology: TopologyType, name: str) -> DAGInstance:
        cfg = self.config
        if topology not in cfg.topologies:
            raise ValueError(f"Topology {topology!r} is not enabled in this configuration.")

        workloads = np.clip(
            self.rng.lognormal(math.log(cfg.workload_median), cfg.workload_log_sigma, cfg.num_tasks),
            cfg.workload_min,
            cfg.workload_max,
        )
        processor = self.rng.choice(
            cfg.processor_demand_values,
            size=cfg.num_tasks,
            p=cfg.processor_demand_probabilities,
        )
        memory = self.rng.choice(
            cfg.memory_demand_values,
            size=cfg.num_tasks,
            p=cfg.memory_demand_probabilities,
        )
        demands = np.stack((processor, memory), axis=1)

        capacities = np.asarray(cfg.capacity_bases) * self.rng.uniform(
            1.0 - cfg.capacity_jitter_fraction,
            1.0 + cfg.capacity_jitter_fraction,
            size=(cfg.num_pools, 2),
        )
        frequencies = self._sample_ranges(cfg.frequency_ranges)
        powers = self._sample_ranges(cfg.power_ranges)
        target_unit_costs = self._sample_ranges(cfg.unit_cost_ranges)
        costs = target_unit_costs * frequencies / powers

        bandwidth = np.zeros((cfg.num_pools, cfg.num_pools), dtype=float)
        for source in range(cfg.num_pools):
            for target in range(cfg.num_pools):
                if source == target:
                    continue
                bandwidth[source, target] = cfg.bandwidth_bases[source][target] * self.rng.uniform(
                    1.0 - cfg.bandwidth_jitter_fraction,
                    1.0 + cfg.bandwidth_jitter_fraction,
                )

        workloads = np.round(workloads, 6)
        capacities = np.round(capacities, 6)
        frequencies = np.round(frequencies, 6)
        powers = np.round(powers, 6)
        costs = np.round(costs, 6)
        bandwidth = np.round(bandwidth, 6)
        edges = tuple(_topology_edges(cfg.num_tasks, topology, self.rng))

        c_min = float(sum(
            min(
                workloads[task] / frequencies[pool] * powers[pool] * costs[pool]
                for pool in range(cfg.num_pools)
            )
            for task in range(cfg.num_tasks)
        ))
        instance = DAGInstance(
            name=name,
            # Retained only for the legacy schema; CommBudget execution uses workloads.
            task_durations=tuple(float(value) for value in workloads),
            task_demands=tuple(tuple(float(value) for value in row) for row in demands),
            pool_capacities=tuple(tuple(float(value) for value in row) for row in capacities),
            compatibility=tuple(tuple(1.0 for _ in range(cfg.num_pools)) for _ in range(cfg.num_tasks)),
            edges=edges,
            bandwidth=tuple(tuple(float(value) for value in row) for row in bandwidth),
            task_workloads=tuple(float(value) for value in workloads),
            pool_frequencies=tuple(float(value) for value in frequencies),
            pool_powers=tuple(float(value) for value in powers),
            pool_costs=tuple(float(value) for value in costs),
            d_size=cfg.d_size,
            budget=cfg.budget_alpha * c_min,
        )
        instance.validate()
        return instance

    def _sample_ranges(self, ranges: Sequence[tuple[float, float]]) -> np.ndarray:
        return np.asarray([self.rng.uniform(lower, upper) for lower, upper in ranges], dtype=float)


def generate_comm_budget_dataset(
    config: CommBudgetSyntheticConfig,
    count: int,
    seed: int,
    prefix: str,
) -> list[DAGInstance]:
    """Generate a balanced deterministic topology cycle."""
    if count < 1:
        raise ValueError("count must be positive.")
    generator = CommBudgetSyntheticGenerator(config, seed)
    return [
        generator.generate(
            config.topologies[index % len(config.topologies)],
            f"{prefix}-{config.topologies[index % len(config.topologies)]}-{index:05d}",
        )
        for index in range(count)
    ]
