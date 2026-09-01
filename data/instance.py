"""Data structures and deterministic random-instance generation for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import json
import math

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class EdgeCommunication:
    """Explicit Phase-2 communication metadata for one directed DAG edge."""

    source: int
    target: int
    data_size: int


@dataclass(frozen=True)
class DAGInstance:
    """Offline heterogeneous-DAG instance using the notation of the WeCAN paper.

    Communication is optional. A missing communication section deliberately retains the
    Phase-1 zero-delay semantics. When enabled, edge records and directed network
    matrices use strict integer-tick semantics validated by :meth:`validate`.
    """

    name: str
    task_durations: tuple[float, ...]
    task_demands: tuple[tuple[float, ...], ...]
    pool_capacities: tuple[tuple[float, ...], ...]
    compatibility: tuple[tuple[float, ...], ...]
    edges: tuple[tuple[int, int], ...]
    edge_communications: tuple[EdgeCommunication, ...] = ()
    bandwidth: tuple[tuple[int, ...], ...] | None = None
    latency_ticks: tuple[tuple[int, ...], ...] | None = None

    @property
    def num_tasks(self) -> int:
        return len(self.task_durations)

    @property
    def num_pools(self) -> int:
        return len(self.pool_capacities)

    @property
    def resource_dims(self) -> int:
        return len(self.pool_capacities[0])

    @property
    def communication_enabled(self) -> bool:
        return self.bandwidth is not None or self.latency_ticks is not None or bool(self.edge_communications)

    @property
    def parents(self) -> tuple[tuple[int, ...], ...]:
        values: list[list[int]] = [[] for _ in range(self.num_tasks)]
        for source, destination in self.edges:
            values[destination].append(source)
        return tuple(tuple(sorted(parent_list)) for parent_list in values)

    @property
    def children(self) -> tuple[tuple[int, ...], ...]:
        values: list[list[int]] = [[] for _ in range(self.num_tasks)]
        for source, destination in self.edges:
            values[source].append(destination)
        return tuple(tuple(sorted(child_list)) for child_list in values)

    @property
    def edge_communication_by_edge(self) -> dict[tuple[int, int], EdgeCommunication]:
        return {(record.source, record.target): record for record in self.edge_communications}

    def edge_data_size(self, source: int, target: int) -> int:
        if not self.communication_enabled:
            return 0
        try:
            return self.edge_communication_by_edge[source, target].data_size
        except KeyError as error:
            raise ValueError(f"No communication record exists for edge {source}->{target}.") from error

    def actual_duration(self, task: int, pool: int) -> float:
        coefficient = self.compatibility[task][pool]
        if coefficient <= 0:
            raise ValueError(f"Task {task} is incompatible with pool {pool}.")
        return self.task_durations[task] / coefficient

    def communication_delay_ticks(self, source: int, target: int, source_pool: int, target_pool: int) -> int:
        """Return the exact non-competitive child-release delay in integer ticks."""
        if source_pool == target_pool or not self.communication_enabled:
            return 0
        if self.bandwidth is None or self.latency_ticks is None:
            raise ValueError("Communication fields are incomplete.")
        data_size = self.edge_data_size(source, target)
        bandwidth = self.bandwidth[source_pool][target_pool]
        if data_size % bandwidth:
            raise ValueError(
                f"Communication {source}->{target} is not integral on pool direction {source_pool}->{target_pool}."
            )
        return self.latency_ticks[source_pool][target_pool] + data_size // bandwidth

    def topological_order(self) -> tuple[int, ...]:
        indegree = [0] * self.num_tasks
        children = self.children
        for _, destination in self.edges:
            indegree[destination] += 1
        ready = [task for task, value in enumerate(indegree) if value == 0]
        order: list[int] = []
        while ready:
            task = ready.pop(0)
            order.append(task)
            for child in children[task]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(order) != self.num_tasks:
            raise ValueError("The instance edges do not form a DAG.")
        return tuple(order)

    def validate(self) -> None:
        if self.num_tasks == 0 or self.num_pools == 0:
            raise ValueError("An instance needs at least one task and one pool.")
        if len(self.task_demands) != self.num_tasks or len(self.compatibility) != self.num_tasks:
            raise ValueError("Task arrays have inconsistent lengths.")
        if any(duration <= 0 for duration in self.task_durations):
            raise ValueError("All base task durations must be positive.")
        if any(len(demand) != self.resource_dims for demand in self.task_demands):
            raise ValueError("Task demand dimension differs from pool capacity dimension.")
        if any(len(row) != self.num_pools for row in self.compatibility):
            raise ValueError("Compatibility shape must be [num_tasks, num_pools].")
        for task in range(self.num_tasks):
            feasible = False
            for pool in range(self.num_pools):
                compatible = self.compatibility[task][pool] > 0
                capacity_ok = all(
                    self.task_demands[task][dimension] <= self.pool_capacities[pool][dimension] + _EPS
                    for dimension in range(self.resource_dims)
                )
                feasible = feasible or (compatible and capacity_ok)
            if not feasible:
                raise ValueError(f"Task {task} has no individually feasible compatible pool.")
        seen: set[tuple[int, int]] = set()
        for source, destination in self.edges:
            if not (0 <= source < self.num_tasks and 0 <= destination < self.num_tasks):
                raise ValueError("Edge endpoint is outside the task range.")
            if source == destination or (source, destination) in seen:
                raise ValueError("Edges must be unique non-self loops.")
            seen.add((source, destination))
        self.topological_order()
        self._validate_communication(seen)

    def _validate_communication(self, edge_set: set[tuple[int, int]]) -> None:
        if not self.communication_enabled:
            if self.edge_communications or self.bandwidth is not None or self.latency_ticks is not None:
                raise ValueError("Communication fields must be either all absent or all present.")
            return
        if self.bandwidth is None or self.latency_ticks is None:
            raise ValueError("Communication-enabled instances require edge_communications, bandwidth, and latency_ticks.")
        records: dict[tuple[int, int], EdgeCommunication] = {}
        for record in self.edge_communications:
            key = (record.source, record.target)
            if key in records:
                raise ValueError(f"Duplicate communication record for edge {record.source}->{record.target}.")
            if key not in edge_set:
                raise ValueError(f"Communication record {record.source}->{record.target} is not a DAG edge.")
            if not _is_strict_int(record.data_size) or record.data_size < 0:
                raise ValueError(f"Communication data_size for edge {record.source}->{record.target} must be a non-negative integer.")
            records[key] = record
        if len(records) != len(edge_set) or set(records) != edge_set:
            raise ValueError("Communication-enabled instances require exactly one record for every DAG edge.")
        if len(self.bandwidth) != self.num_pools or len(self.latency_ticks) != self.num_pools:
            raise ValueError("Communication matrices must have one row per pool.")
        for source_pool in range(self.num_pools):
            if len(self.bandwidth[source_pool]) != self.num_pools or len(self.latency_ticks[source_pool]) != self.num_pools:
                raise ValueError("Communication matrices must be square [num_pools, num_pools].")
            for target_pool in range(self.num_pools):
                bandwidth = self.bandwidth[source_pool][target_pool]
                latency = self.latency_ticks[source_pool][target_pool]
                if not _is_strict_int(latency) or latency < 0:
                    raise ValueError("Communication latency_ticks must be non-negative integers.")
                if not _is_strict_int(bandwidth) or bandwidth <= 0:
                    raise ValueError("Communication bandwidth must be a positive integer.")
                if source_pool == target_pool:
                    if latency != 0:
                        raise ValueError("Communication latency diagonal must be zero because same-pool communication is zero.")
                    continue
                for record in self.edge_communications:
                    if record.data_size % bandwidth:
                        raise ValueError(
                            f"Communication data_size for edge {record.source}->{record.target} must divide bandwidth "
                            f"exactly on direction {source_pool}->{target_pool}."
                        )
        for task in range(self.num_tasks):
            for pool in range(self.num_pools):
                if self.compatibility[task][pool] <= 0:
                    continue
                duration = self.actual_duration(task, pool)
                if not _is_integer_tick(duration):
                    raise ValueError("Communication-enabled instances require every compatible actual task duration to be an integer tick.")

    def to_dict(self) -> dict:
        payload = asdict(self)
        # Preserve Phase-1 serialized payloads exactly: zero-communication defaults are
        # semantic defaults, not fields retrofitted into frozen Phase-1 artifacts.
        if not self.communication_enabled:
            payload.pop("edge_communications", None)
            payload.pop("bandwidth", None)
            payload.pop("latency_ticks", None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "DAGInstance":
        communication_payload = payload.get("edge_communications", ())
        instance = cls(
            name=str(payload["name"]),
            task_durations=tuple(float(value) for value in payload["task_durations"]),
            task_demands=tuple(tuple(float(value) for value in row) for row in payload["task_demands"]),
            pool_capacities=tuple(tuple(float(value) for value in row) for row in payload["pool_capacities"]),
            compatibility=tuple(tuple(float(value) for value in row) for row in payload["compatibility"]),
            edges=tuple(tuple(int(value) for value in edge) for edge in payload["edges"]),
            edge_communications=tuple(
                EdgeCommunication(int(record["source"]), int(record["target"]), record["data_size"])
                for record in communication_payload
            ),
            bandwidth=None if payload.get("bandwidth") is None else tuple(tuple(value for value in row) for row in payload["bandwidth"]),
            latency_ticks=None if payload.get("latency_ticks") is None else tuple(tuple(value for value in row) for row in payload["latency_ticks"]),
        )
        instance.validate()
        return instance


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_integer_tick(value: float) -> bool:
    return math.isfinite(value) and abs(value - round(value)) <= _EPS


@dataclass(frozen=True)
class GeneratorConfig:
    num_tasks_min: int = 12
    num_tasks_max: int = 24
    num_pools: int = 3
    resource_dims: int = 2
    max_width: int = 5
    edge_probability: float = 0.35
    duration_low: float = 5.0
    duration_high: float = 30.0
    demand_fraction_low: float = 0.08
    demand_fraction_high: float = 0.45
    capacity_low: float = 40.0
    capacity_high: float = 100.0
    incompatibility_probability: float = 0.15
    compatibility_low: float = 0.6
    compatibility_high: float = 1.5

    def validate(self) -> None:
        if not (1 <= self.num_tasks_min <= self.num_tasks_max):
            raise ValueError("Task count range is invalid.")
        if self.num_pools < 1 or self.resource_dims < 1 or self.max_width < 1:
            raise ValueError("Pool count, resource dimensions, and max width must be positive.")
        if not 0 <= self.edge_probability <= 1:
            raise ValueError("edge_probability must be in [0, 1].")


class RandomDAGGenerator:
    """Layered random-DAG generator with guaranteed individual task feasibility."""

    def __init__(self, config: GeneratorConfig, seed: int) -> None:
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str) -> DAGInstance:
        cfg = self.config
        task_count = int(self.rng.integers(cfg.num_tasks_min, cfg.num_tasks_max + 1))
        capacities = self.rng.uniform(
            cfg.capacity_low, cfg.capacity_high, size=(cfg.num_pools, cfg.resource_dims)
        )
        durations = self.rng.uniform(cfg.duration_low, cfg.duration_high, size=task_count)
        demands = np.empty((task_count, cfg.resource_dims), dtype=float)
        for task in range(task_count):
            feasible_pool = int(self.rng.integers(0, cfg.num_pools))
            fractions = self.rng.uniform(cfg.demand_fraction_low, cfg.demand_fraction_high, size=cfg.resource_dims)
            demands[task] = fractions * capacities[feasible_pool]

        compatibility = self.rng.uniform(
            cfg.compatibility_low, cfg.compatibility_high, size=(task_count, cfg.num_pools)
        )
        incompatible = self.rng.random((task_count, cfg.num_pools)) < cfg.incompatibility_probability
        compatibility[incompatible] = 0.0
        for task in range(task_count):
            individually_feasible = np.all(demands[task] <= capacities + _EPS, axis=1)
            eligible = np.flatnonzero(individually_feasible)
            if not np.any((compatibility[task] > 0) & individually_feasible):
                pool = int(self.rng.choice(eligible))
                compatibility[task, pool] = float(self.rng.uniform(cfg.compatibility_low, cfg.compatibility_high))

        edges = self._generate_edges(task_count)
        instance = DAGInstance(
            name=name,
            task_durations=tuple(float(value) for value in durations),
            task_demands=tuple(tuple(float(value) for value in row) for row in demands),
            pool_capacities=tuple(tuple(float(value) for value in row) for row in capacities),
            compatibility=tuple(tuple(float(value) for value in row) for row in compatibility),
            edges=tuple(edges),
        )
        instance.validate()
        return instance

    def _generate_edges(self, task_count: int) -> list[tuple[int, int]]:
        cfg = self.config
        layers: list[list[int]] = []
        next_task = 0
        while next_task < task_count:
            size = min(int(self.rng.integers(1, cfg.max_width + 1)), task_count - next_task)
            layers.append(list(range(next_task, next_task + size)))
            next_task += size
        edges: set[tuple[int, int]] = set()
        for layer_index in range(1, len(layers)):
            previous = [task for layer in layers[:layer_index] for task in layer]
            current = layers[layer_index]
            for destination in current:
                parent = int(self.rng.choice(layers[layer_index - 1]))
                edges.add((parent, destination))
                for source in previous:
                    if self.rng.random() < cfg.edge_probability:
                        edges.add((source, destination))
        return sorted(edges)


def save_dataset(instances: Sequence[DAGInstance], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps([instance.to_dict() for instance in instances], indent=2), encoding="utf-8")


def load_dataset(path: str | Path) -> list[DAGInstance]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DAGInstance.from_dict(item) for item in payload]


def generate_dataset(config: GeneratorConfig, count: int, seed: int, prefix: str) -> list[DAGInstance]:
    if count < 1:
        raise ValueError("count must be positive.")
    generator = RandomDAGGenerator(config, seed)
    return [generator.generate(f"{prefix}-{index:05d}") for index in range(count)]
