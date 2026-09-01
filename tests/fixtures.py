"""Oracle-specific hand-crafted and randomized instance generators for tests."""

from __future__ import annotations

from data.instance import DAGInstance, GeneratorConfig, generate_dataset
from oracle.fixtures import handcrafted_instances


def random_tiny_instances(count: int = 50, seed: int = 2026) -> list[DAGInstance]:
    """Generate reproducible 4–7 task integer-time instances across stress buckets."""
    output: list[DAGInstance] = []
    densities = (0.15, 0.35, 0.65)
    incompatibilities = (0.0, 0.2, 0.45)
    widths = (2, 3, 4)
    for index in range(count):
        configuration = GeneratorConfig(
            num_tasks_min=4 + index % 4,
            num_tasks_max=4 + index % 4,
            num_pools=2 + (index % 2),
            resource_dims=2,
            max_width=widths[index % len(widths)],
            edge_probability=densities[index % len(densities)],
            duration_low=1.0,
            duration_high=5.0,
            demand_fraction_low=0.2,
            demand_fraction_high=0.7,
            capacity_low=4.0,
            capacity_high=8.0,
            incompatibility_probability=incompatibilities[index % len(incompatibilities)],
            compatibility_low=1.0,
            compatibility_high=1.0,
        )
        candidate = generate_dataset(configuration, 1, seed + index, f"oracle-random-{index:03d}")[0]
        # Random generator samples continuous values. Quantize explicitly for exact tick Oracle fixtures.
        quantized = DAGInstance(
            name=candidate.name,
            task_durations=tuple(float(max(1, round(value))) for value in candidate.task_durations),
            task_demands=tuple(tuple(float(max(1, round(value))) for value in row) for row in candidate.task_demands),
            pool_capacities=tuple(tuple(float(max(1, round(value))) for value in row) for row in candidate.pool_capacities),
            compatibility=candidate.compatibility,
            edges=candidate.edges,
        )
        quantized.validate()
        output.append(quantized)
    return output
