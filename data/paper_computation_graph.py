"""Paper Appendix C.2 synthetic computation-graph generator.

Generates DAGInstance objects following the three topology types described in
Appendix C.2 of the WeCAN paper. Each "problem" consists of GRAPHS_PER_PROBLEM
independent sub-graphs (no cross-graph edges), each with TASKS_PER_GRAPH tasks,
sharing the same three heterogeneous resource pools.

Annotations:
  [paper]      – value explicitly stated in the paper.
  [assumption] – local implementation choice; not stated in the paper.

Paper-specified values:
  Tasks per graph        : 50
  Graphs per problem     : 10  →  ~500 tasks per DAGInstance
  Duration model         : t(v) = round(100 * m(v)) + 1,
                           m(v) from 4-component GMM with uniform weights,
                           means=(0.5, 1, 3, 5), std=(0.5, 1, 1, 1),
                           projected to non-negative
  Processor demand       : uniform discrete from {2, 4, 8, 16}
  Memory demand          : uniform discrete from {1, 2, 3}
  Task types             : {0, 1, 2} with probs (1/6, 1/6, 2/3)
  Pools (3 total):
    pool 0: type 0, capacity (16, 15)
    pool 1: type 1, capacity (12, 20)
    pool 2: type 1, capacity (64, 50)
  Compatibility (pool_type × task_type):
    pool type 0: task_type 0 → 1.0,   task_type 1 → 1/0.8,  task_type 2 → 1/1.2
    pool type 1: task_type 0 → 0,     task_type 1 → 1/1.2,  task_type 2 → 1/0.8
  Topology parameters:
    Layered               : σ_N=0.75, ρ_E=0.2, ρ_S=0.14
    Erdős–Rényi DAG       : p=0.05
    Stochastic Block DAG  : p_in=0.3, p_out=0.005

Implementation assumptions (not stated in paper):
  GMM non-negative projection : max(0, sample)
  Layered graph               : num_layers = max(2, round((1-σ_N)*N));
                                tasks assigned uniformly to layers;
                                adjacent-layer edge prob = ρ_E,
                                skip-layer (distance ≥ 2) edge prob = ρ_S;
                                each non-root task guaranteed one parent
  ER DAG                      : random topological ranks by permutation;
                                edge (i→j) added iff rank(i) < rank(j) with prob p
  Stochastic Block DAG        : B = max(2, round(sqrt(N))) blocks [assumption];
                                tasks assigned uniformly with non-empty guarantee;
                                intra-block edge prob = p_in (lower rank → higher),
                                inter-block edge prob = p_out (lower block → higher)
  Sub-graph connectivity      : no edges between sub-graphs in a problem
  No communication metadata   : phase-1 zero-delay semantics (no bandwidth/latency)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import numpy as np

from data.instance import DAGInstance

# ── Paper-specified constants ─────────────────────────────────────────────────

TASKS_PER_GRAPH: int = 50          # [paper]
GRAPHS_PER_PROBLEM: int = 10       # [paper]

# Pool definitions [paper]
POOL_TYPES: tuple[int, int, int] = (0, 1, 1)
POOL_CAPACITIES: tuple[tuple[int, int], ...] = (
    (16, 15),   # pool 0: type 0  [paper]
    (12, 20),   # pool 1: type 1  [paper]
    (64, 50),   # pool 2: type 1  [paper]
)
NUM_POOLS = len(POOL_CAPACITIES)

# Compatibility coefficients: COMPAT_BY_POOL_TYPE[pool_type][task_type] [paper]
# 0.0 = incompatible; positive = speed multiplier (actual_duration = base / coeff)
COMPAT_BY_POOL_TYPE: tuple[tuple[float, float, float], ...] = (
    (1.0,    1.0 / 0.8,  1.0 / 1.2),  # pool type 0  [paper]
    (0.0,    1.0 / 1.2,  1.0 / 0.8),  # pool type 1  [paper]
)

# Task type distribution [paper]
TASK_TYPE_PROBS: tuple[float, float, float] = (1 / 6, 1 / 6, 2 / 3)

# Duration GMM [paper]
GMM_MEANS: tuple[float, ...] = (0.5, 1.0, 3.0, 5.0)
GMM_STDS: tuple[float, ...] = (0.5, 1.0, 1.0, 1.0)
GMM_NUM_COMPONENTS: int = 4        # [paper] equal-weight components

# Topology hyper-parameters [paper]
LAYERED_SIGMA_N: float = 0.75
LAYERED_RHO_E: float = 0.2
LAYERED_RHO_S: float = 0.14
ER_P: float = 0.05
SBD_P_IN: float = 0.3
SBD_P_OUT: float = 0.005

TopologyType = Literal["layered", "erdos_renyi", "stochastic_block"]


# ── Duration / task sampling ──────────────────────────────────────────────────

def _sample_duration(rng: np.random.Generator) -> float:
    """t(v) = round(100 * m(v)) + 1 with m(v) from GMM projected to non-negative.

    [paper] formula and GMM parameters.
    [assumption] Non-negative projection = max(0, sample).
    """
    component = int(rng.integers(0, GMM_NUM_COMPONENTS))
    m = float(rng.normal(GMM_MEANS[component], GMM_STDS[component]))
    m = max(0.0, m)                   # [assumption] non-negative projection
    return float(int(round(100.0 * m)) + 1)


def _sample_task_type(rng: np.random.Generator) -> int:
    """Sample task type ∈ {0, 1, 2} with probabilities (1/6, 1/6, 2/3). [paper]"""
    return int(rng.choice(3, p=list(TASK_TYPE_PROBS)))


def _sample_demands(rng: np.random.Generator) -> tuple[int, int]:
    """(processor, memory): processor ∈ {2,4,8,16}, memory ∈ {1,2,3}. [paper]"""
    processor = int(rng.choice([2, 4, 8, 16]))
    memory = int(rng.integers(1, 4))   # {1, 2, 3}
    return processor, memory


def _compat_row(task_type: int) -> tuple[float, float, float]:
    """Compatibility coefficients for this task across the 3 pools. [paper]"""
    return tuple(                       # type: ignore[return-value]
        COMPAT_BY_POOL_TYPE[POOL_TYPES[p]][task_type]
        for p in range(NUM_POOLS)
    )


# ── Topology generators ───────────────────────────────────────────────────────

def _layered_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Layered-graph edge set for n tasks.

    [paper] σ_N=0.75, ρ_E=0.2, ρ_S=0.14.
    [assumption] num_layers = max(2, round((1-σ_N)*n)) → 13 for n=50.
      Tasks assigned uniformly to layers; empty layers filled from the largest.
      Adjacent-layer edges (distance=1) added with prob ρ_E.
      Skip-layer edges (distance≥2) added with prob ρ_S.
      Each non-root task is guaranteed at least one parent.
    """
    num_layers = max(2, round((1.0 - LAYERED_SIGMA_N) * n))  # [assumption]

    # Assign tasks to layers; repair any empty layers [assumption]
    layer_idx_per_task = rng.integers(0, num_layers, size=n).tolist()
    layers: list[list[int]] = [[] for _ in range(num_layers)]
    for task, li in enumerate(layer_idx_per_task):
        layers[li].append(task)
    for li in range(num_layers):
        if not layers[li]:
            largest = max(range(num_layers), key=lambda i: len(layers[i]))
            task = layers[largest].pop()
            layers[li].append(task)

    layer_of = [0] * n
    for li, tasks_in_layer in enumerate(layers):
        for task in tasks_in_layer:
            layer_of[task] = li

    edges: set[tuple[int, int]] = set()
    for src_li in range(num_layers):
        for tgt_li in range(src_li + 1, num_layers):
            distance = tgt_li - src_li
            prob = LAYERED_RHO_E if distance == 1 else LAYERED_RHO_S
            for src in layers[src_li]:
                for tgt in layers[tgt_li]:
                    if rng.random() < prob:
                        edges.add((src, tgt))

    # Guarantee connectivity: each non-root task has at least one parent [assumption]
    for li in range(1, num_layers):
        earlier = [t for prev_li in range(li) for t in layers[prev_li]]
        for task in layers[li]:
            if not any((p, task) in edges for p in earlier):
                parent = int(rng.choice(earlier))
                edges.add((parent, task))

    return sorted(edges)


def _er_dag_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Erdős–Rényi DAG edge set for n tasks.

    [paper] p=0.05.
    [assumption] Random topological ranks by permutation; edge (i→j) added iff
      rank(i) < rank(j) with probability p. Expected edges ≈ C(n,2)*p ≈ 61 for n=50.
    """
    perm = rng.permutation(n)
    rank = np.empty(n, dtype=int)
    for pos, task in enumerate(perm):
        rank[task] = pos

    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if rank[i] < rank[j] and rng.random() < ER_P:
                edges.append((i, j))
    return sorted(edges)


def _sbd_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Stochastic Block DAG edge set for n tasks.

    [paper] p_in=0.3, p_out=0.005.
    [assumption] B = max(2, round(sqrt(n))) blocks → 7 for n=50.
      Tasks assigned uniformly; empty blocks filled from the largest.
      DAG order = block index order; within a block, random rank ordering.
      Intra-block edge (lower rank→higher rank) with prob p_in.
      Inter-block edge (lower block→higher block) with prob p_out.
    """
    b = max(2, round(n ** 0.5))      # [assumption]

    block_idx_per_task = rng.integers(0, b, size=n).tolist()
    blocks: list[list[int]] = [[] for _ in range(b)]
    for task, bi in enumerate(block_idx_per_task):
        blocks[bi].append(task)
    for bi in range(b):
        if not blocks[bi]:
            largest = max(range(b), key=lambda i: len(blocks[i]))
            task = blocks[largest].pop()
            blocks[bi].append(task)

    block_of = [0] * n
    intra_rank = [0] * n
    for bi, block_tasks in enumerate(blocks):
        perm = rng.permutation(len(block_tasks)).tolist()   # [assumption]
        for rank, local_idx in enumerate(perm):
            task = block_tasks[local_idx]
            block_of[task] = bi
            intra_rank[task] = rank

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bi, bj = block_of[i], block_of[j]
            if bi == bj:
                if intra_rank[i] < intra_rank[j] and rng.random() < SBD_P_IN:
                    edges.add((i, j))
            elif bi < bj:
                if rng.random() < SBD_P_OUT:
                    edges.add((i, j))
    return sorted(edges)


def _topology_edges(n: int, topology: TopologyType, rng: np.random.Generator) -> list[tuple[int, int]]:
    if topology == "layered":
        return _layered_edges(n, rng)
    if topology == "erdos_renyi":
        return _er_dag_edges(n, rng)
    if topology == "stochastic_block":
        return _sbd_edges(n, rng)
    raise ValueError(f"Unknown topology: {topology}")


# ── Sub-graph and problem generators ─────────────────────────────────────────

def _generate_subgraph(
    n: int,
    topology: TopologyType,
    rng: np.random.Generator,
    task_offset: int = 0,
) -> tuple[
    list[float],                      # durations
    list[tuple[int, int]],            # demands (processor, memory)
    list[int],                        # task types
    list[tuple[float, float, float]], # compat rows
    list[tuple[int, int]],            # edges (with offset applied)
]:
    """Generate one sub-graph of n tasks with given topology.

    Returns the raw lists; the caller combines multiple sub-graphs into one
    DAGInstance with shared pool definitions.
    """
    durations = [_sample_duration(rng) for _ in range(n)]
    demands = [_sample_demands(rng) for _ in range(n)]
    task_types = [_sample_task_type(rng) for _ in range(n)]
    compat_rows = [_compat_row(t) for t in task_types]
    raw_edges = _topology_edges(n, topology, rng)
    edges = [(src + task_offset, tgt + task_offset) for src, tgt in raw_edges]
    return durations, demands, task_types, compat_rows, edges


def generate_problem(
    topology: TopologyType,
    rng: np.random.Generator,
    name: str,
    tasks_per_graph: int = TASKS_PER_GRAPH,
    graphs_per_problem: int = GRAPHS_PER_PROBLEM,
) -> DAGInstance:
    """Generate one problem: graphs_per_problem independent sub-graphs of tasks_per_graph tasks.

    [paper] Default tasks_per_graph=50, graphs_per_problem=10 → 500 tasks.
    Pool capacities and compatibility are shared across all tasks. [paper]
    Sub-graphs are independent (no cross-graph edges). [assumption: implementation]
    """
    all_durations: list[float] = []
    all_demands: list[tuple[int, int]] = []
    all_compat: list[tuple[float, float, float]] = []
    all_edges: list[tuple[int, int]] = []

    for g in range(graphs_per_problem):
        offset = g * tasks_per_graph
        durations, demands, _, compat_rows, edges = _generate_subgraph(
            tasks_per_graph, topology, rng, task_offset=offset
        )
        all_durations.extend(durations)
        all_demands.extend(demands)
        all_compat.extend(compat_rows)
        all_edges.extend(edges)

    instance = DAGInstance(
        name=name,
        task_durations=tuple(all_durations),
        task_demands=tuple(tuple(float(v) for v in d) for d in all_demands),
        pool_capacities=tuple(tuple(float(v) for v in cap) for cap in POOL_CAPACITIES),
        compatibility=tuple(row for row in all_compat),
        edges=tuple(sorted(set(all_edges))),
    )
    instance.validate()
    return instance


def generate_single_graph(
    topology: TopologyType,
    rng: np.random.Generator,
    name: str,
    n: int = TASKS_PER_GRAPH,
) -> DAGInstance:
    """Generate a single sub-graph of n tasks (without combining into a full problem).

    Useful for unit tests and small-scale checks.
    """
    durations, demands, _, compat_rows, edges = _generate_subgraph(n, topology, rng)
    instance = DAGInstance(
        name=name,
        task_durations=tuple(durations),
        task_demands=tuple(tuple(float(v) for v in d) for d in demands),
        pool_capacities=tuple(tuple(float(v) for v in cap) for cap in POOL_CAPACITIES),
        compatibility=tuple(row for row in compat_rows),
        edges=tuple(sorted(set(edges))),
    )
    instance.validate()
    return instance


# ── Inspection helpers ────────────────────────────────────────────────────────

def instance_stats(instance: DAGInstance) -> dict:
    """Return a summary dict for quick inspection of a generated instance."""
    import statistics
    durations = list(instance.task_durations)
    processor_demands = [int(d[0]) for d in instance.task_demands]
    memory_demands = [int(d[1]) for d in instance.task_demands]

    compat = instance.compatibility
    n, p = instance.num_tasks, instance.num_pools
    compatible_pairs = sum(1 for t in range(n) for pl in range(p) if compat[t][pl] > 0)

    # Task types: infer from compat row.
    # type 0 → pools 1,2 incompatible: row[1] == 0.0
    # type 1 → pool 0 coeff 1/0.8 > pool 1 coeff 1/1.2: row[0] > row[1] > 0
    # type 2 → pool 0 coeff 1/1.2 < pool 1 coeff 1/0.8: row[0] < row[1]
    type_counts = {0: 0, 1: 0, 2: 0}
    for t in range(n):
        row = compat[t]
        if row[1] == 0.0:        # type 0: incompatible with pool type 1
            type_counts[0] += 1
        elif row[0] > row[1]:    # type 1: pool 0 faster (1/0.8 > 1/1.2)
            type_counts[1] += 1
        else:                    # type 2: pool 1/2 faster (1/0.8 > 1/1.2)
            type_counts[2] += 1

    return {
        "num_tasks": n,
        "num_edges": len(instance.edges),
        "duration_mean": statistics.mean(durations),
        "duration_min": min(durations),
        "duration_max": max(durations),
        "processor_demand_dist": {v: processor_demands.count(v) for v in [2, 4, 8, 16]},
        "memory_demand_dist": {v: memory_demands.count(v) for v in [1, 2, 3]},
        "task_type_counts": type_counts,
        "task_type_fracs": {k: v / n for k, v in type_counts.items()},
        "compatible_pairs": compatible_pairs,
        "compatible_frac": compatible_pairs / (n * p),
        "num_pools": p,
    }
