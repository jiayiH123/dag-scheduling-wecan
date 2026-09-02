"""WeCAN faithful reimplementation with paper and smoke architecture profiles.

The paper profile follows Appendix F.1/F.2 where specified: 512-dimensional
initial embeddings, initial WeCA + 8-layer/16-head LDDGNN, 512→128 projections,
8-head later WeCA, Dmax=500 LDD biasing/masks, and a sigmoid skip head.
Unspecified details (projection placement and alternating WeCA count) remain explicit
configuration assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import nn

from data.instance import DAGInstance

POS_INF: Final[int] = 10**9
NEG_INF: Final[int] = -10**9


@dataclass(frozen=True)
class WeCANConfig:
    profile: str = "smoke"
    high_dim: int = 128
    low_dim: int = 128
    weca_heads: int = 8
    ldd_heads: int = 8
    ldd_layers: int = 2
    alternating_weca_layers: int = 2
    dmax: int = 500
    skip_hidden_dim: int = 64
    skip_parameterization: str = "softplus"
    minimum_skip_parameter: float = 1e-4
    comm_budget_enabled: bool = False
    # Legacy aliases accepted so existing Phase-1 checkpoints/configs load correctly.
    hidden_dim: int | None = None
    heads: int | None = None

    def __post_init__(self) -> None:
        if self.hidden_dim is not None:
            object.__setattr__(self, "high_dim", self.hidden_dim)
            object.__setattr__(self, "low_dim", self.hidden_dim)
        if self.heads is not None:
            object.__setattr__(self, "weca_heads", self.heads)
            object.__setattr__(self, "ldd_heads", self.heads)
        if self.profile not in {"paper", "smoke", "custom"}:
            raise ValueError("profile must be paper, smoke, or custom.")
        if self.profile == "paper":
            expected = (512, 128, 8, 16, 8, 500, "paper_sigmoid")
            actual = (
                self.high_dim, self.low_dim, self.weca_heads, self.ldd_heads,
                self.ldd_layers, self.dmax, self.skip_parameterization,
            )
            if actual != expected:
                raise ValueError(f"paper profile requires {expected}, received {actual}.")
        if self.high_dim % self.weca_heads or self.low_dim % self.weca_heads:
            raise ValueError("WeCA dimensions must be divisible by weca_heads.")
        if self.high_dim % self.ldd_heads:
            raise ValueError("high_dim must be divisible by ldd_heads.")
        if self.dmax < 1 or self.ldd_layers < 0 or self.alternating_weca_layers < 0:
            raise ValueError("Invalid non-negative layer count or Dmax.")
        if self.skip_parameterization not in {"paper_sigmoid", "softplus"}:
            raise ValueError("skip_parameterization must be paper_sigmoid or softplus.")


@dataclass(frozen=True)
class WeCANOutput:
    task_pool_scores: torch.Tensor
    skip_parameters: torch.Tensor


class InitialEmbedder(nn.Module):
    """Paper F.2 initial Linear + GELU embedder."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class PoolNetworkEncoder(nn.Module):
    """One lightweight directed message-passing layer over the pool network.

    For each directed source→target pair, log-bandwidth is embedded and added to
    a projected source-pool message. Each target mean-aggregates its incoming
    off-diagonal messages, followed by an output projection, residual connection,
    and LayerNorm. DAG edges are deliberately not part of this encoder.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.source_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bandwidth_embedder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pool_embeddings: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
        pool_count = pool_embeddings.shape[0]
        if bandwidth.shape != (pool_count, pool_count):
            raise ValueError("bandwidth must have shape [num_pools, num_pools].")
        # Axis order is [source_pool, target_pool, hidden].
        source_messages = self.source_projection(pool_embeddings).unsqueeze(1)
        edge_messages = self.bandwidth_embedder(torch.log1p(bandwidth).unsqueeze(-1))
        messages = torch.nn.functional.gelu(source_messages + edge_messages)
        if pool_count == 1:
            aggregated = torch.zeros_like(pool_embeddings)
        else:
            off_diagonal = ~torch.eye(pool_count, dtype=torch.bool, device=pool_embeddings.device)
            aggregated = (messages * off_diagonal.unsqueeze(-1)).sum(dim=0) / float(pool_count - 1)
        return self.norm(pool_embeddings + self.output(aggregated))


class WeightedCrossAttention(nn.Module):
    """Multi-head WeCA with compatibility multiplication outside softmax."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must divide evenly across WeCA heads.")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_nodes: torch.Tensor, context_nodes: torch.Tensor, compatibility_gate: torch.Tensor) -> torch.Tensor:
        query_count, context_count = query_nodes.shape[0], context_nodes.shape[0]
        query = self.query(query_nodes).reshape(query_count, self.heads, self.head_dim).transpose(0, 1)
        key = self.key(context_nodes).reshape(context_count, self.heads, self.head_dim).transpose(0, 1)
        value = self.value(context_nodes).reshape(context_count, self.heads, self.head_dim).transpose(0, 1)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(logits, dim=-1)
        # Equation in §IV-C: compatibility is deliberately outside normalization.
        attention = attention * compatibility_gate.unsqueeze(0)
        message = torch.matmul(attention, value).transpose(0, 1).reshape(query_count, self.hidden_dim)
        updated = self.norm(query_nodes + self.output(message))
        return self.feed_forward_norm(updated + self.feed_forward(updated))


class LDDGraphAttention(nn.Module):
    """Appendix F.1 LDD-biased, per-head-masked task attention."""

    def __init__(self, hidden_dim: int, heads: int, dmax: int = 500) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must divide evenly across LDD heads.")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dmax = dmax
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.distance_bias = nn.Embedding(2 * dmax + 1, heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        count = nodes.shape[0]
        query = self.query(nodes).reshape(count, self.heads, self.head_dim).transpose(0, 1)
        key = self.key(nodes).reshape(count, self.heads, self.head_dim).transpose(0, 1)
        value = self.value(nodes).reshape(count, self.heads, self.head_dim).transpose(0, 1)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        indices = ldd_bias_indices(distances, self.dmax)
        bias = self.distance_bias(indices).permute(2, 0, 1)
        allowed = ldd_attention_mask(distances, self.heads)
        masked_logits = (logits + bias).masked_fill(~allowed, -torch.inf)
        has_key = allowed.any(dim=-1, keepdim=True)
        # Dedicated heads may legitimately have no matching relation for a query. Their
        # attention/message is defined as zero; no forbidden self-loop is opened.
        safe_logits = torch.where(has_key, masked_logits, torch.zeros_like(masked_logits))
        attention = torch.softmax(safe_logits, dim=-1)
        attention = torch.where(allowed, attention, torch.zeros_like(attention))
        message = torch.matmul(attention, value).transpose(0, 1).reshape(count, self.hidden_dim)
        updated = self.norm(nodes + self.output(message))
        return self.feed_forward_norm(updated + self.feed_forward(updated))


def fold_ldd_distance(distance: int, dmax: int = 500) -> int:
    """User-confirmed Appendix F.1 folding, separating finite and infinite cases."""
    if distance == POS_INF:
        return dmax
    if distance == NEG_INF:
        return -dmax
    if distance >= dmax:
        return dmax - 1
    if distance <= -dmax:
        return -(dmax - 1)
    return distance


def ldd_bias_indices(distances: torch.Tensor, dmax: int = 500) -> torch.Tensor:
    folded = torch.empty_like(distances, dtype=torch.long)
    folded[distances == POS_INF] = dmax
    folded[distances == NEG_INF] = -dmax
    finite = (distances != POS_INF) & (distances != NEG_INF)
    finite_values = distances[finite]
    folded[finite] = torch.where(
        finite_values >= dmax,
        torch.full_like(finite_values, dmax - 1),
        torch.where(finite_values <= -dmax, torch.full_like(finite_values, -(dmax - 1)), finite_values),
    )
    return folded + dmax


def ldd_attention_mask(distances: torch.Tensor, heads: int) -> torch.Tensor:
    """Appendix F.1 N1..N8 masks; heads need not be paper-sized in smoke mode."""
    groups = torch.div(torch.arange(heads, device=distances.device) * 8, heads, rounding_mode="floor")
    masks: list[torch.Tensor] = []
    finite = (distances != POS_INF) & (distances != NEG_INF)
    for group in groups.tolist():
        if group == 0:  # N1
            mask = torch.ones_like(distances, dtype=torch.bool)
        elif group == 1:  # N2
            mask = distances == 1
        elif group == 2:  # N3
            mask = distances == -1
        elif group == 3:  # N4
            mask = distances == 2
        elif group == 4:  # N5
            mask = distances == -2
        elif group == 5:  # N6
            mask = finite & (distances >= 3)
        elif group == 6:  # N7
            mask = finite & (distances <= -3)
        else:  # N8
            mask = distances == POS_INF
        masks.append(mask)
    return torch.stack(masks, dim=0)


class WeCAN(nn.Module):
    """Single forward pass creates static task-pool scores and skip parameters."""

    def __init__(self, resource_dims: int, config: WeCANConfig = WeCANConfig()) -> None:
        super().__init__()
        self.resource_dims = resource_dims
        self.config = config
        self.task_embedder = InitialEmbedder(1 + resource_dims, config.high_dim)
        pool_input_dim = resource_dims + 4 if config.comm_budget_enabled else resource_dims
        self.pool_embedder = InitialEmbedder(pool_input_dim, config.high_dim)
        self.pool_network_encoder = PoolNetworkEncoder(config.high_dim) if config.comm_budget_enabled else None
        self.initial_weca = WeightedCrossAttention(config.high_dim, config.weca_heads)
        self.ldd_layers = nn.ModuleList(
            LDDGraphAttention(config.high_dim, config.ldd_heads, config.dmax) for _ in range(config.ldd_layers)
        )
        self.task_projection = nn.Identity() if config.high_dim == config.low_dim else nn.Linear(config.high_dim, config.low_dim)
        self.pool_projection = nn.Identity() if config.high_dim == config.low_dim else nn.Linear(config.high_dim, config.low_dim)
        self.task_updates = nn.ModuleList(
            WeightedCrossAttention(config.low_dim, config.weca_heads) for _ in range(config.alternating_weca_layers)
        )
        self.pool_updates = nn.ModuleList(
            WeightedCrossAttention(config.low_dim, config.weca_heads) for _ in range(config.alternating_weca_layers)
        )
        self.score_task = nn.Linear(config.low_dim, config.low_dim, bias=False)
        self.score_pool = nn.Linear(config.low_dim, config.low_dim, bias=False)
        self.skip_head = nn.Sequential(
            nn.Linear(config.low_dim, config.skip_hidden_dim),
            nn.GELU(),
            nn.Linear(config.skip_hidden_dim, config.skip_hidden_dim),
            nn.GELU(),
            nn.Linear(config.skip_hidden_dim, 3),
        )
        self.forward_calls = 0

    def reset_forward_counter(self) -> None:
        self.forward_calls = 0

    def forward(self, instance: DAGInstance) -> WeCANOutput:
        self.forward_calls += 1
        if instance.comm_budget_enabled != self.config.comm_budget_enabled:
            raise ValueError(
                "CommBudget mode must be enabled consistently in both WeCANConfig and DAGInstance."
            )
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        task_scalar = instance.task_workloads if instance.comm_budget_enabled else instance.task_durations
        assert task_scalar is not None
        task_features = torch.tensor(
            [[task_scalar[task], *instance.task_demands[task]] for task in range(instance.num_tasks)],
            dtype=dtype,
            device=device,
        )
        pool_features = torch.tensor(
            [instance.pool_features(pool) for pool in range(instance.num_pools)], dtype=dtype, device=device
        )
        compatibility = torch.tensor(instance.compatibility, dtype=dtype, device=device)
        task_embeddings = self.task_embedder(task_features)
        pool_embeddings = self.pool_embedder(pool_features)
        if self.pool_network_encoder is not None:
            assert instance.bandwidth is not None
            bandwidth = torch.tensor(instance.bandwidth, dtype=dtype, device=device)
            pool_embeddings = self.pool_network_encoder(pool_embeddings, bandwidth)
        task_embeddings = self.initial_weca(task_embeddings, pool_embeddings, compatibility)
        distances = ldd_distances(instance, device)
        for layer in self.ldd_layers:
            task_embeddings = layer(task_embeddings, distances)
        task_embeddings = self.task_projection(task_embeddings)
        pool_embeddings = self.pool_projection(pool_embeddings)
        for task_update, pool_update in zip(self.task_updates, self.pool_updates):
            task_embeddings = task_update(task_embeddings, pool_embeddings, compatibility)
            pool_embeddings = pool_update(pool_embeddings, task_embeddings, compatibility.transpose(0, 1))
        scores = torch.matmul(self.score_task(task_embeddings), self.score_pool(pool_embeddings).transpose(0, 1))
        scores = scores + torch.where(compatibility > 0, torch.log(compatibility), torch.full_like(compatibility, -1e9))
        raw_skip = self.skip_head(task_embeddings.mean(dim=0))
        if self.config.skip_parameterization == "paper_sigmoid":
            skip_parameters = torch.sigmoid(raw_skip)
        else:
            skip_parameters = torch.nn.functional.softplus(raw_skip) + self.config.minimum_skip_parameter
        return WeCANOutput(task_pool_scores=scores, skip_parameters=skip_parameters)


def ldd_distances(instance: DAGInstance, device: torch.device | None = None) -> torch.Tensor:
    """Signed longest directed distances; +∞ incomparable, -∞ disconnected."""
    task_count = instance.num_tasks
    negative = -10**8
    longest = [[negative] * task_count for _ in range(task_count)]
    for task in range(task_count):
        longest[task][task] = 0
    for source in reversed(instance.topological_order()):
        for child in instance.children[source]:
            longest[source][child] = max(longest[source][child], 1)
            for destination in range(task_count):
                if longest[child][destination] > negative:
                    longest[source][destination] = max(longest[source][destination], 1 + longest[child][destination])
    adjacency = [set() for _ in range(task_count)]
    for source, destination in instance.edges:
        adjacency[source].add(destination)
        adjacency[destination].add(source)
    connected = [[False] * task_count for _ in range(task_count)]
    for source in range(task_count):
        stack = [source]
        connected[source][source] = True
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if not connected[source][neighbour]:
                    connected[source][neighbour] = True
                    stack.append(neighbour)
    values = torch.empty((task_count, task_count), dtype=torch.long, device=device)
    for source in range(task_count):
        for destination in range(task_count):
            if source == destination:
                values[source, destination] = 0
            elif longest[source][destination] > 0:
                values[source, destination] = longest[source][destination]
            elif longest[destination][source] > 0:
                values[source, destination] = -longest[destination][source]
            elif connected[source][destination]:
                values[source, destination] = POS_INF
            else:
                values[source, destination] = NEG_INF
    return values
