"""REINFORCE trainer with explicit multi-trajectory baseline variants and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence
import copy
import json
import random

import numpy as np
import torch

from data.instance import DAGInstance
from models.wecan import WeCAN
from scheduler.generator import DecodeTrace, SkipExtendedGenerator

BaselineMode = Literal[
    "batch_global_mean",
    "instance_mean",
    "instance_leave_one_out",
    "current_policy_greedy",
    "snapshot_greedy_rollout",
]
OptimizerName = Literal["adam"]


@dataclass(frozen=True)
class TrainConfig:
    optimizer: OptimizerName = "adam"
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 20
    max_updates: int | None = None
    rollouts_per_instance: int = 8
    baseline_mode: BaselineMode = "instance_leave_one_out"
    snapshot_sync_interval_updates: int = 50
    seed: int = 2026
    checkpoint_dir: str = "checkpoints/wecan_phase15"

    def __post_init__(self) -> None:
        if self.optimizer != "adam":
            raise ValueError("Only the paper's Adam optimizer is supported.")
        if self.rollouts_per_instance not in {1, 4, 8, 16}:
            raise ValueError("rollouts_per_instance must be 1, 4, 8, or 16.")
        if self.baseline_mode in {"instance_mean", "instance_leave_one_out"} and self.rollouts_per_instance < 2:
            raise ValueError("instance_mean and leave-one-out require at least two trajectories.")
        if self.snapshot_sync_interval_updates < 1:
            raise ValueError("snapshot_sync_interval_updates must be positive.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_advantages(makespans: torch.Tensor, mode: BaselineMode, greedy_makespans: torch.Tensor | None = None) -> torch.Tensor:
    """Return detached-baseline makespan advantages shaped [instances, K]."""
    if makespans.ndim != 2:
        raise ValueError("makespans must have shape [instances, trajectories].")
    if mode == "batch_global_mean":
        baseline = makespans.mean().detach()
    elif mode == "instance_mean":
        if makespans.shape[1] < 2:
            raise ValueError("instance_mean needs at least two trajectories.")
        baseline = makespans.mean(dim=1, keepdim=True).detach()
    elif mode == "instance_leave_one_out":
        if makespans.shape[1] < 2:
            raise ValueError("leave-one-out needs at least two trajectories.")
        baseline = ((makespans.sum(dim=1, keepdim=True) - makespans) / (makespans.shape[1] - 1)).detach()
    elif mode in {"current_policy_greedy", "snapshot_greedy_rollout"}:
        if greedy_makespans is None or greedy_makespans.shape != (makespans.shape[0],):
            raise ValueError("Greedy baselines need one detached makespan per instance.")
        baseline = greedy_makespans.detach().unsqueeze(1)
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")
    return makespans - baseline


class ReinforceTrainer:
    """One policy forward per instance; K generator trajectories reuse its static outputs."""

    def __init__(self, model: WeCAN, config: TrainConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.generator = SkipExtendedGenerator()
        self.history: list[dict[str, float | int | str | bool]] = []
        self.update_count = 0
        self.baseline_model: WeCAN | None = None
        self.last_snapshot_sync_update: int | None = None
        if config.baseline_mode == "snapshot_greedy_rollout":
            self.baseline_model = copy.deepcopy(self.model).to(device).eval()
            for parameter in self.baseline_model.parameters():
                parameter.requires_grad_(False)
            self.last_snapshot_sync_update = 0

    def train(
        self,
        train_instances: Sequence[DAGInstance],
        validation_instances: Sequence[DAGInstance],
        start_epoch: int = 0,
    ) -> list[dict[str, float | int | str | bool]]:
        if not train_instances:
            raise ValueError("Training data cannot be empty.")
        set_seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_validation = float("inf")
        stop = False
        for epoch in range(start_epoch + 1, start_epoch + self.config.epochs + 1):
            for offset in range(0, len(train_instances), self.config.batch_size):
                indices = rng.permutation(len(train_instances)) if offset == 0 else indices
                batch = [train_instances[index] for index in indices[offset : offset + self.config.batch_size]]
                metrics = self._train_update(batch)
                metrics["epoch"] = epoch
                self.history.append(metrics)
                self.update_count += 1
                if self.config.max_updates is not None and self.update_count >= self.config.max_updates:
                    stop = True
                    break
            validation_makespan = self._mean_greedy_makespan(validation_instances)
            self.history[-1]["validation_greedy_makespan"] = validation_makespan
            self._save_checkpoint(checkpoint_dir / "last.pt", epoch)
            if validation_makespan < best_validation:
                best_validation = validation_makespan
                self._save_checkpoint(checkpoint_dir / "best.pt", epoch)
            (checkpoint_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            if stop:
                break
        return self.history

    def _train_update(self, batch: Sequence[DAGInstance]) -> dict[str, float | int | str | bool]:
        self.model.train()
        per_instance_traces: list[list[DecodeTrace]] = []
        rollout_times: list[float] = []
        forward_times: list[float] = []
        current_outputs = []
        for instance_index, instance in enumerate(batch):
            forward_start = perf_counter()
            output = self.model(instance)
            forward_times.append(perf_counter() - forward_start)
            current_outputs.append(output)
            traces = []
            rollout_start = perf_counter()
            for trajectory in range(self.config.rollouts_per_instance):
                random_generator = torch.Generator(device=self.device)
                random_generator.manual_seed(self.config.seed + 1_000_003 * self.update_count + 1_009 * instance_index + trajectory)
                traces.append(self.generator.decode(
                    instance, output.task_pool_scores, output.skip_parameters, mode="sample",
                    generator=random_generator, track_log_probability=True,
                ))
            rollout_times.append(perf_counter() - rollout_start)
            per_instance_traces.append(traces)
        makespans = torch.tensor(
            [[trace.schedule.makespan for trace in traces] for traces in per_instance_traces],
            dtype=torch.float32,
            device=self.device,
        )
        greedy_makespans = self._baseline_makespans(batch, current_outputs)
        advantages = compute_advantages(makespans, self.config.baseline_mode, greedy_makespans)
        log_probabilities = torch.stack([
            trace.log_probability for traces in per_instance_traces for trace in traces if trace.log_probability is not None
        ]).reshape_as(makespans)
        entropy_sums = torch.stack([
            trace.entropy_sum for traces in per_instance_traces for trace in traces if trace.entropy_sum is not None
        ]).reshape_as(makespans)
        action_counts = torch.tensor([[trace.action_count for trace in traces] for traces in per_instance_traces], device=self.device)
        policy_loss = (advantages * log_probabilities).mean()
        self.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf")).detach().cpu())
        self.optimizer.step()
        skips = sum(trace.skip_count for traces in per_instance_traces for trace in traces)
        active_waits = sum(trace.active_wait_count for traces in per_instance_traces for trace in traces)
        passive_time_advances = sum(trace.passive_time_advance_count for traces in per_instance_traces for trace in traces)
        actions = sum(trace.action_count for traces in per_instance_traces for trace in traces)
        flat_rollout_times = np.repeat(np.array(rollout_times) / self.config.rollouts_per_instance, self.config.rollouts_per_instance)
        return {
            "update": self.update_count,
            "baseline_mode": self.config.baseline_mode,
            "rollouts_per_instance": self.config.rollouts_per_instance,
            "policy_loss": float(policy_loss.detach().cpu()),
            "mean_makespan": float(makespans.mean().detach().cpu()),
            "median_makespan": float(makespans.median().detach().cpu()),
            "makespan_std": float(makespans.std(unbiased=False).detach().cpu()),
            "mean_advantage": float(advantages.mean().detach().cpu()),
            "advantage_std": float(advantages.std(unbiased=False).detach().cpu()),
            "entropy": float((entropy_sums / action_counts).mean().detach().cpu()),
            "gradient_norm": gradient_norm,
            # Kept for existing consumers: total skips include active waits and
            # unavoidable passive time advances. Use the explicit split below.
            "skip_ratio": float(skips / actions) if actions else 0.0,
            "active_wait_ratio": float(active_waits / actions) if actions else 0.0,
            "passive_time_advance_ratio": float(passive_time_advances / actions) if actions else 0.0,
            "forward_seconds_per_instance": float(np.mean(forward_times)),
            "rollout_seconds_total_per_instance": float(np.mean(rollout_times)),
            "rollout_seconds_mean": float(np.mean(flat_rollout_times)),
            "rollout_seconds_median": float(np.median(flat_rollout_times)),
            "rollout_seconds_p50": float(np.percentile(flat_rollout_times, 50)),
            "rollout_seconds_p95": float(np.percentile(flat_rollout_times, 95)),
            "snapshot_sync_interval_updates": self.config.snapshot_sync_interval_updates,
            "snapshot_last_sync_update": -1 if self.last_snapshot_sync_update is None else self.last_snapshot_sync_update,
            "snapshot_extra_forward": self.config.baseline_mode == "snapshot_greedy_rollout",
        }

    @torch.no_grad()
    def _baseline_makespans(self, instances: Sequence[DAGInstance], current_outputs) -> torch.Tensor | None:
        mode = self.config.baseline_mode
        if mode not in {"current_policy_greedy", "snapshot_greedy_rollout"}:
            return None
        values = []
        if mode == "current_policy_greedy":
            for instance, output in zip(instances, current_outputs):
                values.append(self.generator.decode(instance, output.task_pool_scores, output.skip_parameters, mode="greedy").schedule.makespan)
        else:
            assert self.baseline_model is not None
            if self.update_count and self.update_count % self.config.snapshot_sync_interval_updates == 0:
                self.baseline_model.load_state_dict(self.model.state_dict())
                self.last_snapshot_sync_update = self.update_count
            self.baseline_model.eval()
            for instance in instances:
                output = self.baseline_model(instance)
                values.append(self.generator.decode(instance, output.task_pool_scores, output.skip_parameters, mode="greedy").schedule.makespan)
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _mean_greedy_makespan(self, instances: Sequence[DAGInstance]) -> float:
        self.model.eval()
        values = []
        for instance in instances:
            output = self.model(instance)
            values.append(self.generator.decode(instance, output.task_pool_scores, output.skip_parameters, mode="greedy").schedule.makespan)
        self.model.train()
        return float(np.mean(values))

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save({
            "epoch": epoch,
            "update": self.update_count,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "baseline_model_state": None if self.baseline_model is None else self.baseline_model.state_dict(),
            "last_snapshot_sync_update": self.last_snapshot_sync_update,
            "model_config": self.model.config.__dict__,
            "resource_dims": self.model.resource_dims,
            "train_config": self.config.__dict__,
        }, path)
