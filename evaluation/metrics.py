"""Comparable evaluation metrics for Phase-1 schedulers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
import torch

from baselines.algorithms import ca_heft_schedule, greedy_schedule, random_schedule, standard_heft_schedule
from data.instance import DAGInstance
from models.wecan import WeCAN
from scheduler.generator import SkipExtendedGenerator
from scheduler.types import Schedule
from scheduler.validator import validate_schedule


@dataclass(frozen=True)
class PerInstanceResult:
    instance: str
    algorithm: str
    makespan: float
    feasible: bool
    forward_seconds: float
    generation_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SummaryResult:
    algorithm: str
    instances: int
    feasible_rate: float
    makespan_mean: float
    makespan_median: float
    makespan_std: float
    forward_seconds_mean: float
    generation_seconds_mean: float
    total_seconds_mean: float
    improvement_vs_greedy_mean: float | None = None
    win_rate_vs_greedy: float | None = None
    improvement_vs_heft_mean: float | None = None


def evaluate_baseline(
    instances: Sequence[DAGInstance], algorithm: str, schedule_fn: Callable[[DAGInstance, int], Schedule], seed: int
) -> list[PerInstanceResult]:
    results = []
    for index, instance in enumerate(instances):
        start = perf_counter()
        schedule = schedule_fn(instance, seed + index)
        duration = perf_counter() - start
        validation = validate_schedule(instance, schedule)
        results.append(PerInstanceResult(instance.name, algorithm, schedule.makespan, validation.feasible, 0.0, duration, duration))
    return results


@torch.no_grad()
def evaluate_wecan(instances: Sequence[DAGInstance], model: WeCAN, mode: str = "greedy", seed: int = 2026) -> list[PerInstanceResult]:
    model.eval()
    scheduler = SkipExtendedGenerator()
    results = []
    for index, instance in enumerate(instances):
        model.reset_forward_counter()
        start = perf_counter()
        output = model(instance)
        forward_seconds = perf_counter() - start
        generator = torch.Generator(device=output.task_pool_scores.device)
        generator.manual_seed(seed + index)
        generation_start = perf_counter()
        trace = scheduler.decode(instance, output.task_pool_scores, output.skip_parameters, mode=mode, generator=generator)
        generation_seconds = perf_counter() - generation_start
        validation = validate_schedule(instance, trace.schedule)
        results.append(
            PerInstanceResult(
                instance.name, f"wecan_{mode}", trace.schedule.makespan, validation.feasible,
                forward_seconds, generation_seconds, forward_seconds + generation_seconds,
            )
        )
        if model.forward_calls != 1:
            raise AssertionError("Inference violated the one-forward-pass requirement.")
    return results


def summarize(
    results: Sequence[PerInstanceResult], greedy: Sequence[PerInstanceResult] | None = None, heft: Sequence[PerInstanceResult] | None = None
) -> SummaryResult:
    if not results:
        raise ValueError("Cannot summarize an empty result sequence.")
    makespans = np.array([result.makespan for result in results])
    forward = np.array([result.forward_seconds for result in results])
    generation = np.array([result.generation_seconds for result in results])
    total = np.array([result.total_seconds for result in results])
    improvements_greedy = None
    win_rate = None
    improvements_heft = None
    if greedy is not None:
        baseline = np.array([result.makespan for result in greedy])
        improvements_greedy = float(np.mean((baseline - makespans) / baseline))
        win_rate = float(np.mean(makespans < baseline))
    if heft is not None:
        baseline = np.array([result.makespan for result in heft])
        improvements_heft = float(np.mean((baseline - makespans) / baseline))
    return SummaryResult(
        algorithm=results[0].algorithm,
        instances=len(results),
        feasible_rate=float(np.mean([result.feasible for result in results])),
        makespan_mean=float(makespans.mean()),
        makespan_median=float(np.median(makespans)),
        makespan_std=float(makespans.std(ddof=0)),
        forward_seconds_mean=float(forward.mean()),
        generation_seconds_mean=float(generation.mean()),
        total_seconds_mean=float(total.mean()),
        improvement_vs_greedy_mean=improvements_greedy,
        win_rate_vs_greedy=win_rate,
        improvement_vs_heft_mean=improvements_heft,
    )


def standard_baselines(
    instances: Sequence[DAGInstance], seed: int
) -> tuple[list[PerInstanceResult], list[PerInstanceResult], list[PerInstanceResult], list[PerInstanceResult]]:
    random = evaluate_baseline(instances, "random", lambda instance, item_seed: random_schedule(instance, item_seed), seed)
    greedy = evaluate_baseline(instances, "greedy", lambda instance, _: greedy_schedule(instance), seed)
    ca_heft = evaluate_baseline(instances, "ca_heft", lambda instance, _: ca_heft_schedule(instance), seed)
    standard_heft = evaluate_baseline(instances, "standard_heft", lambda instance, _: standard_heft_schedule(instance), seed)
    return random, greedy, ca_heft, standard_heft
