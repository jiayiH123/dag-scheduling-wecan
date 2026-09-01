"""Gated Phase 1.5 diagnostics and metric semantics; no long training is implicit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json
import math

import torch

from baselines.algorithms import ca_heft_schedule, greedy_schedule
from data.instance import DAGInstance, GeneratorConfig, generate_dataset
from models.wecan import WeCAN, WeCANConfig
from oracle.milp_oracle import solve_milp_oracle
from oracle.exhaustive_oracle import solve_exhaustive_without_active_wait
from scheduler.generator import DecodeTrace, SkipExtendedGenerator
from scheduler.validator import validate_schedule
from training.reinforce import ReinforceTrainer, TrainConfig, set_seed


A1A_DESCRIPTION = "policy contraction and training-correctness validation"
A1A_ORACLE_CLOSURE_THRESHOLD = 1.0
A1A_SAMPLE_MEAN_REDUCTION_THRESHOLD = 0.05
A1A_ZERO_TOLERANCE = 1e-6


def build_integer_diagnostic_instance(stage: str, seed: int) -> DAGInstance:
    """Recreate the original A1/A2 integer-tick diagnostic instance deterministically."""
    if stage not in {"A1", "A2"}:
        raise ValueError("stage must be A1 or A2")
    tasks = 10 if stage == "A1" else 20
    raw = generate_dataset(
        GeneratorConfig(
            num_tasks_min=tasks,
            num_tasks_max=tasks,
            num_pools=3,
            resource_dims=2,
            max_width=4,
            edge_probability=0.35,
            duration_low=1,
            duration_high=5,
            demand_fraction_low=0.15,
            demand_fraction_high=0.5,
            capacity_low=6,
            capacity_high=12,
            incompatibility_probability=0.15,
            compatibility_low=1,
            compatibility_high=1,
        ),
        1,
        seed,
        stage.lower(),
    )[0]
    instance = DAGInstance(
        name=raw.name,
        task_durations=tuple(float(max(1, round(value))) for value in raw.task_durations),
        task_demands=tuple(
            tuple(float(max(1, round(value))) for value in row) for row in raw.task_demands
        ),
        pool_capacities=tuple(
            tuple(float(max(1, round(value))) for value in row) for row in raw.pool_capacities
        ),
        compatibility=raw.compatibility,
        edges=raw.edges,
    )
    instance.validate()
    return instance


def _ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= A1A_ZERO_TOLERANCE:
        return None
    return numerator / denominator


def a1_metrics(
    *,
    oracle_makespan: float | None,
    external_greedy_makespan: float,
    ca_heft_makespan: float,
    initial_policy_greedy_makespan: float,
    final_policy_greedy_makespan: float,
    initial_sample_mean_makespan: float,
    final_sample_mean_makespan: float,
) -> dict[str, float | None]:
    """Keep external heuristic and policy-decoding metrics explicitly disjoint."""
    policy_delta = initial_policy_greedy_makespan - final_policy_greedy_makespan
    oracle_gap_closure = None
    if oracle_makespan is not None:
        oracle_gap_closure = _ratio(policy_delta, initial_policy_greedy_makespan - oracle_makespan)
    return {
        "oracle_makespan": oracle_makespan,
        "external_greedy_makespan": external_greedy_makespan,
        "ca_heft_makespan": ca_heft_makespan,
        "initial_policy_greedy_makespan": initial_policy_greedy_makespan,
        "final_policy_greedy_makespan": final_policy_greedy_makespan,
        "initial_sample_mean_makespan": initial_sample_mean_makespan,
        "final_sample_mean_makespan": final_sample_mean_makespan,
        "policy_improvement_ratio": _ratio(policy_delta, initial_policy_greedy_makespan),
        "oracle_gap_closure_ratio": oracle_gap_closure,
        "improvement_over_external_greedy": _ratio(
            external_greedy_makespan - final_policy_greedy_makespan,
            external_greedy_makespan,
        ),
    }


def _all_finite(history: Sequence[dict[str, Any]]) -> bool:
    for row in history:
        for key, value in row.items():
            if key in {"baseline_mode"} or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return False
    return True


def _legacy_a1a_passes(
    metrics: dict[str, float | None],
    history: Sequence[dict[str, Any]],
    replay: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    final_oracle = metrics["oracle_makespan"]
    final_best_of_k = replay["final_best_of_k_makespan"]
    terminal = history[-1]
    checks = {
        "replay_feasible_rate_is_100_percent": replay["replay_feasible_rate"] == 1.0,
        "sampled_mean_makespan_materially_decreased": (
            metrics["initial_sample_mean_makespan"] - metrics["final_sample_mean_makespan"]
            >= A1A_SAMPLE_MEAN_REDUCTION_THRESHOLD * metrics["initial_sample_mean_makespan"]
        ),
        "final_best_of_k_reaches_oracle": (
            final_oracle is not None and abs(final_best_of_k - final_oracle) <= A1A_ZERO_TOLERANCE
        ),
        "final_policy_greedy_reaches_oracle": (
            final_oracle is not None
            and abs(metrics["final_policy_greedy_makespan"] - final_oracle) <= A1A_ZERO_TOLERANCE
        ),
        "oracle_gap_closure_reaches_threshold": (
            metrics["oracle_gap_closure_ratio"] is not None
            and metrics["oracle_gap_closure_ratio"] >= A1A_ORACLE_CLOSURE_THRESHOLD - A1A_ZERO_TOLERANCE
        ),
        "terminal_loo_advantage_and_gradient_are_zero": (
            abs(float(terminal["advantage_std"])) <= A1A_ZERO_TOLERANCE
            and abs(float(terminal["gradient_norm"])) <= A1A_ZERO_TOLERANCE
        ),
        "all_logged_metrics_are_finite": _all_finite(history),
        "replay_traces_are_valid_without_time_regression": replay["all_trace_valid"],
    }
    return all(checks.values()), checks


def reanalyse_legacy_a1a(
    legacy_directory: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    """Read the historic A1 run and emit a separate A1-a report without training.

    The historic history labels its post-first-update greedy decode as update 0. That
    preserved observation is deliberately reported as the legacy initial policy greedy
    value; no external heuristic is substituted for it.
    """
    source = Path(legacy_directory)
    destination = Path(output_directory)
    if source.resolve() == destination.resolve():
        raise ValueError("A1-a output must be separate from the historic A1 directory.")
    gate = json.loads((source / "gate.json").read_text(encoding="utf-8"))
    history = json.loads((source / "checkpoint" / "history.json").read_text(encoding="utf-8"))
    if not history:
        raise ValueError("Historic A1 history is empty.")
    instance = build_integer_diagnostic_instance("A1", seed)
    oracle = solve_milp_oracle(instance)
    if oracle.status != "optimal" or oracle.makespan is None or oracle.schedule is None:
        raise RuntimeError("Historic A1 reanalysis requires a proven MILP optimum.")
    if not validate_schedule(instance, oracle.schedule).feasible:
        raise RuntimeError("Recomputed A1 Oracle schedule did not pass the independent Validator.")

    checkpoint = torch.load(source / "checkpoint" / "last.pt", map_location="cpu", weights_only=False)
    model = WeCAN(instance.resource_dims, WeCANConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    generator = SkipExtendedGenerator()
    with torch.no_grad():
        output = model(instance)
        greedy_trace = generator.decode(
            instance, output.task_pool_scores, output.skip_parameters, mode="greedy"
        )
        replayed_samples = []
        # Training's final saved checkpoint follows update 299 (zero-based), while its
        # checkpoint metadata stores the completed count (300).
        final_sample_update = int(checkpoint["update"]) - 1
        for trajectory in range(8):
            random_generator = torch.Generator(device="cpu")
            random_generator.manual_seed(seed + 1_000_003 * final_sample_update + trajectory)
            replayed_samples.append(
                generator.decode(
                    instance,
                    output.task_pool_scores,
                    output.skip_parameters,
                    mode="sample",
                    generator=random_generator,
                    track_log_probability=True,
                )
            )
    validation_rows = [
        validate_schedule(instance, trace.schedule, trace=trace.decisions).to_dict()
        for trace in [greedy_trace, *replayed_samples]
    ]
    sample_makespans = [trace.schedule.makespan for trace in replayed_samples]
    replay = {
        "final_checkpoint_path": str(source / "checkpoint" / "last.pt"),
        "final_checkpoint_update_count": int(checkpoint["update"]),
        "final_checkpoint_sample_seed_update": final_sample_update,
        "replay_policy_greedy_makespan": greedy_trace.schedule.makespan,
        "replay_sample_makespans": sample_makespans,
        "final_best_of_k_makespan": min(sample_makespans),
        "replay_feasible_rate": sum(row["feasible"] for row in validation_rows) / len(validation_rows),
        "all_trace_valid": all(row["feasible"] and row["trace_checked"] for row in validation_rows),
        "validator_results": validation_rows,
    }
    metrics = a1_metrics(
        oracle_makespan=oracle.makespan,
        external_greedy_makespan=greedy_schedule(instance).makespan,
        ca_heft_makespan=ca_heft_schedule(instance).makespan,
        initial_policy_greedy_makespan=float(history[0]["validation_greedy_makespan"]),
        final_policy_greedy_makespan=float(history[-1]["validation_greedy_makespan"]),
        initial_sample_mean_makespan=float(history[0]["mean_makespan"]),
        final_sample_mean_makespan=float(history[-1]["mean_makespan"]),
    )
    passed, checks = _legacy_a1a_passes(metrics, history, replay)
    result = {
        "stage": "A1-a",
        "description": A1A_DESCRIPTION,
        "passed": passed,
        "source": {
            "legacy_directory": str(source),
            "legacy_gate_path": str(source / "gate.json"),
            "legacy_history_path": str(source / "checkpoint" / "history.json"),
            "legacy_gate_passed": gate["passed"],
            "instance_seed": seed,
            "instance": instance.to_dict(),
        },
        "metrics": metrics,
        "criteria": {
            "oracle_gap_closure_threshold": A1A_ORACLE_CLOSURE_THRESHOLD,
            "sample_mean_relative_reduction_threshold": A1A_SAMPLE_MEAN_REDUCTION_THRESHOLD,
            "zero_tolerance": A1A_ZERO_TOLERANCE,
            "checks": checks,
        },
        "terminal_history": {
            "update": history[-1]["update"],
            "advantage_std": history[-1]["advantage_std"],
            "gradient_norm": history[-1]["gradient_norm"],
            "entropy": history[-1]["entropy"],
        },
        "replay": replay,
        "oracle": oracle.to_dict(),
        "interpretation": (
            "The legacy gate compared the external Greedy heuristic with the final policy. "
            "A1-a instead evaluates contraction from the recorded policy greedy decode at "
            "update 0 to the final policy decode, keeping external heuristics separate."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _is_active_skip(instance: DAGInstance, trace: DecodeTrace) -> bool:
    """Return whether a decoded trace actively skips while a dispatch is feasible."""
    current_time = 0.0
    unscheduled = set(range(instance.num_tasks))
    completed: set[int] = set()
    running = []
    available = [list(capacity) for capacity in instance.pool_capacities]
    for event in trace.decisions:
        mask = SkipExtendedGenerator._dispatch_mask(instance, unscheduled, completed, available)
        if event == "skip":
            if any(mask):
                return True
            if not running:
                return False
            current_time, newly_completed = SkipExtendedGenerator._advance_time(running)
            completed.update(newly_completed)
            SkipExtendedGenerator._release(instance, running, available, newly_completed)
            continue
        _, task_text, pool_text = event.split(":")
        task, pool = int(task_text), int(pool_text)
        placement = next(item for item in trace.schedule.placements if item.task == task)
        if abs(placement.start - current_time) > A1A_ZERO_TOLERANCE:
            return False
        unscheduled.remove(task)
        running.append(placement)
        for dimension, demand in enumerate(instance.task_demands[task]):
            available[pool][dimension] -= demand
    return False


def _trace_summary(instance: DAGInstance, traces: Sequence[DecodeTrace]) -> dict[str, Any]:
    validations = [validate_schedule(instance, trace.schedule, trace=trace.decisions).to_dict() for trace in traces]
    active_waits = [trace.active_wait_count for trace in traces]
    passive_time_advances = [trace.passive_time_advance_count for trace in traces]
    makespans = [trace.schedule.makespan for trace in traces]
    best_index = min(range(len(traces)), key=lambda index: makespans[index])
    return {
        "makespans": makespans,
        "sample_mean_makespan": sum(makespans) / len(makespans),
        "best_of_k_makespan": makespans[best_index],
        "best_trace_index": best_index,
        "best_trace_active_wait_count": active_waits[best_index],
        "best_trace_passive_time_advance_count": passive_time_advances[best_index],
        "best_trace_contains_active_wait": active_waits[best_index] > 0,
        "best_trace": {
            "decisions": list(traces[best_index].decisions),
            "placements": [item.__dict__ for item in traces[best_index].schedule.ordered()],
        },
        "any_trace_contains_active_wait": any(active_waits),
        "active_wait_count": sum(active_waits),
        "passive_time_advance_count": sum(passive_time_advances),
        "total_time_advance_count": sum(active_waits) + sum(passive_time_advances),
        "active_wait_ratio": sum(active_waits) / sum(trace.action_count for trace in traces),
        "passive_time_advance_ratio": sum(passive_time_advances) / sum(trace.action_count for trace in traces),
        "feasible_rate": sum(result["feasible"] for result in validations) / len(validations),
        "all_trace_valid": all(result["feasible"] and result["trace_checked"] for result in validations),
        "validator_results": validations,
    }


def _a1c_passes(metrics: dict[str, float | None], history: Sequence[dict[str, Any]], final_samples: dict[str, Any], no_wait: Any) -> tuple[bool, dict[str, bool]]:
    terminal = history[-1]
    checks = {
        "feasible_rate_is_100_percent": final_samples["feasible_rate"] == 1.0,
        "all_metrics_are_finite": _all_finite(history),
        "final_sampled_mean_materially_decreased": (
            metrics["initial_sample_mean_makespan"] - metrics["final_sample_mean_makespan"]
            >= A1A_SAMPLE_MEAN_REDUCTION_THRESHOLD * metrics["initial_sample_mean_makespan"]
        ),
        "final_best_of_k_reaches_oracle": metrics["oracle_makespan"] is not None and abs(final_samples["best_of_k_makespan"] - metrics["oracle_makespan"]) <= A1A_ZERO_TOLERANCE,
        "final_policy_greedy_is_at_most_14": metrics["final_policy_greedy_makespan"] <= 14.0 + A1A_ZERO_TOLERANCE,
        "best_final_trajectory_contains_active_wait": final_samples["best_trace_contains_active_wait"],
        "without_wait_proof_cannot_reach_oracle": (
            no_wait.status == "optimal"
            and no_wait.makespan is not None
            and metrics["oracle_makespan"] is not None
            and no_wait.makespan > metrics["oracle_makespan"] + A1A_ZERO_TOLERANCE
        ),
        "terminal_advantage_and_gradient_are_finite": (
            math.isfinite(float(terminal["advantage_std"])) and math.isfinite(float(terminal["gradient_norm"]))
        ),
    }
    return all(checks.values()), checks


def run_frozen_instance_gate(
    instance: DAGInstance,
    model_config: WeCANConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Train one explicitly approved frozen diagnostic instance within its configured budget."""
    if stage not in {"A1-c", "A1-b-exact"}:
        raise ValueError("frozen diagnostics support A1-c or A1-b-exact only")
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty diagnostic output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    oracle = solve_milp_oracle(instance)
    if oracle.status != "optimal" or oracle.makespan is None:
        raise RuntimeError(f"{stage} requires a proven MILP optimum.")
    external_greedy = greedy_schedule(instance)
    ca_heft = ca_heft_schedule(instance)
    external_validation = {
        "external_greedy": validate_schedule(instance, external_greedy).to_dict(),
        "ca_heft": validate_schedule(instance, ca_heft).to_dict(),
    }
    if not all(result["feasible"] for result in external_validation.values()):
        raise RuntimeError("A1-c external baseline failed independent validation.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_config.seed)
    model = WeCAN(instance.resource_dims, model_config)
    trainer = ReinforceTrainer(model, train_config, device)
    initial_policy_greedy = trainer._mean_greedy_makespan([instance])
    initial_output = trainer.model(instance)
    initial_traces = []
    for trajectory in range(train_config.rollouts_per_instance):
        random_generator = torch.Generator(device=device)
        random_generator.manual_seed(train_config.seed + trajectory)
        initial_traces.append(trainer.generator.decode(
            instance, initial_output.task_pool_scores, initial_output.skip_parameters,
            mode="sample", generator=random_generator, track_log_probability=True,
        ))
    initial_samples = _trace_summary(instance, initial_traces)
    trainer.train([instance], [instance])
    final_policy_greedy = trainer._mean_greedy_makespan([instance])
    trainer.model.eval()
    with torch.no_grad():
        final_output = trainer.model(instance)
        final_traces = []
        final_update = trainer.update_count - 1
        for trajectory in range(train_config.rollouts_per_instance):
            random_generator = torch.Generator(device=device)
            random_generator.manual_seed(train_config.seed + 1_000_003 * final_update + trajectory)
            final_traces.append(trainer.generator.decode(
                instance, final_output.task_pool_scores, final_output.skip_parameters,
                mode="sample", generator=random_generator, track_log_probability=True,
            ))
    final_samples = _trace_summary(instance, final_traces)
    metrics = a1_metrics(
        oracle_makespan=oracle.makespan,
        external_greedy_makespan=external_greedy.makespan,
        ca_heft_makespan=ca_heft.makespan,
        initial_policy_greedy_makespan=initial_policy_greedy,
        final_policy_greedy_makespan=final_policy_greedy,
        initial_sample_mean_makespan=float(initial_samples["sample_mean_makespan"]),
        final_sample_mean_makespan=float(final_samples["sample_mean_makespan"]),
    )
    no_active_wait = solve_exhaustive_without_active_wait(instance)
    if stage == "A1-c":
        passed, checks = _a1c_passes(metrics, trainer.history, final_samples, no_active_wait)
        status = "passed" if passed else "failed_a1c_criteria"
    else:
        terminal = trainer.history[-1]
        checks = {
            "feasible_rate_is_100_percent": final_samples["feasible_rate"] == 1.0,
            "all_metrics_are_finite": _all_finite(trainer.history),
            "final_sampled_mean_materially_decreased": (
                metrics["initial_sample_mean_makespan"] - metrics["final_sample_mean_makespan"]
                >= A1A_SAMPLE_MEAN_REDUCTION_THRESHOLD * metrics["initial_sample_mean_makespan"]
            ),
            "final_best_of_k_reaches_oracle": abs(final_samples["best_of_k_makespan"] - oracle.makespan) <= A1A_ZERO_TOLERANCE,
            "final_policy_greedy_reaches_oracle": abs(final_policy_greedy - oracle.makespan) <= A1A_ZERO_TOLERANCE,
            "best_final_trajectory_has_no_active_wait": final_samples["best_trace_active_wait_count"] == 0,
            "strictly_improves_over_external_greedy": final_policy_greedy < external_greedy.makespan - A1A_ZERO_TOLERANCE,
            "terminal_advantage_and_gradient_are_finite": (
                math.isfinite(float(terminal["advantage_std"])) and math.isfinite(float(terminal["gradient_norm"]))
            ),
        }
        passed = all(checks.values())
        status = (
            "passed" if passed else
            "optimal trajectory discovered but deterministic decode not converged"
            if checks["final_best_of_k_reaches_oracle"] and not checks["final_policy_greedy_reaches_oracle"]
            else "failed_a1b_exact_criteria"
        )
    result = {
        "stage": stage,
        "status": status,
        "passed": passed,
        "frozen_instance": instance.to_dict(),
        "metrics": metrics,
        "external_baselines": {
            "external_greedy_schedule": {
                "makespan": external_greedy.makespan,
                "placements": [item.__dict__ for item in external_greedy.ordered()],
            },
            "ca_heft_schedule": {
                "makespan": ca_heft.makespan,
                "placements": [item.__dict__ for item in ca_heft.ordered()],
            },
            "validator": external_validation,
        },
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "terminal_training_metrics": {
            key: trainer.history[-1][key]
            for key in (
                "entropy", "mean_advantage", "advantage_std", "gradient_norm", "skip_ratio",
                "active_wait_ratio", "passive_time_advance_ratio",
            )
        },
        "no_active_wait_counterfactual": no_active_wait.to_dict(),
        "criteria": {"checks": checks, "sample_mean_relative_reduction_threshold": A1A_SAMPLE_MEAN_REDUCTION_THRESHOLD},
        "oracle": oracle.to_dict(),
        "history": trainer.history,
    }
    (directory / "gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_a1c_frozen_instance_gate(
    instance: DAGInstance,
    model_config: WeCANConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compatibility wrapper for the completed A1-c frozen diagnostic."""
    return run_frozen_instance_gate(instance, model_config, train_config, output_dir, stage="A1-c")


def run_single_instance_gate(
    instance: DAGInstance,
    model_config: WeCANConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Run an explicitly requested future A1/A2 gate with unambiguous metric fields."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    oracle = solve_milp_oracle(instance)
    if stage == "A1" and oracle.status != "optimal":
        result = {
            "stage": stage,
            "passed": False,
            "blocked": "A1 requires proven MILP optimum",
            "oracle": oracle.to_dict(),
        }
        (directory / "gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    external_greedy = greedy_schedule(instance).makespan
    ca_heft = ca_heft_schedule(instance).makespan
    model = WeCAN(instance.resource_dims, model_config)
    trainer = ReinforceTrainer(model, train_config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    initial_policy_greedy = trainer._mean_greedy_makespan([instance])
    trainer.train([instance], [instance])
    final_policy_greedy = trainer._mean_greedy_makespan([instance])
    metrics = a1_metrics(
        oracle_makespan=oracle.makespan if oracle.status == "optimal" else None,
        external_greedy_makespan=external_greedy,
        ca_heft_makespan=ca_heft,
        initial_policy_greedy_makespan=initial_policy_greedy,
        final_policy_greedy_makespan=final_policy_greedy,
        initial_sample_mean_makespan=float(trainer.history[0]["mean_makespan"]),
        final_sample_mean_makespan=float(trainer.history[-1]["mean_makespan"]),
    )
    result = {
        "stage": stage,
        "passed": (
            metrics["policy_improvement_ratio"] is not None
            and metrics["policy_improvement_ratio"] >= 0.05
        ),
        "metrics": metrics,
        "oracle": oracle.to_dict(),
        "history": trainer.history,
        "deprecated_fields": {
            "initial_greedy_makespan": "Use external_greedy_makespan or initial_policy_greedy_makespan.",
            "final_greedy_makespan": "Use final_policy_greedy_makespan.",
        },
    }
    (directory / "gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
