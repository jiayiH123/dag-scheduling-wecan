#!/usr/bin/env python3
"""GPU short-run capability check for paper-size WeCAN on A100.

Purpose: verify the paper 512/128 model can train on a single A100 without OOM.
Does NOT claim any training convergence; 10 updates are used only for timing.

Usage:
    CUDA_VISIBLE_DEVICES=<id> python scripts/run_gpu_short.py [--updates 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instance import GeneratorConfig, generate_dataset
from environment.config import load_yaml
from models.wecan import WeCAN, WeCANConfig
from scheduler.generator import SkipExtendedGenerator
from training.reinforce import TrainConfig, ReinforceTrainer, set_seed


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_paper.yaml")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # ── GPU info ──────────────────────────────────────────────────────────────
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(idx)
        free_mb = (torch.cuda.get_device_properties(idx).total_memory
                   - torch.cuda.memory_allocated(idx)) / 1024 ** 2
        print(f"GPU {idx}: {gpu_name}")
        print(f"Free memory before model load: {free_mb:.0f} MB")
    else:
        gpu_name = "cpu"
        free_mb = 0.0
        print("Running on CPU")

    # ── Config ────────────────────────────────────────────────────────────────
    configuration = load_yaml(args.config)
    gen_cfg = GeneratorConfig(**configuration["generator"])

    # ── Generate inline data (small instances for speed check) ───────────────
    set_seed(configuration["seed"])
    batch_size = configuration["training"]["batch_size"]   # 64 per paper
    print(f"\nGenerating {batch_size * 2} instances (batch_size={batch_size})")
    train_instances = generate_dataset(gen_cfg, batch_size * 2, seed=configuration["seed"], prefix="gpu_train")
    val_instances = generate_dataset(gen_cfg, 16, seed=configuration["seed"] + 1, prefix="gpu_val")
    n_tasks_mean = sum(i.num_tasks for i in train_instances) / len(train_instances)
    print(f"Mean tasks/instance: {n_tasks_mean:.1f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = WeCANConfig(**configuration["model"])
    model = WeCAN(train_instances[0].resource_dims, model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # ── Build trainer; override max_updates and epochs ────────────────────────
    training_values = dict(configuration["training"])
    training_values.update(
        seed=configuration["seed"],
        checkpoint_dir="checkpoints/gpu_short_test",
        max_updates=args.updates,
        epochs=args.updates,   # at most 1 epoch of data; stop by max_updates
    )
    training_values["baseline_mode"] = training_values.pop("reinforce_baseline", "instance_mean")
    allowed = {
        "optimizer", "learning_rate", "batch_size", "epochs", "max_updates",
        "rollouts_per_instance", "baseline_mode", "snapshot_sync_interval_updates",
        "seed", "checkpoint_dir",
    }
    train_config = TrainConfig(**{k: v for k, v in training_values.items() if k in allowed})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    trainer = ReinforceTrainer(model, train_config, device)

    # ── Warm-up: 1 update ─────────────────────────────────────────────────────
    print(f"\nRunning {args.updates} update(s) on {device} ...")
    t_start = time.perf_counter()
    history = trainer.train(train_instances, val_instances)
    total_elapsed = time.perf_counter() - t_start

    peak_mb = _peak_memory_mb(device)
    updates_done = len(history)
    per_update_s = total_elapsed / max(1, updates_done)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GPU SHORT RUN REPORT")
    print("=" * 60)
    print(f"Device          : {args.device}  ({gpu_name})")
    print(f"Free mem before : {free_mb:.0f} MB")
    print(f"Peak GPU mem    : {peak_mb:.1f} MB")
    print(f"Model params    : {n_params:,}")
    print(f"Batch size      : {batch_size}")
    print(f"Tasks/instance  : {n_tasks_mean:.1f} (mean)")
    print(f"Rollouts/inst   : {train_config.rollouts_per_instance}")
    print(f"Updates done    : {updates_done}")
    print(f"Total elapsed   : {total_elapsed:.2f} s")
    print(f"Per-update time : {per_update_s:.2f} s")
    print()
    for entry in history:
        nan_loss = float("nan") if entry["policy_loss"] != entry["policy_loss"] else None
        print(
            f"  update {entry['update']:>3d} | "
            f"loss={entry['policy_loss']:+.4f} | "
            f"makespan={entry['mean_makespan']:.1f} | "
            f"grad_norm={entry['gradient_norm']:.1f} | "
            f"skip={entry['skip_ratio']:.3f} | "
            f"fwd={entry['forward_seconds_per_instance']*1000:.1f}ms | "
            f"rollout={entry['rollout_seconds_mean']*1000:.1f}ms"
        )
        if nan_loss is not None:
            print("  *** NaN loss detected ***")

    # Check for NaN/Inf
    has_nan = any(
        entry["policy_loss"] != entry["policy_loss"] or
        (entry["gradient_norm"] != entry["gradient_norm"])
        for entry in history
    )
    has_oom = False  # if we got here, no OOM

    print()
    print(f"NaN/Inf detected: {has_nan}")
    print(f"OOM detected    : {has_oom}")
    print(f"batch=64 runnable: {not has_oom and not has_nan}")
    print("=" * 60)

    if args.output:
        report = {
            "device": args.device,
            "gpu_name": gpu_name,
            "free_mb_before": free_mb,
            "peak_gpu_mb": peak_mb,
            "model_params": n_params,
            "batch_size": batch_size,
            "tasks_per_instance_mean": n_tasks_mean,
            "rollouts_per_instance": train_config.rollouts_per_instance,
            "updates_done": updates_done,
            "total_elapsed_s": total_elapsed,
            "per_update_s": per_update_s,
            "has_nan": has_nan,
            "has_oom": has_oom,
            "history": history,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
