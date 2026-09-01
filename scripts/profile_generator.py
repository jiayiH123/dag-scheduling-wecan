"""Generator profiling: N=50/100/200 scaling + CPU-vs-GPU A/B.

Usage:
    python scripts/profile_generator.py
"""

import cProfile
import io
import os
import pstats
import time

import numpy as np
import torch

from data.instance import GeneratorConfig, RandomDAGGenerator
from models.wecan import WeCAN, WeCANConfig
from scheduler.generator import SkipExtendedGenerator

import yaml


def make_instance(n: int, seed: int = 0):
    """Random DAG with exactly n tasks; P=3 pools."""
    cfg = GeneratorConfig(
        num_tasks_min=n, num_tasks_max=n,
        num_pools=3,
        edge_probability=0.15,
    )
    gen = RandomDAGGenerator(cfg, seed=seed)
    return gen.generate(name=f"n{n}_s{seed}")


def make_model(device: torch.device) -> WeCAN:
    with open("configs/phase1_paper.yaml") as f:
        raw = yaml.safe_load(f)
    mc = raw["model"]
    cfg = WeCANConfig(
        profile=mc["profile"],
        high_dim=mc["high_dim"], low_dim=mc["low_dim"],
        weca_heads=mc["weca_heads"], ldd_heads=mc["ldd_heads"],
        ldd_layers=mc["ldd_layers"], dmax=mc["dmax"],
        skip_parameterization=mc["skip_parameterization"],
    )
    return WeCAN(resource_dims=2, config=cfg).to(device)


def time_sections(instance, scores, skip_params, device):
    """Run one greedy decode with manual section timers (same semantics as reference)."""
    from scheduler.types import EPS, Schedule, TaskPlacement
    from scheduler.action_bounds import max_decode_actions

    t_mask = t_next = t_score = t_update = 0.0
    action_count = skip_count = 0

    current_time = 0.0
    unscheduled = set(range(instance.num_tasks))
    completed: set[int] = set()
    running = []
    available = [list(cap) for cap in instance.pool_capacities]
    placements_by_task: dict = {}

    gen = SkipExtendedGenerator()

    while unscheduled:
        # ---------- dispatch mask ----------
        t0 = time.perf_counter()
        mask = gen._dispatch_mask(instance, unscheduled, completed, available, placements_by_task, current_time)
        t_mask += time.perf_counter() - t0

        # ---------- next event ----------
        t0 = time.perf_counter()
        next_time, next_reason = gen._next_event(instance, unscheduled, completed, running, placements_by_task, current_time)
        t_next += time.perf_counter() - t0

        # ---------- action scoring / argmax ----------
        t0 = time.perf_counter()
        mask_tensor = torch.tensor(mask, device=device, dtype=torch.bool)
        feasible = bool(mask_tensor.any().item())
        skip_avail = next_time is not None and (feasible or not feasible)  # always check
        skip_avail = (next_time is not None and feasible) or (next_time is not None and not feasible)
        skip_avail = next_time is not None  # simplified: skip is available iff next event exists

        flat_scores = scores.reshape(-1)
        from scheduler.generator import skip_log_score
        skip_score = skip_log_score(skip_params, action_count, instance.num_tasks)
        action_scores = torch.cat((flat_scores, skip_score.reshape(1)))
        full_mask = torch.cat((mask_tensor, torch.tensor([skip_avail], device=device)))
        masked = action_scores.masked_fill(~full_mask, -torch.inf)
        action = int(torch.argmax(masked).item())
        t_score += time.perf_counter() - t0

        # ---------- state update ----------
        t0 = time.perf_counter()
        n_tasks = instance.num_tasks * instance.num_pools
        if action == n_tasks:
            current_time, newly_done = gen._advance_time(running, next_time)
            for t in newly_done:
                completed.add(t)
            gen._release(instance, running, available, newly_done)
            skip_count += 1
        else:
            task = action // instance.num_pools
            pool = action % instance.num_pools
            duration = instance.actual_duration(task, pool)
            from scheduler.types import TaskPlacement
            pl = TaskPlacement(task=task, pool=pool, start=current_time, end=current_time + duration)
            placements_by_task[task] = pl
            running.append(pl)
            unscheduled.remove(task)
            for dim, demand in enumerate(instance.task_demands[task]):
                available[pool][dim] -= demand
        t_update += time.perf_counter() - t0
        action_count += 1
        if action_count > max_decode_actions(instance):
            raise RuntimeError("exceeded action bound")

    return {
        "action_count": action_count,
        "skip_count": skip_count,
        "t_mask_ms": t_mask * 1000,
        "t_next_ms": t_next * 1000,
        "t_score_ms": t_score * 1000,
        "t_update_ms": t_update * 1000,
        "t_total_ms": (t_mask + t_next + t_score + t_update) * 1000,
    }


def main():
    print(f"PID={os.getpid()}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    model = make_model(device)
    gen = SkipExtendedGenerator()

    # ── Section 1: N scaling (50, 100, 200) ─────────────────────────────────
    print("\n" + "="*60)
    print("SECTION 1 — N scaling (GPU scores, greedy decode)")
    print("="*60)
    header = f"{'N':>5} {'edges':>6} {'actions':>8} {'skips':>6} "
    header += f"{'mask_ms':>9} {'next_ms':>9} {'score_ms':>9} {'upd_ms':>8} {'total_ms':>9}"
    print(header)

    scale_results = {}
    for n in [50, 100, 200]:
        inst = make_instance(n, seed=42)
        with torch.no_grad():
            out = model(inst)
        r = time_sections(inst, out.task_pool_scores, out.skip_parameters, device)
        scale_results[n] = r
        row = (f"{n:>5} {len(inst.edges):>6} {r['action_count']:>8} {r['skip_count']:>6} "
               f"{r['t_mask_ms']:>9.1f} {r['t_next_ms']:>9.1f} {r['t_score_ms']:>9.1f} "
               f"{r['t_update_ms']:>8.1f} {r['t_total_ms']:>9.1f}")
        print(row)

    # print scaling ratios
    print("\nScaling ratios (relative to N=50):")
    base = scale_results[50]["t_total_ms"]
    for n in [100, 200]:
        r = scale_results[n]["t_total_ms"]
        actions_ratio = scale_results[n]["action_count"] / scale_results[50]["action_count"]
        print(f"  N={n}: total_ms ratio={r/base:.2f}x  action_count ratio={actions_ratio:.2f}x  "
              f"per-action ratio={(r/scale_results[n]['action_count'])/(base/scale_results[50]['action_count']):.2f}x")

    # ── Section 2: CPU vs GPU A/B on N=100 and N=200 ─────────────────────────
    print("\n" + "="*60)
    print("SECTION 2 — CPU vs GPU tensor A/B (greedy decode)")
    print("="*60)
    print(f"{'N':>5} {'mode':>8} {'total_ms':>10} {'score_ms':>10}")

    for n in [100, 200]:
        inst = make_instance(n, seed=42)
        with torch.no_grad():
            out = model(inst)

        # A: scores stay on GPU (current path)
        rA = time_sections(inst, out.task_pool_scores, out.skip_parameters, device)

        # B: scores copied to CPU once before decode
        scores_cpu = out.task_pool_scores.detach().cpu()
        skip_cpu = out.skip_parameters.detach().cpu()
        rB = time_sections(inst, scores_cpu, skip_cpu, torch.device("cpu"))

        print(f"{n:>5} {'GPU':>8} {rA['t_total_ms']:>10.1f} {rA['t_score_ms']:>10.1f}")
        print(f"{n:>5} {'CPU':>8} {rB['t_total_ms']:>10.1f} {rB['t_score_ms']:>10.1f}")
        speedup = rA["t_total_ms"] / rB["t_total_ms"] if rB["t_total_ms"] > 0 else float("nan")
        score_speedup = rA["t_score_ms"] / rB["t_score_ms"] if rB["t_score_ms"] > 0 else float("nan")
        print(f"  → CPU speedup total={speedup:.2f}x  score_section={score_speedup:.2f}x")

    # ── Section 3: cProfile breakdown on N=100 GPU ───────────────────────────
    print("\n" + "="*60)
    print("SECTION 3 — cProfile breakdown (N=100, GPU scores)")
    print("="*60)
    inst100 = make_instance(100, seed=42)
    with torch.no_grad():
        out100 = model(inst100)

    pr = cProfile.Profile()
    pr.enable()
    gen.decode(inst100, out100.task_pool_scores, out100.skip_parameters, mode="greedy")
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    print(buf.getvalue())

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
