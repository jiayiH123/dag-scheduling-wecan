"""Paper-scale training throughput preflight.

Measures a single optimizer update on 500-task layered problems using the
paper-size WeCAN model, K=8 rollouts, instance_mean baseline.

Phases timed separately:
  forward | per-rollout decode | K=8 total | loss construction | backward | optimizer

Batch sizes tested: 1, 2, 4 (one update each).
batch=64 is estimated from scaling, NOT run.

Run from project root:
  CUDA_VISIBLE_DEVICES=1 python -m scripts.run_preflight_training
"""
import os, sys, math, time
import numpy as np
import torch
import yaml

from data.paper_computation_graph import generate_problem
from models.wecan import WeCAN, WeCANConfig
from scheduler.generator import SkipExtendedGenerator
from scheduler.validator import validate_schedule

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 8
LR = 1e-4


def load_model():
    with open("configs/phase1_paper.yaml") as f:
        raw = yaml.safe_load(f)
    mc = raw["model"]
    cfg = WeCANConfig(
        profile=mc["profile"], high_dim=mc["high_dim"], low_dim=mc["low_dim"],
        weca_heads=mc["weca_heads"], ldd_heads=mc["ldd_heads"],
        ldd_layers=mc["ldd_layers"], dmax=mc["dmax"],
        skip_parameterization=mc["skip_parameterization"],
    )
    return WeCAN(resource_dims=2, config=cfg).to(DEVICE)


def make_problem(seed):
    rng = np.random.default_rng(seed)
    return generate_problem("layered", rng, f"layered500_s{seed}")


def reset_peak():
    torch.cuda.reset_peak_memory_stats(DEVICE)


def peak_alloc_mb():
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


def peak_reserved_mb():
    return torch.cuda.max_memory_reserved(DEVICE) / 1024**2


def has_nan_inf(model):
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                return True
    return False


def sec(t): return f"{t:.3f}s"
def ms(t):  return f"{t*1000:.1f}ms"


# ─────────────────────────────────────────────────────────────────────────────
# Part I: detailed phase breakdown for batch=1
# ─────────────────────────────────────────────────────────────────────────────
def run_phase_breakdown(problem, seed_offset=0):
    print("\n" + "="*70)
    print("PART I: Phase breakdown  (batch=1, K=8, 500-task layered)")
    print("="*70)
    gen = SkipExtendedGenerator()

    model = load_model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"  Tasks: {problem.num_tasks}  Edges: {len(problem.edges)}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    reset_peak()
    total_start = time.perf_counter()

    # Phase 1: neural forward
    t0 = time.perf_counter()
    output = model(problem)
    torch.cuda.synchronize(DEVICE)
    fwd_time = time.perf_counter() - t0
    fwd_mem = peak_alloc_mb()
    print(f"\n  [1] Forward:          {ms(fwd_time)}   peak_alloc={fwd_mem:.1f}MB")

    # Phase 2: K=8 sampled rollouts
    reset_peak()
    traces = []
    rollout_times = []
    for k in range(K):
        g = torch.Generator(device=DEVICE)
        g.manual_seed(1000 + seed_offset + k)
        rt0 = time.perf_counter()
        tr = gen.decode(
            problem, output.task_pool_scores, output.skip_parameters,
            mode="sample", generator=g, track_log_probability=True,
        )
        rollout_times.append(time.perf_counter() - rt0)
        traces.append(tr)

    k8_total = sum(rollout_times)
    k8_mem = peak_alloc_mb()
    print(f"  [2] Rollout k=0:      {ms(rollout_times[0])}")
    print(f"  [2] Rollout k=1:      {ms(rollout_times[1])}")
    print(f"  [2] Rollout k=7:      {ms(rollout_times[-1])}")
    print(f"  [2] K=8 total:        {sec(k8_total)}   peak_alloc={k8_mem:.1f}MB")
    print(f"  [2] K=8 mean/rollout: {ms(k8_total/K)}")

    all_feasible = all(validate_schedule(problem, tr.schedule).feasible for tr in traces)
    makespans = [tr.schedule.makespan for tr in traces]
    print(f"  [2] Makespans: mean={np.mean(makespans):.1f}  std={np.std(makespans):.1f}  all_feasible={all_feasible}")

    # Phase 3: loss construction
    reset_peak()
    t0 = time.perf_counter()
    makespan_t = torch.tensor(makespans, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, K]
    baseline = makespan_t.mean(dim=1, keepdim=True).detach()
    advantages = makespan_t - baseline
    log_probs = torch.stack([tr.log_probability for tr in traces]).unsqueeze(0)  # [1, K]
    loss = (advantages * log_probs).mean()
    loss_time = time.perf_counter() - t0
    loss_mem = peak_alloc_mb()
    print(f"\n  [3] Loss construction: {ms(loss_time)}   loss={loss.item():.4f}")

    # Phase 4: backward
    reset_peak()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(DEVICE)
    bwd_time = time.perf_counter() - t0
    bwd_mem = peak_alloc_mb()
    bwd_res = peak_reserved_mb()
    nan_inf = has_nan_inf(model)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).detach().cpu())
    print(f"  [4] Backward:          {ms(bwd_time)}   peak_alloc={bwd_mem:.1f}MB  peak_reserved={bwd_res:.1f}MB")
    print(f"  [4] grad_norm={grad_norm:.4f}  nan/inf_in_grads={nan_inf}")

    # Phase 5: optimizer step
    t0 = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(DEVICE)
    opt_time = time.perf_counter() - t0
    print(f"  [5] Optimizer step:    {ms(opt_time)}")

    total_time = time.perf_counter() - total_start
    overall_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    overall_res = torch.cuda.max_memory_reserved(DEVICE) / 1024**2
    print(f"\n  TOTAL update time:   {sec(total_time)}")
    print(f"  breakdown: fwd={ms(fwd_time)}  K8_rollouts={sec(k8_total)}  bwd={ms(bwd_time)}  opt={ms(opt_time)}")
    print(f"  peak_alloc(overall): {overall_mem:.1f}MB  peak_reserved: {overall_res:.1f}MB")

    return {
        "fwd": fwd_time, "k8_rollout": k8_total, "rollout_per": k8_total/K,
        "loss": loss_time, "bwd": bwd_time, "opt": opt_time, "total": total_time,
        "peak_alloc_mb": overall_mem, "peak_reserved_mb": overall_res,
        "grad_norm": grad_norm, "nan_inf": nan_inf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part II: batch=1/2/4 scaling
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_scaling(problems_pool):
    print("\n" + "="*70)
    print("PART II: Batch scaling  (K=8, 500-task layered per instance)")
    print("="*70)
    print("  (trainer is a Python for-loop over instances)")

    gen = SkipExtendedGenerator()
    results = {}

    for batch_size in [1, 2, 4]:
        batch = problems_pool[:batch_size]
        model = load_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        reset_peak()
        total_start = time.perf_counter()

        per_instance_traces = []
        fwd_total = 0.0
        rollout_total = 0.0

        for idx, inst in enumerate(batch):
            t0 = time.perf_counter()
            output = model(inst)
            fwd_total += time.perf_counter() - t0

            traces = []
            t0 = time.perf_counter()
            for k in range(K):
                g = torch.Generator(device=DEVICE)
                g.manual_seed(2000 + idx * 100 + k)
                tr = gen.decode(
                    inst, output.task_pool_scores, output.skip_parameters,
                    mode="sample", generator=g, track_log_probability=True,
                )
                traces.append(tr)
            rollout_total += time.perf_counter() - t0
            per_instance_traces.append((output, traces))

        makespans = torch.tensor(
            [[tr.schedule.makespan for tr in traces] for _, traces in per_instance_traces],
            dtype=torch.float32, device=DEVICE,
        )  # [batch, K]
        baseline = makespans.mean(dim=1, keepdim=True).detach()
        advantages = makespans - baseline
        log_probs = torch.stack([
            torch.stack([tr.log_probability for tr in traces])
            for _, traces in per_instance_traces
        ])  # [batch, K]
        loss = (advantages * log_probs).mean()

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(DEVICE)
        bwd_time = time.perf_counter() - t0
        optimizer.step()

        total_time = time.perf_counter() - total_start
        peak_alloc = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        peak_res   = torch.cuda.max_memory_reserved(DEVICE) / 1024**2

        print(f"\n  batch={batch_size}:")
        print(f"    fwd_total={sec(fwd_total)}  rollout_total={sec(rollout_total)}  bwd={ms(bwd_time)}")
        print(f"    total={sec(total_time)}  peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_res:.1f}MB")

        results[batch_size] = {
            "fwd": fwd_total, "rollout": rollout_total, "bwd": bwd_time, "total": total_time,
            "peak_alloc_mb": peak_alloc, "peak_reserved_mb": peak_res,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Part III: estimate batch=64
# ─────────────────────────────────────────────────────────────────────────────
def estimate_batch64(batch_results):
    print("\n" + "="*70)
    print("PART III: Estimated batch=64 (NOT run, extrapolated from 1/2/4)")
    print("="*70)

    times = {b: r["total"] for b, r in batch_results.items()}
    # Fit linear scaling: total ~ a*batch + b
    bs = sorted(times.keys())
    xs = np.array(bs, dtype=float)
    ys = np.array([times[b] for b in bs])

    # simple per-instance cost from batch=4 (most reliable)
    per_inst_4 = times[4] / 4
    per_inst_2 = times[2] / 2
    per_inst_1 = times[1] / 1

    print(f"\n  Per-instance time (total/batch):")
    print(f"    batch=1 → {sec(per_inst_1)}/inst")
    print(f"    batch=2 → {sec(per_inst_2)}/inst")
    print(f"    batch=4 → {sec(per_inst_4)}/inst")

    # Use batch=4 measurement as the more stable estimate
    est_per_inst = per_inst_4
    est_b64_update = est_per_inst * 64  # linear loop
    est_800_total = est_b64_update * 800

    print(f"\n  [ESTIMATE] batch=64 single update: ~{est_b64_update/60:.1f} min  ({est_b64_update:.0f}s)")
    print(f"  [ESTIMATE] 800 updates total:       ~{est_800_total/3600:.1f} h   ({est_800_total:.0f}s)")
    print(f"  (assumes linear scaling, Python for-loop, no data loading overlap)")
    print(f"  (based on {sec(per_inst_4)}/instance from batch=4 measurement)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"device={DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
        free, total = torch.cuda.mem_get_info(DEVICE)
        print(f"VRAM free: {free/1024**3:.1f} GB / {total/1024**3:.1f} GB")

    print("\nGenerating 500-task layered problems ...", flush=True)
    problems = [make_problem(seed) for seed in range(4)]
    print(f"  problems[0]: tasks={problems[0].num_tasks}  edges={len(problems[0].edges)}")

    phase_result = run_phase_breakdown(problems[0])
    batch_results = run_batch_scaling(problems)
    estimate_batch64(batch_results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Forward (paper model, 500 tasks): {ms(phase_result['fwd'])}")
    print(f"  Single sampled rollout (500 tasks): {ms(phase_result['rollout_per'])}")
    print(f"  K=8 rollouts total: {sec(phase_result['k8_rollout'])}")
    print(f"  Backward: {ms(phase_result['bwd'])}")
    print(f"  Optimizer step: {ms(phase_result['opt'])}")
    print(f"  Full update (batch=1, K=8): {sec(phase_result['total'])}")
    print(f"  Peak GPU alloc: {phase_result['peak_alloc_mb']:.1f} MB")
    print(f"  Peak GPU reserved: {phase_result['peak_reserved_mb']:.1f} MB")
    print(f"  grad_norm: {phase_result['grad_norm']:.4f}  nan/inf: {phase_result['nan_inf']}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
