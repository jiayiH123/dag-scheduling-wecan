"""Capture reference outputs from the unmodified SkipExtendedGenerator.

Run this BEFORE applying any generator optimisation to record ground truth.
Output: /tmp/oracle_outputs.json
"""
import json, os, sys
import numpy as np
import torch

from data.instance import GeneratorConfig, RandomDAGGenerator
from data.paper_computation_graph import generate_problem
from models.wecan import WeCAN, WeCANConfig
from scheduler.generator import SkipExtendedGenerator
from scheduler.validator import validate_schedule
import yaml


def make_rand_instance(n: int, seed: int):
    cfg = GeneratorConfig(num_tasks_min=n, num_tasks_max=n, num_pools=3, edge_probability=0.15)
    return RandomDAGGenerator(cfg, seed=seed).generate(name=f"rand_n{n}_s{seed}")


def make_paper_instance(seed: int):
    rng = np.random.default_rng(seed)
    return generate_problem("layered", rng, f"paper_s{seed}")


def load_model(device):
    with open("configs/phase1_paper.yaml") as f:
        raw = yaml.safe_load(f)
    mc = raw["model"]
    from models.wecan import WeCANConfig
    cfg = WeCANConfig(
        profile=mc["profile"], high_dim=mc["high_dim"], low_dim=mc["low_dim"],
        weca_heads=mc["weca_heads"], ldd_heads=mc["ldd_heads"],
        ldd_layers=mc["ldd_layers"], dmax=mc["dmax"],
        skip_parameterization=mc["skip_parameterization"],
    )
    model = WeCAN(resource_dims=2, config=cfg).to(device)
    model.eval()
    return model


def capture_greedy(instance, model, device, gen):
    from scheduler.validator import validate_schedule
    with torch.no_grad():
        out = model(instance)
    tr = gen.decode(instance, out.task_pool_scores, out.skip_parameters, mode="greedy")
    val = validate_schedule(instance, tr.schedule)
    return {
        "mode": "greedy",
        "action_count": tr.action_count,
        "skip_count": tr.skip_count,
        "active_wait_count": tr.active_wait_count,
        "passive_time_advance_count": tr.passive_time_advance_count,
        "makespan": float(tr.schedule.makespan),
        "decisions": list(tr.decisions),
        "placements": [
            {"task": p.task, "pool": p.pool, "start": float(p.start), "end": float(p.end)}
            for p in sorted(tr.schedule.placements, key=lambda p: p.task)
        ],
        "feasible": val.feasible,
    }


def capture_sample(instance, model, device, gen, rng_seed: int):
    from scheduler.validator import validate_schedule
    with torch.no_grad():
        out = model(instance)
    g = torch.Generator(device=device)
    g.manual_seed(rng_seed)
    tr = gen.decode(
        instance, out.task_pool_scores, out.skip_parameters,
        mode="sample", generator=g, track_log_probability=True,
    )
    val = validate_schedule(instance, tr.schedule)
    return {
        "mode": "sample",
        "rng_seed": rng_seed,
        "action_count": tr.action_count,
        "skip_count": tr.skip_count,
        "active_wait_count": tr.active_wait_count,
        "passive_time_advance_count": tr.passive_time_advance_count,
        "makespan": float(tr.schedule.makespan),
        "decisions": list(tr.decisions),
        "placements": [
            {"task": p.task, "pool": p.pool, "start": float(p.start), "end": float(p.end)}
            for p in sorted(tr.schedule.placements, key=lambda p: p.task)
        ],
        "log_probability": float(tr.log_probability.detach().cpu()),
        "feasible": val.feasible,
    }


def main():
    print(f"PID={os.getpid()}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    model = load_model(device)
    gen = SkipExtendedGenerator()

    cases = []

    # Random DAG instances: N=20, 50, 100 × 3 seeds
    for n in [20, 50, 100]:
        for seed in [0, 7, 42]:
            print(f"  rand n={n} seed={seed}", flush=True)
            inst = make_rand_instance(n, seed)
            for sample_seed in [1, 2, 3]:
                rec = {
                    "instance": f"rand_n{n}_s{seed}",
                    "num_tasks": inst.num_tasks,
                    "num_edges": len(inst.edges),
                }
                rec["greedy"] = capture_greedy(inst, model, device, gen)
                rec["sample"] = capture_sample(inst, model, device, gen, rng_seed=sample_seed)
                cases.append(rec)
                break  # one sample seed per instance is enough for oracle

    # Paper computation graph: 1 instance (small: 1 graph × 50 tasks instead of full 500)
    from data.paper_computation_graph import generate_single_graph
    for seed in [0, 7]:
        rng = np.random.default_rng(seed)
        inst = generate_single_graph("layered", rng, f"paper50_s{seed}", n=50)
        print(f"  paper50 seed={seed}", flush=True)
        rec = {
            "instance": f"paper50_s{seed}",
            "num_tasks": inst.num_tasks,
            "num_edges": len(inst.edges),
        }
        rec["greedy"] = capture_greedy(inst, model, device, gen)
        rec["sample"] = capture_sample(inst, model, device, gen, rng_seed=1)
        cases.append(rec)

    out_path = "/tmp/oracle_outputs.json"
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    torch.save(model.state_dict(), "/tmp/oracle_model_state.pt")
    print(f"\nOracle saved to {out_path}  ({len(cases)} cases)")
    print("Model state saved to /tmp/oracle_model_state.pt")

    # Summary
    all_feasible = all(c["greedy"]["feasible"] and c["sample"]["feasible"] for c in cases)
    print(f"All schedules feasible: {all_feasible}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
