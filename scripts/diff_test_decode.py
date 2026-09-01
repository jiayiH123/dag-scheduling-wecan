"""Run greedy + sample decode for all saved inputs and write results to JSON.

This script is designed to run UNCHANGED in both codebases:
  - reference worktree (87b2bb6): will use old instance.parents-based decode
  - current workspace (6aafbc5):  will use _TopologyCache-based decode

Usage (from project root of either codebase):
  python scripts/diff_test_decode.py <inputs_dir> <output_json>

Example:
  cd /tmp/wecan_ref_87b2bb6
  python scripts/diff_test_decode.py /tmp/diff_test_inputs /tmp/diff_ref_results.json

  cd /mnt/volumes/ss-sai-bd-ga/huangjiayi/codes_my_own/dag_0729
  python -m scripts.diff_test_decode /tmp/diff_test_inputs /tmp/diff_cached_results.json
"""
import sys, os, pickle, json
import torch
from scheduler.generator import SkipExtendedGenerator
from scheduler.validator import validate_schedule

SAMPLE_SEEDS = [1, 2, 3]


def decode_case(inst, scores_cpu, skip_cpu, device, gen):
    scores = scores_cpu.to(device)
    skip = skip_cpu.to(device)

    # greedy
    tr_g = gen.decode(inst, scores, skip, mode="greedy")
    val_g = validate_schedule(inst, tr_g.schedule)
    greedy = {
        "action_count": tr_g.action_count,
        "skip_count": tr_g.skip_count,
        "active_wait_count": tr_g.active_wait_count,
        "passive_time_advance_count": tr_g.passive_time_advance_count,
        "makespan": float(tr_g.schedule.makespan),
        "decisions": list(tr_g.decisions),
        "placements": [
            {"task": p.task, "pool": p.pool, "start": float(p.start), "end": float(p.end)}
            for p in sorted(tr_g.schedule.placements, key=lambda p: p.task)
        ],
        "feasible": val_g.feasible,
    }

    # sample × multiple seeds
    samples = []
    for rng_seed in SAMPLE_SEEDS:
        g = torch.Generator(device=device)
        g.manual_seed(rng_seed)
        tr_s = gen.decode(inst, scores, skip, mode="sample",
                          generator=g, track_log_probability=True)
        val_s = validate_schedule(inst, tr_s.schedule)
        samples.append({
            "rng_seed": rng_seed,
            "action_count": tr_s.action_count,
            "skip_count": tr_s.skip_count,
            "active_wait_count": tr_s.active_wait_count,
            "passive_time_advance_count": tr_s.passive_time_advance_count,
            "makespan": float(tr_s.schedule.makespan),
            "decisions": list(tr_s.decisions),
            "placements": [
                {"task": p.task, "pool": p.pool, "start": float(p.start), "end": float(p.end)}
                for p in sorted(tr_s.schedule.placements, key=lambda p: p.task)
            ],
            "log_probability": float(tr_s.log_probability.detach().cpu()),
            "feasible": val_s.feasible,
        })

    return {"greedy": greedy, "samples": samples}


def main():
    inputs_dir = sys.argv[1]
    output_path = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = SkipExtendedGenerator()

    with open(f"{inputs_dir}/cases.json") as f:
        cases = json.load(f)

    results = {}
    for c in cases:
        key = c["key"]
        print(f"  {key} ...", flush=True)
        with open(f"{inputs_dir}/{key}_instance.pkl", "rb") as f:
            inst = pickle.load(f)
        scores_cpu = torch.load(f"{inputs_dir}/{key}_scores.pt", map_location="cpu")
        skip_cpu = torch.load(f"{inputs_dir}/{key}_skip.pt", map_location="cpu")
        results[key] = decode_case(inst, scores_cpu, skip_cpu, device, gen)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results → {output_path}")


if __name__ == "__main__":
    main()
