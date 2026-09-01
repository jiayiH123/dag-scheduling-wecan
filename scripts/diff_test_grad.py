"""Gradient equivalence test: run sample decode with track_log_probability=True,
call backward, save task_pool_scores.grad and skip_parameters.grad.

Designed to run UNCHANGED in both 87b2bb6 and 6aafbc5 codebases.

Usage (from project root):
  python scripts/diff_test_grad.py <inputs_dir> <output_pt>

Runs on a small subset (N<=50) for speed.
"""
import sys, os, pickle
import torch
from scheduler.generator import SkipExtendedGenerator

GRAD_SAMPLE_SEED = 42
GRAD_KEYS = ["rand_n20_s0", "rand_n20_s7", "rand_n50_s0", "paper_layered_s0", "paper_erdos_renyi_s0", "paper_stochastic_block_s0"]


def main():
    inputs_dir = sys.argv[1]
    output_path = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = SkipExtendedGenerator()

    grad_results = {}
    for key in GRAD_KEYS:
        with open(f"{inputs_dir}/{key}_instance.pkl", "rb") as f:
            inst = pickle.load(f)
        scores_cpu = torch.load(f"{inputs_dir}/{key}_scores.pt", map_location="cpu")
        skip_cpu = torch.load(f"{inputs_dir}/{key}_skip.pt", map_location="cpu")

        # requires_grad on a fresh clone
        scores = scores_cpu.to(device).detach().requires_grad_(True)
        skip = skip_cpu.to(device).detach().requires_grad_(True)

        g = torch.Generator(device=device)
        g.manual_seed(GRAD_SAMPLE_SEED)
        tr = gen.decode(inst, scores, skip, mode="sample",
                        generator=g, track_log_probability=True)
        logp = tr.log_probability
        logp.backward()

        grad_results[key] = {
            "logp": float(logp.detach().cpu()),
            "decisions": list(tr.decisions),
            "scores_grad": scores.grad.cpu(),
            "skip_grad": skip.grad.cpu(),
        }
        print(f"  {key}: logp={logp.item():.6f}  scores_grad_norm={scores.grad.norm().item():.6f}", flush=True)

    torch.save(grad_results, output_path)
    print(f"Saved grad results → {output_path}")


if __name__ == "__main__":
    main()
