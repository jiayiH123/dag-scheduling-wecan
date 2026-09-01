"""Save fixed instances + model scores to /tmp/diff_test_inputs/ so that
reference (87b2bb6) and cached (6aafbc5) decoders operate on identical tensors.

Run from current workspace (6aafbc5) before running diff_test_decode.py.
"""
import os, pickle
import numpy as np
import torch
import yaml

from data.instance import GeneratorConfig, RandomDAGGenerator
from data.paper_computation_graph import generate_single_graph
from models.wecan import WeCAN, WeCANConfig

OUT = "/tmp/diff_test_inputs"
os.makedirs(OUT, exist_ok=True)


def load_model(device):
    with open("configs/phase1_paper.yaml") as f:
        raw = yaml.safe_load(f)
    mc = raw["model"]
    cfg = WeCANConfig(
        profile=mc["profile"], high_dim=mc["high_dim"], low_dim=mc["low_dim"],
        weca_heads=mc["weca_heads"], ldd_heads=mc["ldd_heads"],
        ldd_layers=mc["ldd_layers"], dmax=mc["dmax"],
        skip_parameterization=mc["skip_parameterization"],
    )
    m = WeCAN(resource_dims=2, config=cfg).to(device)
    torch.manual_seed(99)  # fixed init
    m.eval()
    return m


def make_rand(n, seed):
    cfg = GeneratorConfig(num_tasks_min=n, num_tasks_max=n, num_pools=3, edge_probability=0.15)
    return RandomDAGGenerator(cfg, seed=seed).generate(name=f"rand_n{n}_s{seed}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = load_model(device)
    torch.save(model.state_dict(), f"{OUT}/model_state.pt")
    print(f"Saved model state → {OUT}/model_state.pt")

    cases = []

    # Random DAG
    for n in [20, 50, 100]:
        for seed in [0, 7, 42]:
            inst = make_rand(n, seed)
            with torch.no_grad():
                out = model(inst)
            key = f"rand_n{n}_s{seed}"
            torch.save(out.task_pool_scores.cpu(), f"{OUT}/{key}_scores.pt")
            torch.save(out.skip_parameters.cpu(), f"{OUT}/{key}_skip.pt")
            with open(f"{OUT}/{key}_instance.pkl", "wb") as f:
                pickle.dump(inst, f)
            cases.append({"key": key, "topology": "rand", "n": n, "seed": seed})
            print(f"  {key}  tasks={inst.num_tasks} edges={len(inst.edges)}")

    # Paper synthetic: 50-task each of layered, erdos_renyi, stochastic_block
    for topo in ["layered", "erdos_renyi", "stochastic_block"]:
        for seed in [0, 7]:
            rng = np.random.default_rng(seed)
            inst = generate_single_graph(topo, rng, f"paper_{topo}_s{seed}", n=50)
            with torch.no_grad():
                out = model(inst)
            key = f"paper_{topo}_s{seed}"
            torch.save(out.task_pool_scores.cpu(), f"{OUT}/{key}_scores.pt")
            torch.save(out.skip_parameters.cpu(), f"{OUT}/{key}_skip.pt")
            with open(f"{OUT}/{key}_instance.pkl", "wb") as f:
                pickle.dump(inst, f)
            cases.append({"key": key, "topology": topo, "n": 50, "seed": seed})
            print(f"  {key}  tasks={inst.num_tasks} edges={len(inst.edges)}")

    import json
    with open(f"{OUT}/cases.json", "w") as f:
        json.dump(cases, f, indent=2)
    print(f"\n{len(cases)} cases saved to {OUT}/")


if __name__ == "__main__":
    main()
