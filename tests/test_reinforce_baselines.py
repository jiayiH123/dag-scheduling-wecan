from __future__ import annotations

import torch

from data.instance import generate_dataset, GeneratorConfig
from models.wecan import WeCAN, WeCANConfig
from training.reinforce import ReinforceTrainer, TrainConfig, compute_advantages


def test_advantage_modes_follow_expected_math() -> None:
    makespans = torch.tensor([[2.0, 6.0, 10.0, 14.0], [4.0, 8.0, 12.0, 16.0]])
    assert torch.allclose(compute_advantages(makespans, "batch_global_mean"), makespans - 9.0)
    assert torch.allclose(compute_advantages(makespans, "instance_mean"), torch.tensor([[-6., -2., 2., 6.], [-6., -2., 2., 6.]]))
    loo = compute_advantages(makespans, "instance_leave_one_out")
    assert torch.allclose(loo, torch.tensor([[-8., -8/3, 8/3, 8.], [-8., -8/3, 8/3, 8.]]))
    greedy = torch.tensor([5.0, 9.0])
    assert torch.allclose(compute_advantages(makespans, "current_policy_greedy", greedy), makespans - greedy[:, None])
    assert torch.allclose(compute_advantages(makespans, "snapshot_greedy_rollout", greedy), makespans - greedy[:, None])


def test_loobaseline_requires_multiple_trajectories() -> None:
    try:
        TrainConfig(rollouts_per_instance=1, baseline_mode="instance_leave_one_out")
    except ValueError:
        pass
    else:
        raise AssertionError("LOO with one trajectory must fail.")


def test_k_trajectory_update_logs_required_diagnostics_and_snapshot_sync(tmp_path) -> None:
    instances = generate_dataset(
        GeneratorConfig(num_tasks_min=4, num_tasks_max=4, num_pools=2, resource_dims=2, max_width=2,
                        duration_low=1, duration_high=3, capacity_low=4, capacity_high=6,
                        demand_fraction_low=.2, demand_fraction_high=.5, compatibility_low=1, compatibility_high=1),
        count=2, seed=31, prefix="train",
    )
    model = WeCAN(2, WeCANConfig(profile="smoke", high_dim=32, low_dim=32, weca_heads=4, ldd_heads=4, ldd_layers=1, alternating_weca_layers=1))
    trainer = ReinforceTrainer(model, TrainConfig(
        batch_size=2, epochs=1, max_updates=1, rollouts_per_instance=4,
        baseline_mode="snapshot_greedy_rollout", snapshot_sync_interval_updates=1,
        checkpoint_dir=str(tmp_path), seed=31,
    ), torch.device("cpu"))
    history = trainer.train(instances, instances)
    row = history[0]
    for key in (
        "policy_loss", "mean_advantage", "advantage_std", "entropy", "gradient_norm",
        "skip_ratio", "rollout_seconds_total_per_instance", "rollout_seconds_p95",
    ):
        assert key in row
        assert torch.isfinite(torch.tensor(float(row[key])))
    assert row["rollouts_per_instance"] == 4
    assert row["snapshot_extra_forward"] is True
    assert trainer.baseline_model is not None
    assert all(not parameter.requires_grad for parameter in trainer.baseline_model.parameters())
