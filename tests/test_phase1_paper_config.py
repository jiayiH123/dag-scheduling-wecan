from __future__ import annotations

from pathlib import Path

import torch

from data.instance import GeneratorConfig, generate_dataset
from environment.config import load_yaml
from models.wecan import WeCAN, WeCANConfig
from scheduler.generator import SkipExtendedGenerator, skip_log_score
from scheduler.validator import validate_schedule
from training.reinforce import ReinforceTrainer, TrainConfig


ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = ROOT / "configs" / "phase1_paper.yaml"


def _load_runtime_configs(tmp_path: Path) -> tuple[dict, GeneratorConfig, WeCANConfig, TrainConfig]:
    configuration = load_yaml(PAPER_CONFIG)
    generator_config = GeneratorConfig(**configuration["generator"])
    model_config = WeCANConfig(**configuration["model"])
    training_values = dict(configuration["training"])
    training_values["baseline_mode"] = training_values.pop("reinforce_baseline")
    training_values.update(seed=configuration["seed"], checkpoint_dir=str(tmp_path))
    train_config = TrainConfig(**training_values)
    return configuration, generator_config, model_config, train_config


def test_paper_config_is_complete_and_matches_declared_contract(tmp_path: Path) -> None:
    configuration, generator_config, model_config, train_config = _load_runtime_configs(tmp_path)
    specification = configuration["paper_specification"]
    assumptions = configuration["implementation_assumptions"]

    assert generator_config.resource_dims == 2
    assert specification == {
        "task_input_dim": 3,
        "pool_input_dim": 2,
        "high_dim": 512,
        "low_dim": 128,
        "weca_heads": 8,
        "ldd_layers": 8,
        "ldd_heads": 16,
        "ldd_mask_types": 8,
        "heads_per_ldd_mask": 2,
        "dmax": 500,
        "skip_hidden_layers": 2,
        "skip_hidden_dim": 64,
        "skip_parameterization": "paper_sigmoid",
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "batch_size": 64,
        "training_batches": 800,
        "reinforce_baseline": "instance_mean",
    }
    assert model_config.profile == "paper"
    assert (model_config.high_dim, model_config.low_dim) == (512, 128)
    assert (model_config.weca_heads, model_config.ldd_heads, model_config.ldd_layers) == (8, 16, 8)
    assert model_config.dmax == 500
    assert model_config.skip_hidden_dim == 64
    assert model_config.skip_parameterization == "paper_sigmoid"
    assert train_config.optimizer == "adam"
    assert train_config.learning_rate == 1e-4
    assert train_config.batch_size == 64
    assert train_config.max_updates == 800
    assert train_config.baseline_mode == "instance_mean"
    assert assumptions["alternating_weca_layers"] == model_config.alternating_weca_layers
    assert assumptions["rollouts_per_instance"] == train_config.rollouts_per_instance
    assert assumptions["ldd_bias"] == "head_specific_pending_paper_code_confirmation"


def test_paper_config_forward_rollout_skip_and_reinforce_backward(tmp_path: Path) -> None:
    _, generator_config, model_config, train_config = _load_runtime_configs(tmp_path)
    tiny_generator = GeneratorConfig(
        **{
            **generator_config.__dict__,
            "num_tasks_min": 4,
            "num_tasks_max": 4,
            "num_pools": 2,
            "max_width": 2,
        }
    )
    instances = generate_dataset(tiny_generator, count=1, seed=2026, prefix="paper-config-test")
    model = WeCAN(tiny_generator.resource_dims, model_config)
    model.reset_forward_counter()
    output = model(instances[0])

    assert model.task_embedder.layers[0].in_features == 3
    assert model.pool_embedder.layers[0].in_features == 2
    assert output.task_pool_scores.shape == (4, 2)
    assert torch.all((output.skip_parameters > 0) & (output.skip_parameters < 1))
    assert skip_log_score(output.skip_parameters, 0, 4) > skip_log_score(output.skip_parameters, 1, 4)

    trace = SkipExtendedGenerator().decode(
        instances[0],
        output.task_pool_scores,
        output.skip_parameters,
        mode="sample",
        generator=torch.Generator().manual_seed(2026),
        track_log_probability=True,
    )
    assert model.forward_calls == 1
    assert trace.log_probability is not None
    assert trace.skip_count > 0
    assert validate_schedule(instances[0], trace.schedule, trace=trace.decisions).feasible

    smoke_config = TrainConfig(
        optimizer=train_config.optimizer,
        learning_rate=train_config.learning_rate,
        batch_size=train_config.batch_size,
        epochs=1,
        max_updates=1,
        rollouts_per_instance=train_config.rollouts_per_instance,
        baseline_mode=train_config.baseline_mode,
        snapshot_sync_interval_updates=train_config.snapshot_sync_interval_updates,
        seed=train_config.seed,
        checkpoint_dir=str(tmp_path),
    )
    history = ReinforceTrainer(model, smoke_config, torch.device("cpu")).train(instances, instances)
    assert len(history) == 1
    for key in ("policy_loss", "mean_makespan", "gradient_norm", "skip_ratio"):
        assert key in history[0]
        assert torch.isfinite(torch.tensor(float(history[0][key])))
