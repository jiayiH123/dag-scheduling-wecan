#!/usr/bin/env python3
"""Train the Phase-1 WeCAN faithful reimplementation with REINFORCE."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.instance import load_dataset
from environment.config import load_yaml
from models.wecan import WeCAN, WeCANConfig
from training.reinforce import ReinforceTrainer, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--train-data", default="data/generated/train.json")
    parser.add_argument("--validation-data", default="data/generated/validation.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/wecan_phase1")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    configuration = load_yaml(args.config)
    train_instances = load_dataset(args.train_data)
    validation_instances = load_dataset(args.validation_data)
    model_config = WeCANConfig(**configuration["model"])
    model = WeCAN(train_instances[0].resource_dims, model_config)
    resume_epoch = 0
    resume_optimizer_state = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        resume_epoch = int(checkpoint.get("epoch", 0))
        resume_optimizer_state = checkpoint.get("optimizer_state")
    training_values = dict(configuration["training"])
    training_values.update(seed=configuration["seed"], checkpoint_dir=args.checkpoint_dir)
    training_values["baseline_mode"] = training_values.pop("reinforce_baseline", "instance_leave_one_out")
    if args.epochs is not None:
        training_values["epochs"] = args.epochs
    allowed = {
        "optimizer", "learning_rate", "batch_size", "epochs", "max_updates", "rollouts_per_instance",
        "baseline_mode", "snapshot_sync_interval_updates", "seed", "checkpoint_dir",
    }
    train_config = TrainConfig(**{key: value for key, value in training_values.items() if key in allowed})
    trainer = ReinforceTrainer(model, train_config, torch.device(args.device))
    if resume_optimizer_state is not None:
        trainer.optimizer.load_state_dict(resume_optimizer_state)
    history = trainer.train(train_instances, validation_instances, start_epoch=resume_epoch)
    print("Last epoch:", history[-1])
    print(f"Best checkpoint: {Path(args.checkpoint_dir) / 'best.pt'}")


if __name__ == "__main__":
    main()
