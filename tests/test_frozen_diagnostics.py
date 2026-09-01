from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.algorithms import ca_heft_schedule, greedy_schedule
from data.instance import DAGInstance
from models.wecan import WeCANConfig
from oracle.milp_oracle import solve_milp_oracle
from scheduler.validator import validate_schedule
from scripts.run_diagnostics import _assert_safe_output, load_frozen_instance
from training.diagnostics import run_a1c_frozen_instance_gate
from training.reinforce import TrainConfig


FROZEN_PATH = Path("results/diagnostics/A1-c-screen/selected_instance.json")
FROZEN_RESULT_PATH = Path("results/diagnostics/A1-c-screen/selected_result.json")
FROZEN_A1B_EXACT_PATH = Path("results/diagnostics/A1-b-exact-screen/selected_instance.json")
FROZEN_A1B_EXACT_RESULT_PATH = Path("results/diagnostics/A1-b-exact-screen/selected_result.json")


def test_frozen_a1c_instance_round_trips_exactly_from_json() -> None:
    payload = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    instance = load_frozen_instance(FROZEN_PATH)
    assert instance == DAGInstance.from_dict(payload)
    assert json.loads(json.dumps(instance.to_dict())) == payload


def test_frozen_a1c_oracle_heuristics_and_validator_match_screened_result() -> None:
    instance = load_frozen_instance(FROZEN_PATH)
    screened = json.loads(FROZEN_RESULT_PATH.read_text(encoding="utf-8"))
    oracle = solve_milp_oracle(instance)
    greedy = greedy_schedule(instance)
    ca_heft = ca_heft_schedule(instance)
    assert oracle.status == "optimal"
    assert oracle.makespan == screened["milp"]["makespan"] == 13.0
    assert greedy.makespan == screened["external_greedy"]["schedule"]["makespan"] == 22.0
    assert ca_heft.makespan == screened["ca_heft"]["schedule"]["makespan"] == 13.0
    assert validate_schedule(instance, oracle.schedule).feasible
    assert validate_schedule(instance, greedy).feasible
    assert validate_schedule(instance, ca_heft).feasible


def test_protected_historical_output_directories_are_rejected() -> None:
    with pytest.raises(ValueError, match="protected historical"):
        _assert_safe_output(Path("results/diagnostics/A1"))
    with pytest.raises(ValueError, match="protected historical"):
        _assert_safe_output(Path("results/diagnostics/A1-c-screen"))
    with pytest.raises(ValueError, match="protected historical"):
        _assert_safe_output(Path("results/diagnostics/A1-b-exact-screen"))


def test_frozen_a1c_gate_rejects_nonempty_output_before_training(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("preserve", encoding="utf-8")
    instance = load_frozen_instance(FROZEN_PATH)
    with pytest.raises(FileExistsError, match="nonempty"):
        run_a1c_frozen_instance_gate(
            instance,
            WeCANConfig(profile="smoke", high_dim=32, low_dim=32, weca_heads=4, ldd_heads=4, ldd_layers=1, alternating_weca_layers=1),
            TrainConfig(max_updates=1, epochs=1, batch_size=1, rollouts_per_instance=4, checkpoint_dir=str(output / "checkpoint")),
            output,
        )
    assert (output / "sentinel").read_text(encoding="utf-8") == "preserve"


def test_frozen_a1b_exact_values_match_screened_result() -> None:
    instance = load_frozen_instance(FROZEN_A1B_EXACT_PATH)
    screened = json.loads(FROZEN_A1B_EXACT_RESULT_PATH.read_text(encoding="utf-8"))
    oracle = solve_milp_oracle(instance)
    greedy = greedy_schedule(instance)
    ca_heft = ca_heft_schedule(instance)
    assert instance.name == "a1b-exact-seed-4002"
    assert oracle.status == "optimal" and oracle.makespan == screened["milp"]["makespan"] == 8.0
    assert greedy.makespan == screened["external_greedy"]["schedule"]["makespan"] == 10.0
    assert ca_heft.makespan == screened["ca_heft"]["schedule"]["makespan"] == 8.0
    assert validate_schedule(instance, oracle.schedule).feasible
    assert validate_schedule(instance, greedy).feasible
    assert validate_schedule(instance, ca_heft).feasible
