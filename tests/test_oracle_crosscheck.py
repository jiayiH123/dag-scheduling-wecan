from __future__ import annotations

import pytest

from baselines.algorithms import greedy_schedule, heft_schedule, random_schedule
from oracle.exhaustive_oracle import solve_exhaustive_oracle
from oracle.milp_oracle import solve_milp_oracle
from scheduler.validator import validate_schedule
from tests.fixtures import handcrafted_instances, random_tiny_instances

EPS = 1e-6


def _crosscheck(instance) -> None:
    milp = solve_milp_oracle(instance)
    exhaustive = solve_exhaustive_oracle(instance)
    assert milp.status == "optimal", milp.to_dict()
    assert exhaustive.status == "optimal", exhaustive.to_dict()
    assert milp.schedule is not None and exhaustive.schedule is not None
    assert milp.validator is not None and milp.validator.feasible
    assert exhaustive.validator is not None and exhaustive.validator.feasible
    assert abs(milp.makespan - exhaustive.makespan) <= EPS, (milp.to_dict(), exhaustive.to_dict())
    for schedule in (random_schedule(instance, 17), greedy_schedule(instance), heft_schedule(instance)):
        validation = validate_schedule(instance, schedule)
        assert validation.feasible, validation.violations
        assert not schedule.makespan < milp.makespan - EPS, (schedule.makespan, milp.makespan, instance.name)


def test_handcrafted_oracles_crosscheck() -> None:
    for instance in handcrafted_instances().values():
        _crosscheck(instance)


@pytest.mark.slow
def test_fifty_random_tiny_instances_crosscheck() -> None:
    for instance in random_tiny_instances(50):
        _crosscheck(instance)
