from __future__ import annotations

from oracle.exhaustive_oracle import solve_exhaustive_oracle
from oracle.milp_oracle import solve_milp_oracle
from tests.fixtures import handcrafted_instances


def test_all_required_handcrafted_edge_cases_have_proven_oracles() -> None:
    expected = {
        "single_chain",
        "diamond",
        "wide_parallel",
        "capacity_exact_saturation",
        "capacity_forces_delay",
        "incompatible_pools",
        "multiple_optima",
        "active_wait_counterexample",
        "a1c_active_wait_six_task",
        "one_pool_degenerate",
        "symmetric_pools",
    }
    fixtures = handcrafted_instances()
    assert set(fixtures) == expected
    for name, instance in fixtures.items():
        milp = solve_milp_oracle(instance)
        exhaustive = solve_exhaustive_oracle(instance)
        assert milp.status == "optimal", name
        assert exhaustive.status == "optimal", name
        assert abs(milp.makespan - exhaustive.makespan) <= 1e-6, name


def test_symmetric_pools_has_an_optimal_schedule() -> None:
    result = solve_exhaustive_oracle(handcrafted_instances()["symmetric_pools"])
    assert result.status == "optimal"
    assert result.schedule is not None
    assert result.makespan == 6.0
