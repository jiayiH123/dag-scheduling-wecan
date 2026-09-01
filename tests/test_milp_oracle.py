from __future__ import annotations

from data.instance import DAGInstance
from oracle.common import make_tick_instance
from oracle.milp_oracle import solve_milp_oracle
from scheduler.validator import validate_schedule
from tests.fixtures import handcrafted_instances


def test_milp_solves_all_handcrafted_instances_to_proven_optimal() -> None:
    for name, instance in handcrafted_instances().items():
        result = solve_milp_oracle(instance)
        assert result.status == "optimal", (name, result.to_dict())
        assert result.schedule is not None
        assert result.validator is not None and result.validator.feasible
        assert validate_schedule(instance, result.schedule).feasible
        assert result.detail["cbc_audit"]["proven_optimal"]
        assert not result.detail["cbc_audit"]["time_limit_reached"]


def test_tick_conversion_accepts_exact_explicit_integer_multiple_scale() -> None:
    instance = handcrafted_instances()["single_chain"]
    converted = make_tick_instance(instance, time_scale=3)
    assert converted is not None
    assert converted.duration_ticks == ((6,), (9,), (6,))


def test_milp_reports_unsupported_scale_without_rounding() -> None:
    instance = DAGInstance(
        "non_exact_scale",
        (1.0,),
        ((1.0,),),
        ((1.0,),),
        ((1.0 / 10007.0,),),
        (),
    )
    instance.validate()
    result = solve_milp_oracle(instance)
    assert result.status == "unsupported_time_scale"
