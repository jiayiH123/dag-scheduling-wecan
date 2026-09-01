"""Small, discrete-time hand-crafted fixtures for Oracle and validator tests."""

from __future__ import annotations

from data.instance import DAGInstance


def handcrafted_instances() -> dict[str, DAGInstance]:
    """Ten integer-time fixtures covering Oracle edge cases required for Phase 1.5."""
    fixtures = {
        "single_chain": DAGInstance(
            "single_chain", (2.0, 3.0, 2.0), ((1.0,),) * 3, ((1.0,),), ((1.0,),) * 3, ((0, 1), (1, 2))
        ),
        "diamond": DAGInstance(
            "diamond", (2.0, 3.0, 3.0, 2.0), ((1.0,),) * 4, ((2.0,),), ((1.0,),) * 4,
            ((0, 1), (0, 2), (1, 3), (2, 3)),
        ),
        "wide_parallel": DAGInstance(
            "wide_parallel", (2.0, 2.0, 2.0, 2.0, 1.0), ((1.0,),) * 5, ((4.0,),), ((1.0,),) * 5,
            ((0, 4), (1, 4), (2, 4), (3, 4)),
        ),
        "capacity_exact_saturation": DAGInstance(
            "capacity_exact_saturation", (3.0, 3.0, 1.0), ((2.0,), (2.0,), (1.0,)), ((4.0,),), ((1.0,),) * 3,
            ((0, 2), (1, 2)),
        ),
        "capacity_forces_delay": DAGInstance(
            "capacity_forces_delay", (4.0, 3.0, 1.0), ((2.0,), (2.0,), (1.0,)), ((3.0,),), ((1.0,),) * 3,
            ((0, 2), (1, 2)),
        ),
        "incompatible_pools": DAGInstance(
            "incompatible_pools", (2.0, 4.0, 2.0), ((1.0,),) * 3, ((2.0,), (2.0,)),
            ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), ((0, 2), (1, 2)),
        ),
        "multiple_optima": DAGInstance(
            "multiple_optima", (3.0, 3.0), ((1.0,), (1.0,)), ((1.0,), (1.0,)), ((1.0, 1.0), (1.0, 1.0)), (),
        ),
        # At t=0 root and blocker are feasible in distinct resource dimensions. After root
        # finishes, critical needs blocker's dimension and unlocks a long task in the other.
        # The optimum actively waits instead of launching blocker, then overlaps blocker with
        # the descendant; every non-delay launch of blocker yields a longer makespan.
        "active_wait_counterexample": DAGInstance(
            "active_wait_counterexample", (1.0, 10.0, 1.0, 10.0),
            ((0.0, 1.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((1.0, 1.0),), ((1.0,),) * 4,
            ((0, 2), (2, 3)),
        ),
        # A1-c is a six-task extension of the active-wait construction. At t=0,
        # root 0 occupies y while independent blocker 1 is feasible on x. Launching
        # blocker is non-delay but postpones critical task 2, which unlocks long y
        # task 3. Waiting for root, then running 2 before blocker, overlaps 1 and 3.
        # The dependent tails 4 and 5 preserve the strict 13-vs-22 makespan gap.
        "a1c_active_wait_six_task": DAGInstance(
            "a1c_active_wait_six_task", (1.0, 10.0, 1.0, 10.0, 1.0, 1.0),
            ((0.0, 1.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0)),
            ((1.0, 1.0),), ((1.0,),) * 6,
            ((0, 2), (2, 3), (1, 4), (3, 5)),
        ),
        "one_pool_degenerate": DAGInstance(
            "one_pool_degenerate", (2.0, 4.0, 3.0, 1.0), ((1.0,),) * 4, ((2.0,),), ((1.0,),) * 4,
            ((0, 2), (1, 2), (2, 3)),
        ),
        "symmetric_pools": DAGInstance(
            "symmetric_pools", (2.0, 2.0, 3.0, 1.0), ((1.0,),) * 4, ((2.0,), (2.0,)),
            ((1.0, 1.0),) * 4, ((0, 2), (1, 2), (2, 3)),
        ),
    }
    for instance in fixtures.values():
        instance.validate()
    return fixtures
