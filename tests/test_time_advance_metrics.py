from __future__ import annotations

import torch

from data.instance import DAGInstance
from scheduler.generator import SkipExtendedGenerator


def test_generator_counts_active_wait_and_passive_time_advance_separately() -> None:
    instance = DAGInstance(
        "time_advance_split",
        (1.0, 1.0, 1.0),
        ((0.0, 1.0), (1.0, 0.0), (1.0, 0.0)),
        ((1.0, 1.0),),
        ((1.0,), (1.0,), (1.0,)),
        ((0, 2),),
    )
    # At t=0, dispatches 0 and 1 are available; a strong skip score causes one
    # active wait. Once only task 2 remains blocked, the later skip is passive.
    scores = torch.zeros((3, 1))
    trace = SkipExtendedGenerator().decode(
        instance,
        scores,
        torch.tensor([100.0, 100.0, 1e-4]),
        mode="greedy",
    )
    assert trace.skip_count == trace.active_wait_count + trace.passive_time_advance_count
    assert trace.active_wait_count >= 1
    assert trace.passive_time_advance_count >= 1
