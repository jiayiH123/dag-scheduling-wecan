"""Static decode-action safety bounds shared by generator and trace validation."""

from __future__ import annotations

from data.instance import DAGInstance


def max_decode_actions(instance: DAGInstance) -> int:
    """Return a conservative finite action bound for an event-driven decode.

    Phase-1 has at most one dispatch and one completion-driven advance per task. A
    communication-enabled decode may additionally advance to a release time before a
    candidate task--pool pair becomes dispatchable. The number of such distinct
    candidate relations is bounded by compatible task--pool pairs. This remains a
    safety guard against non-advancing loops, not a feasibility constraint.
    """
    base = 2 * instance.num_tasks
    if not instance.communication_enabled:
        return base
    compatible_pairs = sum(
        instance.compatibility[task][pool] > 0
        for task in range(instance.num_tasks)
        for pool in range(instance.num_pools)
    )
    return base + compatible_pairs
