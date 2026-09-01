# A1-b-scale: Structured Larger-Instance Proposal (Not an Exact Training Gate)

## Status

This document replaces the former ambiguous A1-b proposal. A1-b-scale is an **8–10 task generator and difficulty-distribution design only**. It is not a strict A1-b training object in this phase because complete `with_wait == without_wait` proof is not available for every 8–10 task instance.

No A1-b-scale candidate generation, Oracle screen, or training runs are authorized by this configuration.

## Future uses

After separate approval, A1-b-scale may support:

1. fixed small-set training;
2. external Greedy gap distribution studies;
3. scale-generalization checks;
4. Phase 2 data-generation reference work.

It must not be called “A1-b-exact” and must not substitute for the exact 6–7 task diagnostic.

## Design

- 8–10 tasks; 2 resource dimensions; 2 pools; integer ticks.
- fast/tight pool capacity `(7,7)` with short 1–3 tick task durations.
- slow/loose pool capacity `(14,14)` with 2–3× duration penalty.
- 4–5 task critical spine and 3–5 branches competing for fast capacity.
- 55–85% key-dimension fast-pool demands, sparse task-specific compatibility, and branch/critical resource preferences.

The corresponding non-executed configuration is `configs/a1b_scale_proposed.yaml`.
