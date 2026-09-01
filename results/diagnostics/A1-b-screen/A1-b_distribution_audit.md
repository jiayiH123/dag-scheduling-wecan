# A1-b Existing Distribution Audit

## Result

All 100 existing candidates have external Greedy gap 0. CA-HEFT's aggregate gap is also reported in the JSON audit.

## Why the existing distribution is easy for external Greedy

- Compatibility coefficients are all 1, so task duration has zero spread across compatible pools.
- Each task has 2.60 compatible pools on average out of 3 (only 13.35% compatibility sparsity), so many different assignments are makespan-equivalent.
- Mean max demand/capacity ratio is only about 0.408; only 9.2% of active Greedy resource intervals hit a capacity boundary.
- Random layered DAGs have moderate density (about 0.396) but no constructed critical-path versus branch competition for a uniquely fast scarce pool.
- Although only 43.9% of Greedy assignments match the selected MILP assignment, every candidate has equal makespan and a structurally different equal-makespan schedule. This is direct evidence of abundant alternative optima, not evidence that the schedules are identical.
- CA-HEFT reaches the MILP optimum in 99 of 100 instances; its only observed absolute gap is 2 ticks.

See `A1-b_distribution_audit.json` for every saved candidate and aggregate statistic.
