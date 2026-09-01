# Phase 1.6 Fixed Small-Set Training Report

**Training seed:** 2026  
**Model:** smoke profile, one model forward per instance, K=8 reused trajectories  
**Optimizer protocol:** instance-local normalized Oracle cost LOO, instance-equal batch loss  
**Scope:** one 2-update dry run and one 300-update formal run; no additional seeds or stages.

## Dry run

`results/phase16_train_dryrun/` passed:

- train records: 24; validation records: 12; test metadata only: 30;
- 2 updates completed with balanced rotations 3/3/2 then 3/2/3;
- every trajectory was Validator-feasible and trace-valid;
- normalized LOO advantages, one-forward/K-reuse, and nonempty-output protection passed;
- no test instance JSON was loaded or evaluated in the dry run.

## Formal training

Output: `results/phase16_train_seed2026/`.

| Item | Result |
|---|---:|
| Updates | 300 |
| Best validation update | 179 |
| Train sampled F1/F2/F3 counts | 800 / 800 / 800 |
| Sampling | with replacement |
| Per-instance count range | 84–122 (range 38) |
| Training trajectory feasibility | 100% |
| Validation feasibility at every checkpoint | 100% |
| Nonfinite metrics | none |

The initial validation mean normalized Oracle gap was 0.6662. The locked best-validation checkpoint at update 179 reached 0.0000, a 100% relative reduction. Ties at later zero-gap validations were resolved in favor of the earlier update.

## Test isolation

The test split was not loaded/evaluated during training or checkpoint selection. After checkpoint selection locked, it was evaluated once for each state: `initial`, `best-validation`, and `last`.

## Test summary

| State | Mean normalized gap | Median gap | Worst gap | Feasibility | Greedy win rate | CA-HEFT win rate |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 0.5976 | 0.5714 | 0.9375 | 100% | 0% | 0% |
| Best validation | 0.0000 | 0.0000 | 0.0000 | 100% | 66.7% | 26.7% |
| Last | 0.0000 | 0.0000 | 0.0000 | 100% | 66.7% | 26.7% |

### Best-validation family results

| Family | Policy greedy | Sample mean | Best-of-8 | Oracle | External Greedy | CA-HEFT | Greedy wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | 15.2 | 15.275 | 15.2 | 15.2 | 15.2 | 15.2 | 0 / 10 |
| F2 | 10.2 | 10.2 | 10.2 | 10.2 | 12.2 | 10.6 | 10 / 10 |
| F3 | 14.7 | 18.55 | 14.7 | 14.7 | 22.2 | 18.9 | 10 / 10 |
| Overall | 13.367 | 14.675 | 13.367 | 13.367 | 16.533 | 14.9 | 20 / 30 |

F2 best trajectories have zero active waits. F3 best trajectories have active waits for every test instance; their no-active-wait greedy counterfactual mean is 22.2 versus learned policy greedy 14.7, while the external Greedy mean is 22.2.

## Required source-version splits

| Group | Count | Mean normalized gap | Greedy win rate | Active-wait ratio |
|---|---:|---:|---:|---:|
| F2 v1 T2 | 6 | 0.0 | 100% | 0.0 |
| F2 v2 T2-v2 | 4 | 0.0 | 100% | 0.0 |
| F2 combined | 10 | 0.0 | 100% | 0.0 |
| F3 v1 T2 | 9 | 0.0 | 100% | 0.0832 |
| F3 v2 T2-v2 | 1 | 0.0 | 100% | 0.0824 |
| F3 combined | 10 | 0.0 | 100% | 0.0832 |

The F3-v2 row is a one-instance report, not a statistical aggregate.

## Terminal optimization diagnostics

| Metric | Value |
|---|---:|
| Entropy | 0.03905 |
| Raw advantage mean / std | approximately 0 / 4.62e-7 |
| Normalized advantage mean / std | 0 / 0 |
| Gradient norm | 0 |
| Active-wait ratio | 0.03448 |
| Passive time-advance ratio | 0.35632 |
| Training forward seconds per instance | 0.00437 |
| Training K-rollout generation seconds per instance | 0.05018 |

The zero terminal normalized advantage/gradient is expected after all K=8 trajectories within an instance converge to equal normalized cost.

## Conclusion

All Phase 1.6 engineering gates passed: feasibility/traces/nonfinite checks, validation normalized-gap improvement, F2 held-out-template improvement over external Greedy without active waits, F3 held-out-template active-wait behavior and improvement over Greedy, test source-version separation, and one-forward-per-instance semantics.

Training stops here. No subsequent phase was started.
