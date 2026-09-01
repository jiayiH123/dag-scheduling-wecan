# Phase 1 Acceptance Report

**Acceptance state:** complete and frozen.  
**Scope:** WeCAN-like Phase 1 functional reimplementation through Phase 1.6 fixed-set engineering diagnostic.  
**Boundary:** no new training or experimental result modification occurred in this closeout.

## Provenance

| Item | Value |
|---|---|
| Git commit | `08dcbc39d437797e9531706780d71c5c82fea9e7` |
| Commit subject | `complete phase16 fixed-set diagnostic` |
| Python | 3.12 environment; exact version in manifest |
| Test suite at closeout | 49 passed |
| Phase16 train seed | 2026 |
| Model profile | smoke |
| REINFORCE baseline | instance_leave_one_out |
| Rollouts per instance | 8 |
| Batch size | 8 |
| Maximum updates | 300 |
| Validation cadence | 20 updates |
| Best validation update | 179 |

## Frozen data and checkpoint identity

The complete SHA256 inventory, paths, file sizes, Python/platform/dependency versions, normalized data configuration hashes, training configuration, checkpoint hashes, and key-result locations is stored in:

```text
results/phase1_acceptance_manifest.json
```

It includes SHA256 values for:

```text
results/phase16_final/train_instances.json
results/phase16_final/validation_instances.json
results/phase16_final/test_instances.json
results/phase16_final/selected_manifest.json
results/phase16_final/oracle_audit.jsonl
results/phase16_train_seed2026/checkpoint/initial.pt
results/phase16_train_seed2026/checkpoint/best.pt
results/phase16_train_seed2026/checkpoint/last.pt
```

The frozen split contains 24 train, 12 validation, and 30 test instances; all 66 structural fingerprints are globally unique and all 66 final semantic rechecks passed.

## Checkpoint selection and test isolation

Checkpoint selection used validation only, ranked by:

1. 100% validation feasibility;
2. lowest validation mean normalized Oracle gap;
3. lower worst-instance normalized Oracle gap;
4. earlier update on remaining ties.

The test split was evaluated only after checkpoint selection locked, and exactly once for each checkpoint state:

```text
initial
best-validation
last
```

It did not determine model parameters, update count, thresholds, sampler, or checkpoint identity.

## Read-only leakage audit

Machine-readable audit:

```text
results/phase1_leakage_audit.json
```

Verified results:

| Audit item | Result |
|---|---|
| Train / validation structural overlap | 0 |
| Train / test structural overlap | 0 |
| Validation / test structural overlap | 0 |
| Test instances in train split/sampler domain | none |
| Family sample totals over 300 updates | F1=800, F2=800, F3=800 |
| Oracle metadata complete for manifest | yes |
| Oracle value part of model DTO/features | no |
| family/template/source version part of model DTO/features | no |
| role-to-final-id part of final model manifest input | no |
| Test-time parameter updates | none |
| Test checkpoint states | initial / best / last only |
| Test rows per checkpoint state | 30 |
| Model forward contract | one forward per instance |
| K trajectory contract | K=8 trajectories reuse static output |
| Train / validation / test trace feasibility | 100% |

## Key result locations

```text
results/phase16_train_dryrun/dry_run_report.json
results/phase16_train_seed2026/history.json
results/phase16_train_seed2026/validation_history.json
results/phase16_train_seed2026/initial_validation_evaluation.json
results/phase16_train_seed2026/test_evaluations.json
results/phase16_train_seed2026/train_report.json
docs/phase16_data_report.md
docs/phase16_train_report.md
```

## Phase boundary

Phase 1 is accepted at its predeclared functional-reimplementation boundary. No additional Phase 1 experiments are authorized by this report. A future Phase 2 must be separately designed and approved before introducing communication delay, cost, energy, or budget constraints.
