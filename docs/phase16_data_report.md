# Phase 1.6 Data Report — v2 Supplement and Final 66-Instance Freeze

**Date:** 2026-08-02  
**Scope:** data generation, exact audit, and finalization only. No Phase 1.6 model training was started.

## 1. Immutable v1 audit and frozen recovery

The original v1 audit remains unchanged under `results/phase16/`. Its 600 candidate rows were read-only inputs.

Before v2 candidate screening, the finalization process reconstructed and froze the fixed v1 selection into:

```text
results/phase16_t2_supplement_v2/v1_selected61_manifest.json
results/phase16_t2_supplement_v2/v1_selected61_manifest.sha256
```

Assertions passed:

| Check | Value |
|---|---:|
| v1 selected records | 61 |
| v1 F2 test T2 | 6 |
| v1 F3 test T2 | 9 |
| v1 selected structural fingerprints | 61 unique |

The finalization process reads this frozen manifest; it does not rerun or reinterpret v1 selection logic.

## 2. v2 configuration and exact-tick semantics

v2 configuration source:

```text
configs/phase16_t2_supplement_v2.yaml
```

The normalized runtime configuration, including generator/pattern/permutation/fingerprint versions, field hash format, field domains, stable pattern definitions, optional-task definitions, seed ranges, quotas, timeouts, and acceptance thresholds, is written to:

```text
results/phase16_t2_supplement_v2/screening_config.json
results/phase16_t2_supplement_v2/screening_config.sha256
```

v2 config hash:

```text
0c476eabf25f52c749934b9cf8bdde3fc950196696842200eb75dd27bfa950a0
```

F2 uses only slow multipliers `{2,4}` and compatibility coefficients `{0.5,0.25}`. All accepted F2 schedules retain exact integer tick semantics; no `1/3` or float approximation is used.

## 3. v2 candidate audit

v2 audit directory:

```text
results/phase16_t2_supplement_v2/
```

It contains the required `candidates.jsonl`, `selection_audit.jsonl`, config/hash, v1 frozen manifest/hash, progress/summary, and selected supplement manifest.

| Family | Seed range | Audited | Solver-level accepted | Selected new unique | Required |
|---|---:|---:|---:|---:|---:|
| F2 T2-v2 | 6000–6099 | 100 | 12 | 4 | 4 |
| F3 T2-v2 | 6100–6199 | 100 | 100 | 1 | 1 |

All 200 candidates were audited even after quotas were met.

F2 solver-level rejection reasons:

| Reason | Count |
|---|---:|
| `f2_greedy_gap_outside_range` | 55 |
| `f2_exact_optima_differ` | 54 |

F3 had no solver-level rejections. Selection audit separately records solver-level rejection, global/v1 duplicate rejection, quota exhaustion, and selected outcomes.

The shared global structural set started with all historical A1 fingerprints and all 61 v1 selected fingerprints. It was shared across F2-v2 and F3-v2 selection. Selected v2 fingerprints were verified not to collide with any historical A1 frozen instance.

### Selected v2 supplement

| Family | Seed | Template | Structural fingerprint |
|---|---:|---|---|
| F2 | 6001 | T2-v2 | `304a520fa02d04bcee1cbb4322c49447092f070623eda26e0f7ca421ffe8baa3` |
| F2 | 6036 | T2-v2 | `12bca9b50ca9f10837839ed3aec15d3fb69352dde8a4e3c986f0b7dbe84fc960` |
| F2 | 6041 | T2-v2 | `eb8454cdb7c738030e1f81f029ba216838019a043ddb73aabeaf83b4004b81ea` |
| F2 | 6048 | T2-v2 | `f5f422f5311f2aeeee3b82bddfe3d8698632df158c8c4ca31918d4f842b2a782` |
| F3 | 6100 | T2-v2 | `2454f30511dbc2b59c4af6f9c1ff33954972902bf25585b5806fe5b24b39ad60` |

Every candidate audit row records `generation_parameters`, including the field-specific task count, durations, demands, multiplier/compatibility, merge/side-work pattern, optional-task role/position, active-wait mechanism, and deterministic task/pool permutation.

## 4. Final 66-instance semantic recheck

Final output directory:

```text
results/phase16_final/
```

Artifacts:

```text
train_instances.json
validation_instances.json
test_instances.json
selected_manifest.json
oracle_audit.jsonl
final_generation_summary.json
```

Finalization checked:

| Check | Result |
|---|---:|
| Total selected instances | 66 |
| Global structural fingerprints | 66 unique |
| Train count | 24 |
| Validation count | 12 |
| Test count | 30 |
| Full semantic rechecks passed | 66 |

`oracle_audit.jsonl` contains complete MILP, Greedy, CA-HEFT, Validator, and—where required—both active-wait exhaustive proofs for every final selected instance.

Family-specific recheck rules were applied:

- **F1:** MILP proven optimal; MILP, Greedy, and CA-HEFT Validator-valid.
- **F2:** both exhaustive searches complete/optimal/valid; MILP = with-active-wait = without-active-wait; no-active-wait optimum witness; Greedy gap remains in `[5%,30%]` with absolute gap ≥2; all schedules valid.
- **F3:** both exhaustive searches complete/optimal/valid; MILP = with-active-wait; with-active-wait strictly better; optimum contains active wait; WaitBenefit ≥10%; GreedyGap ≥5%; all schedules valid.

## 5. Version-separated held-out T2 composition

The final test split explicitly preserves version provenance:

| Family | phase16-v1 T2 | phase16-v2 T2-v2 | Combined |
|---|---:|---:|---:|
| F2 | 6 | 4 | 10 |
| F3 | 9 | 1 | 10 |

Future Phase 1.6 evaluation must report F2/F3 test metrics separately for v1 T2, v2 T2-v2, and the combined result; it must not report only combined averages.

## 6. Stopping state

The Phase 1.6 data set is now frozen and semantically revalidated. No model training was run in this phase. The next action requires separate approval for the single-seed, fixed-small-set training diagnostic.
