# CommBudget-WeCAN V1 synthetic data sanity

## Scope and reproducibility

- Generator config: `configs/comm_budget_synthetic_v1.yaml`
- Dataset seed: `2026`
- Instances: 300, with 100 each of `layered`, `erdos_renyi`, and `stochastic_block`
- Tasks/pools: 30 tasks, 3 pools, 2 resource dimensions
- Global `d_size`: 96.0 in all 300 instances
- Generated sanity dataset: `_runs/comm_budget_v1_sanity/instances_300.json`
- Dataset SHA-256: `af86262c6429407a74c1c9232205ea700ef469e526e6b709ccd16626f026796d`
- Machine-readable statistics: `results/comm_budget_v1_sanity/summary.json`

Regenerate the same dataset and summary with:

```bash
PYTHONPATH=. python scripts/analyze_comm_budget_synthetic.py
```

The dataset is generated entirely from a deterministic RNG stream. No model output,
training result, reward, or scheduling heuristic is used to select its parameters.

## Adopted distributions

| Attribute | Distribution/range |
|---|---|
| Workload | clipped log-normal, median 24, log-sigma 0.45, range [8, 64] |
| CPU demand | categorical {1,2,4,6,8}, probabilities {0.15,0.25,0.30,0.20,0.10} |
| Memory demand | categorical {1,2,3,4,6}, same probabilities |
| Capacities | bases (12,12), (20,16), (36,24), independently jittered ±10% |
| Frequencies | economy [0.85,1.00], balanced [1.30,1.55], performance [1.90,2.25] |
| Powers | economy [0.80,1.00], balanced [1.30,1.65], performance [2.20,2.80] |
| Unit costs | economy [0.65,0.85], balanced [0.95,1.20], performance [1.35,1.75] |
| Cost_i | derived as unitCost_i × frequency_i / power_i |
| Directed bandwidth | base matrix below, independently jittered ±15% |
| Budget | `1.25 × C_min` |

Directed base bandwidth:

```text
          to economy  balanced  performance
economy       0          24          12
balanced     18           0          30
performance  14          22           0
```

This creates three explicit roles: economy is slowest and cheapest; balanced has
intermediate speed/cost and the strongest average network; performance is fastest,
highest-capacity, and most expensive, with a relatively weak economy link.

## 300-instance statistics

### Compute and communication

| Statistic | Execution time | Cross-pool communication time |
|---|---:|---:|
| minimum | 3.562 | 2.786 |
| p05 | 6.920 | 3.024 |
| median | 17.095 | 4.877 |
| p95 | 42.554 | 8.448 |
| maximum | 75.211 | 9.407 |

```text
median(communication) / median(compute) = 0.2853
```

Communication is therefore material but not globally dominant: its median is about
28.5% of median execution time, while the tails overlap enough for pool placement and
network direction to matter on individual dependencies.

### Pool heterogeneity

| Pool role | median frequency | median power | median Cost | median unit cost | median outgoing bw |
|---|---:|---:|---:|---:|---:|
| economy | 0.931 | 0.901 | 0.783 | 0.751 | 17.882 |
| balanced-network | 1.426 | 1.473 | 1.051 | 1.075 | 24.055 |
| performance | 2.078 | 2.495 | 1.291 | 1.564 | 17.946 |

Across instances, median max/min ratios are 2.234 for frequency, 2.822 for power,
2.056 for unit cost, and 2.492 for directed bandwidth. Frequency and unit cost are
strictly ordered across all three roles in 100% of instances.

### Budget tightness and feasibility

| Metric | Result |
|---|---:|
| median `C_min` | 587.203 |
| p05–p95 `C_min` | 478.393–716.034 |
| median Budget | 734.003 |
| median absolute slack | 146.801 |
| Budget / C_min | 1.25 in every instance |
| all-balanced assignment over budget | 90.0% of instances |
| all-performance assignment over budget | 100% of instances |
| median tasks upgradeable to balanced pool | 70.0% |
| median tasks upgradeable to performance pool | 36.7% |
| random pool-assignment budget-feasible rate | 1.342% of 76,800 samples |
| structurally invalid instances | 0 / 300 |
| budget-infeasible instances | 0 / 300 |

For every instance, a sequential schedule using each task's minimum-cost pool was
constructed and independently validated. This proves that the generated budget and
capacity constraints admit at least one schedule. At the same time, the low random
assignment feasibility rate and the all-performance rejection rate show that Budget is
not merely decorative. The upgrade fractions show that it still permits substantial,
but selective, use of faster pools.

## Assessment

The dataset is suitable for beginning a small controlled CommBudget-WeCAN V1 training
smoke test:

- communication is neither negligible nor the dominant global scale;
- all instances have a validated feasible schedule;
- budget is binding for indiscriminate fast-pool allocation but leaves useful choice;
- speed, unit cost, capacity, and directed-network properties provide explicit trade-offs;
- all topology families are balanced and reproducible.

The 1.342% random-assignment feasibility rate means an untrained unconstrained policy
would frequently propose expensive combinations. This is expected for a hard-budget
decoder and should be monitored in later training diagnostics, but it is not instance
infeasibility and is not a reason to change alpha before the first controlled experiment.
