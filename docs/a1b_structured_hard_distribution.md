# Proposed A1-b Structured Hard Distribution (Not Yet Executed)

## Status

This is a pre-registration proposal only. No candidate instance has been generated, screened, solved, or trained under this distribution.

## Suggested seed protocol

- **Primary approved-to-screen range (pending confirmation):** `4000..4099` — exactly 100 candidates.
- **Reserved contingency range:** `4100..4199` — not to be used unless separately approved after inspecting the first range's complete audit.
- The screening process must never silently expand from the primary to contingency range.

## Structural generator

All values are exact integer ticks. Instances have 8–10 tasks, two resource dimensions, and two heterogeneous pools.

| Element | Fast/tight pool 0 | Slow/loose pool 1 |
|---|---:|---:|
| Capacity | `(7, 7)` | `(14, 14)` |
| Compatible task duration | 1–3 ticks | 2–3× pool-0 duration |
| Intended role | critical-path acceleration, but scarce | slow overflow and branch overlap |

The DAG contains a 4–5 task critical spine and 3–5 branches released around the early/middle spine. Critical and branch work compete for the fast pool. A task compatible with fast pool requires 55–85% of a key fast capacity dimension, so at least one competing pair cannot co-run there. The slow pool remains capacity-feasible but creates a duration penalty.

Compatibility is task-specific: target 35–45% single-pool tasks, the remainder two-pool tasks. Critical tasks preferentially benefit from pool 0, while branches mix pool-0 preference, pool-1 preference, and sparse compatibility.

## Candidate acceptance rules

Every candidate must satisfy:

1. MILP status `optimal` and independent Validator pass;
2. external Greedy relative gap in `[5%, 30%]`;
3. absolute external Greedy gap at least 2 ticks;
4. all external schedule validations pass;
5. active waiting is not responsible for the target gap: on the separately generated 6–7 task structural-validation subfamily, both complete enumerations must satisfy `with_wait optimum == without_wait optimum`.

The 8–10 task A1-b candidates use MILP only; exhaustive search remains limited to the 6–7 task structural-validation subfamily.

## Expected budget

The prior easy distribution required 18.3 CPU solver-seconds for 100 instances (mean 0.183 seconds). This structured distribution is expected to require roughly 30–90 seconds for its 100 MILP candidates. Use a 5-minute outer budget and report every actual elapsed time. The 20-fixture 6–7 task no-active-wait validation subfamily is budgeted at under 2 minutes.

## Required tests before screening

1. Seed determinism and exact tick values.
2. Pool-0 / pool-1 capacity and duration heterogeneity invariants.
3. Critical-spine and branch tasks contend for pool 0.
4. Compatibility count and fast-pool demand-ratio targets.
5. No-active-wait equality for every 6–7 task validation fixture.
6. MILP/Validator result persistence, rejection reasons, and first-valid-seed ordering.
7. No screening call occurs from configuration/unit-test code.
