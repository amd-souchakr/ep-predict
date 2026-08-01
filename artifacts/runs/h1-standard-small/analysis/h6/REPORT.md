# H6 result: prediction-guided residency

**Decision:** PILOT_DOES_NOT_SUPPORT

Neither frozen prediction-guided residency policy materially beats the strongest simple placement baseline at equal capacity and movement budget. Routing is predictable, but this pilot does not establish value for the tested residency mechanism.

## Frozen primary scope

- Phase: decode
- Capacity: 16 experts per layer
- Lookahead: Δ=3
- Runtime movement cap: 1 expert per wave
- Prediction can admit only an actually demanded miss; there is no candidate-only prefetch.

| Policy | Residual cold demand | Complete resident-set hits | Useful / wasted MiB per wave | Churn | Oracle recovery (expert / wave) |
|---|---:|---:|---:|---:|---:|
| domain | 49.5% | 3.4% | 0.00 / 0.00 | 0.00% | -6.2% / 0.3% |
| linear | 48.8% | 3.5% | 7.33 / 3.71 | 5.75% | -3.7% / 6.2% |
| lru | 48.1% | 2.8% | 8.14 / 3.53 | 6.07% | 0.0% / 0.0% |
| oracle | 31.2% | 11.8% | 8.94 / 0.49 | 4.91% | 100.0% / 100.0% |
| static | 58.3% | 0.7% | 0.00 / 0.00 | 0.00% | -60.6% / -20.3% |
| transition | 50.2% | 3.4% | 5.73 / 3.80 | 4.97% | -11.1% / 4.0% |

## Frozen gate

| Guided policy | Expert-stall gain | Complete-set gain | Positive scopes | Domains | Layers | Pass |
|---|---:|---:|---:|---:|---:|:---:|
| transition | -3.9 pp | -0.7 pp | 9.6% | 0 | 0 | no |
| linear | -2.5 pp | -0.6 pp | 5.8% | 0 | 0 | no |

## Scope and limitations

- Results use the existing 32-request development split and are pilot evidence, not fresh confirmation.
- Domains are replayed independently to expose within-domain placement value; domain-switch reconfiguration cost is excluded.
- Useful movement means a newly resident copy serves a later hit before eviction; all unresolved insertions are charged as wasted.
- Stall reduction is first-order cold-work/wave elimination, not measured end-to-end latency.
- The broad phase/layer/horizon/capacity scan is descriptive and does not alter the primary gate.
