# H2 result: `h1-standard-small`

**Decision:** PILOT_SUPPORT

At least one routing-only transition baseline passed the held-out decode gate; proceed to a lightweight external predictor after human figure review.

## Held-out design

- Train requests: 96
- Test requests: 32
- Split unit: request, stratified by domain.
- Prefill and decode are evaluated separately.

## Domain-balanced decode results at K=16

| Δ | Baseline | Selection coverage | Complete-token coverage | Candidate churn |
|---:|---|---:|---:|---:|
| 1 | domain_oracle | 49.3% | 3.0% | 0.0% |
| 1 | previous_window | 48.6% | 3.0% | 0.1% |
| 1 | static | 41.0% | 0.6% | 0.0% |
| 1 | transition | 79.0% | 24.1% | 42.8% |
| 2 | domain_oracle | 49.9% | 3.1% | 0.0% |
| 2 | previous_window | 49.2% | 3.2% | 0.1% |
| 2 | static | 41.3% | 0.6% | 0.0% |
| 2 | transition | 77.9% | 23.5% | 41.4% |
| 3 | domain_oracle | 50.5% | 3.4% | 0.0% |
| 3 | previous_window | 49.8% | 3.4% | 0.1% |
| 3 | static | 41.7% | 0.7% | 0.0% |
| 3 | transition | 76.8% | 22.2% | 40.4% |

## Preregistered gate

| Δ | Selection gain | Complete-token gain | Positive scopes | Positive domains | Pass |
|---:|---:|---:|---:|---:|:---:|
| 1 | +38.0 pp | +23.5 pp | 100.0% | 4 | yes |
| 2 | +36.6 pp | +22.9 pp | 100.0% | 4 | yes |
| 3 | +35.1 pp | +21.5 pp | 100.0% | 4 | yes |

This pilot establishes routing information only. It is not a latency, transfer-feasibility, or cross-model result.
