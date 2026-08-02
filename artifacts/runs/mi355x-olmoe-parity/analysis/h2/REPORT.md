# H2 result: `mi355x-olmoe-parity`

**Decision:** PILOT_SUPPORT

At least one routing-only transition baseline passed the held-out prefill gate; proceed to a lightweight external predictor after human figure review.

## Held-out design

- Train requests: 12
- Test requests: 4
- Split unit: request, stratified by domain.
- Prefill and decode are evaluated separately.

## Domain-balanced prefill results at K=16

| Δ | Baseline | Selection coverage | Complete-token coverage | Candidate churn |
|---:|---|---:|---:|---:|
| 1 | domain_oracle | 44.7% | 1.9% | 0.0% |
| 1 | previous_window | 44.7% | 1.9% | 0.0% |
| 1 | static | 35.8% | 0.1% | 0.0% |
| 1 | transition | 75.3% | 20.9% | 43.3% |
| 2 | domain_oracle | 45.1% | 2.0% | 0.0% |
| 2 | previous_window | 45.1% | 2.0% | 0.0% |
| 2 | static | 35.8% | 0.1% | 0.0% |
| 2 | transition | 73.9% | 19.8% | 42.2% |
| 3 | domain_oracle | 45.5% | 2.1% | 0.0% |
| 3 | previous_window | 45.5% | 2.1% | 0.0% |
| 3 | static | 36.0% | 0.1% | 0.0% |
| 3 | transition | 72.5% | 17.8% | 41.5% |
| 4 | domain_oracle | 45.7% | 2.3% | 0.0% |
| 4 | previous_window | 45.7% | 2.3% | 0.0% |
| 4 | static | 35.9% | 0.1% | 0.0% |
| 4 | transition | 70.8% | 16.3% | 40.5% |
| 5 | domain_oracle | 45.6% | 2.3% | 0.0% |
| 5 | previous_window | 45.6% | 2.3% | 0.0% |
| 5 | static | 35.8% | 0.1% | 0.0% |
| 5 | transition | 69.5% | 15.3% | 40.0% |
| 6 | domain_oracle | 45.4% | 2.1% | 0.0% |
| 6 | previous_window | 45.4% | 2.1% | 0.0% |
| 6 | static | 35.7% | 0.1% | 0.0% |
| 6 | transition | 68.1% | 14.2% | 38.9% |
| 7 | domain_oracle | 44.4% | 2.2% | 0.0% |
| 7 | previous_window | 44.4% | 2.2% | 0.0% |
| 7 | static | 35.4% | 0.2% | 0.0% |
| 7 | transition | 67.3% | 12.8% | 39.0% |
| 8 | domain_oracle | 44.0% | 2.1% | 0.0% |
| 8 | previous_window | 44.0% | 2.1% | 0.0% |
| 8 | static | 35.3% | 0.2% | 0.0% |
| 8 | transition | 66.2% | 11.5% | 38.2% |
| 9 | domain_oracle | 43.5% | 2.1% | 0.0% |
| 9 | previous_window | 43.5% | 2.1% | 0.0% |
| 9 | static | 34.8% | 0.0% | 0.0% |
| 9 | transition | 64.0% | 9.1% | 36.9% |
| 10 | domain_oracle | 43.7% | 1.9% | 0.0% |
| 10 | previous_window | 43.7% | 1.9% | 0.0% |
| 10 | static | 34.4% | 0.0% | 0.0% |
| 10 | transition | 61.2% | 7.8% | 35.5% |
| 11 | domain_oracle | 44.3% | 2.1% | 0.0% |
| 11 | previous_window | 44.3% | 2.1% | 0.0% |
| 11 | static | 34.7% | 0.0% | 0.0% |
| 11 | transition | 59.6% | 7.0% | 33.3% |
| 12 | domain_oracle | 43.9% | 2.0% | 0.0% |
| 12 | previous_window | 43.9% | 2.0% | 0.0% |
| 12 | static | 34.1% | 0.0% | 0.0% |
| 12 | transition | 57.0% | 5.3% | 33.8% |
| 13 | domain_oracle | 43.4% | 1.5% | 0.0% |
| 13 | previous_window | 43.4% | 1.5% | 0.0% |
| 13 | static | 33.2% | 0.0% | 0.0% |
| 13 | transition | 54.5% | 5.2% | 32.2% |
| 14 | domain_oracle | 42.3% | 1.3% | 0.0% |
| 14 | previous_window | 42.3% | 1.3% | 0.0% |
| 14 | static | 32.0% | 0.0% | 0.0% |
| 14 | transition | 50.7% | 3.9% | 32.4% |
| 15 | domain_oracle | 43.3% | 1.4% | 0.0% |
| 15 | previous_window | 43.3% | 1.4% | 0.0% |
| 15 | static | 34.2% | 0.0% | 0.0% |
| 15 | transition | 48.9% | 2.0% | 28.7% |

## Preregistered gate

| Δ | Selection gain | Complete-token gain | Positive scopes | Positive domains | Pass |
|---:|---:|---:|---:|---:|:---:|
| 1 | +39.5 pp | +20.8 pp | 100.0% | 4 | yes |
| 2 | +38.1 pp | +19.7 pp | 100.0% | 4 | yes |
| 3 | +36.5 pp | +17.7 pp | 100.0% | 4 | yes |
| 4 | +34.9 pp | +16.2 pp | 100.0% | 4 | yes |
| 5 | +33.7 pp | +15.1 pp | 100.0% | 4 | yes |
| 6 | +32.4 pp | +14.0 pp | 100.0% | 4 | yes |
| 7 | +31.9 pp | +12.6 pp | 100.0% | 4 | yes |
| 8 | +30.9 pp | +11.4 pp | 100.0% | 4 | yes |
| 9 | +29.2 pp | +9.1 pp | 100.0% | 4 | yes |
| 10 | +26.7 pp | +7.8 pp | 100.0% | 4 | yes |
| 11 | +25.0 pp | +7.0 pp | 100.0% | 4 | yes |
| 12 | +22.9 pp | +5.3 pp | 100.0% | 4 | yes |
| 13 | +21.3 pp | +5.2 pp | 100.0% | 4 | yes |
| 14 | +18.7 pp | +3.9 pp | 100.0% | 4 | yes |
| 15 | +14.7 pp | +2.0 pp | 100.0% | 4 | no |

This pilot establishes routing information only. It is not a latency, transfer-feasibility, or cross-model result.
