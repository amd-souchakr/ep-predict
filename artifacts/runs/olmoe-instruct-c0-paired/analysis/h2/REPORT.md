# H2 result: `olmoe-instruct-c0-paired`

**Decision:** PILOT_SUPPORT

At least one routing-only transition baseline passed the held-out prefill gate; proceed to a lightweight external predictor after human figure review.

## Held-out design

- Train requests: 96
- Test requests: 32
- Split unit: request, stratified by domain.
- Prefill and decode are evaluated separately.

## Domain-balanced prefill results at K=16

| Δ | Baseline | Selection coverage | Complete-token coverage | Candidate churn |
|---:|---|---:|---:|---:|
| 1 | domain_oracle | 47.0% | 3.7% | 0.0% |
| 1 | previous_window | 45.6% | 2.5% | 0.2% |
| 1 | static | 38.5% | 0.9% | 0.0% |
| 1 | transition | 77.8% | 23.2% | 41.9% |
| 2 | domain_oracle | 47.4% | 3.9% | 0.0% |
| 2 | previous_window | 46.0% | 2.7% | 0.2% |
| 2 | static | 38.6% | 0.9% | 0.0% |
| 2 | transition | 76.2% | 22.2% | 40.8% |
| 3 | domain_oracle | 47.8% | 4.1% | 0.0% |
| 3 | previous_window | 46.4% | 2.9% | 0.2% |
| 3 | static | 39.0% | 1.0% | 0.0% |
| 3 | transition | 75.1% | 20.4% | 39.9% |
| 4 | domain_oracle | 48.0% | 4.5% | 0.0% |
| 4 | previous_window | 46.6% | 3.1% | 0.2% |
| 4 | static | 38.9% | 1.1% | 0.0% |
| 4 | transition | 73.5% | 19.2% | 39.1% |
| 5 | domain_oracle | 48.0% | 4.7% | 0.0% |
| 5 | previous_window | 46.4% | 3.2% | 0.2% |
| 5 | static | 38.7% | 1.1% | 0.0% |
| 5 | transition | 72.3% | 18.6% | 38.5% |
| 6 | domain_oracle | 47.8% | 4.8% | 0.0% |
| 6 | previous_window | 46.4% | 3.3% | 0.1% |
| 6 | static | 38.8% | 1.2% | 0.0% |
| 6 | transition | 71.0% | 17.9% | 37.3% |
| 7 | domain_oracle | 47.4% | 5.0% | 0.0% |
| 7 | previous_window | 46.1% | 3.4% | 0.1% |
| 7 | static | 38.3% | 1.0% | 0.0% |
| 7 | transition | 69.7% | 17.0% | 37.0% |
| 8 | domain_oracle | 47.1% | 5.0% | 0.0% |
| 8 | previous_window | 45.9% | 3.4% | 0.1% |
| 8 | static | 38.1% | 1.0% | 0.0% |
| 8 | transition | 68.7% | 15.0% | 36.3% |
| 9 | domain_oracle | 47.0% | 5.2% | 0.0% |
| 9 | previous_window | 45.8% | 3.5% | 0.1% |
| 9 | static | 37.7% | 1.0% | 0.0% |
| 9 | transition | 66.4% | 13.9% | 35.4% |
| 10 | domain_oracle | 47.5% | 5.3% | 0.0% |
| 10 | previous_window | 46.2% | 3.6% | 0.1% |
| 10 | static | 37.9% | 1.0% | 0.0% |
| 10 | transition | 63.7% | 11.5% | 33.4% |
| 11 | domain_oracle | 48.2% | 5.7% | 0.0% |
| 11 | previous_window | 46.8% | 3.8% | 0.1% |
| 11 | static | 38.3% | 1.2% | 0.0% |
| 11 | transition | 62.0% | 11.2% | 31.9% |
| 12 | domain_oracle | 48.3% | 5.8% | 0.0% |
| 12 | previous_window | 46.9% | 3.8% | 0.2% |
| 12 | static | 38.1% | 1.0% | 0.0% |
| 12 | transition | 59.5% | 9.7% | 30.8% |
| 13 | domain_oracle | 48.0% | 5.5% | 0.0% |
| 13 | previous_window | 46.6% | 3.4% | 0.2% |
| 13 | static | 38.3% | 0.8% | 0.0% |
| 13 | transition | 57.2% | 8.3% | 29.1% |
| 14 | domain_oracle | 47.3% | 5.0% | 0.0% |
| 14 | previous_window | 45.9% | 3.6% | 0.2% |
| 14 | static | 37.6% | 0.1% | 0.0% |
| 14 | transition | 54.6% | 6.6% | 31.1% |
| 15 | domain_oracle | 48.0% | 3.2% | 0.0% |
| 15 | previous_window | 45.6% | 2.9% | 0.2% |
| 15 | static | 38.4% | 0.2% | 0.0% |
| 15 | transition | 50.9% | 4.9% | 27.9% |

## Preregistered gate

| Δ | Selection gain | Complete-token gain | Positive scopes | Positive domains | Pass |
|---:|---:|---:|---:|---:|:---:|
| 1 | +39.3 pp | +22.4 pp | 100.0% | 4 | yes |
| 2 | +37.6 pp | +21.3 pp | 100.0% | 4 | yes |
| 3 | +36.1 pp | +19.4 pp | 100.0% | 4 | yes |
| 4 | +34.7 pp | +18.2 pp | 100.0% | 4 | yes |
| 5 | +33.5 pp | +17.4 pp | 100.0% | 4 | yes |
| 6 | +32.2 pp | +16.7 pp | 100.0% | 4 | yes |
| 7 | +31.5 pp | +16.0 pp | 100.0% | 4 | yes |
| 8 | +30.6 pp | +14.0 pp | 100.0% | 4 | yes |
| 9 | +28.7 pp | +12.9 pp | 100.0% | 4 | yes |
| 10 | +25.8 pp | +10.5 pp | 100.0% | 4 | yes |
| 11 | +23.7 pp | +10.0 pp | 100.0% | 4 | yes |
| 12 | +21.4 pp | +8.6 pp | 100.0% | 4 | yes |
| 13 | +18.9 pp | +7.5 pp | 100.0% | 4 | yes |
| 14 | +16.9 pp | +6.5 pp | 100.0% | 4 | yes |
| 15 | +12.6 pp | +4.6 pp | 100.0% | 4 | yes |

This pilot establishes routing information only. It is not a latency, transfer-feasibility, or cross-model result.
