# H2 result: `olmoe-base-c0-paired`

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
| 1 | domain_oracle | 46.8% | 3.5% | 0.0% |
| 1 | previous_window | 45.5% | 2.5% | 0.2% |
| 1 | static | 38.0% | 0.5% | 0.0% |
| 1 | transition | 77.2% | 22.2% | 41.9% |
| 2 | domain_oracle | 47.2% | 3.7% | 0.0% |
| 2 | previous_window | 45.9% | 2.7% | 0.2% |
| 2 | static | 38.2% | 0.6% | 0.0% |
| 2 | transition | 75.7% | 20.8% | 40.8% |
| 3 | domain_oracle | 47.7% | 4.0% | 0.0% |
| 3 | previous_window | 46.4% | 2.8% | 0.2% |
| 3 | static | 38.6% | 0.6% | 0.0% |
| 3 | transition | 74.9% | 20.1% | 40.0% |
| 4 | domain_oracle | 48.0% | 4.3% | 0.0% |
| 4 | previous_window | 46.7% | 3.1% | 0.2% |
| 4 | static | 38.6% | 0.6% | 0.0% |
| 4 | transition | 73.1% | 18.4% | 38.8% |
| 5 | domain_oracle | 48.1% | 4.5% | 0.0% |
| 5 | previous_window | 46.7% | 3.2% | 0.2% |
| 5 | static | 38.5% | 0.7% | 0.0% |
| 5 | transition | 72.0% | 17.6% | 38.3% |
| 6 | domain_oracle | 48.0% | 4.6% | 0.0% |
| 6 | previous_window | 46.7% | 3.3% | 0.2% |
| 6 | static | 38.5% | 0.7% | 0.0% |
| 6 | transition | 70.9% | 17.4% | 37.3% |
| 7 | domain_oracle | 47.7% | 4.8% | 0.0% |
| 7 | previous_window | 46.3% | 3.4% | 0.1% |
| 7 | static | 38.1% | 0.6% | 0.0% |
| 7 | transition | 69.8% | 16.2% | 37.1% |
| 8 | domain_oracle | 47.4% | 4.9% | 0.0% |
| 8 | previous_window | 46.2% | 3.4% | 0.1% |
| 8 | static | 37.9% | 0.7% | 0.0% |
| 8 | transition | 68.6% | 14.5% | 36.5% |
| 9 | domain_oracle | 47.5% | 5.1% | 0.0% |
| 9 | previous_window | 46.2% | 3.5% | 0.2% |
| 9 | static | 37.7% | 0.8% | 0.0% |
| 9 | transition | 66.1% | 12.7% | 35.2% |
| 10 | domain_oracle | 48.0% | 5.2% | 0.0% |
| 10 | previous_window | 46.6% | 3.5% | 0.2% |
| 10 | static | 38.0% | 0.8% | 0.0% |
| 10 | transition | 63.5% | 10.9% | 33.0% |
| 11 | domain_oracle | 48.7% | 5.6% | 0.0% |
| 11 | previous_window | 47.3% | 3.7% | 0.2% |
| 11 | static | 38.5% | 0.9% | 0.0% |
| 11 | transition | 61.9% | 10.3% | 31.5% |
| 12 | domain_oracle | 48.9% | 5.8% | 0.0% |
| 12 | previous_window | 47.5% | 4.0% | 0.2% |
| 12 | static | 38.5% | 1.1% | 0.0% |
| 12 | transition | 59.6% | 9.2% | 30.8% |
| 13 | domain_oracle | 48.7% | 5.4% | 0.0% |
| 13 | previous_window | 47.3% | 3.5% | 0.2% |
| 13 | static | 38.8% | 1.1% | 0.0% |
| 13 | transition | 56.8% | 8.2% | 28.6% |
| 14 | domain_oracle | 48.3% | 5.1% | 0.0% |
| 14 | previous_window | 46.7% | 3.7% | 0.2% |
| 14 | static | 38.4% | 0.3% | 0.0% |
| 14 | transition | 54.2% | 6.8% | 29.2% |
| 15 | domain_oracle | 50.2% | 5.6% | 0.0% |
| 15 | previous_window | 47.3% | 3.9% | 0.2% |
| 15 | static | 39.8% | 0.5% | 0.0% |
| 15 | transition | 50.7% | 5.2% | 23.6% |

## Preregistered gate

| Δ | Selection gain | Complete-token gain | Positive scopes | Positive domains | Pass |
|---:|---:|---:|---:|---:|:---:|
| 1 | +39.2 pp | +21.6 pp | 100.0% | 4 | yes |
| 2 | +37.5 pp | +20.2 pp | 100.0% | 4 | yes |
| 3 | +36.2 pp | +19.6 pp | 100.0% | 4 | yes |
| 4 | +34.5 pp | +17.7 pp | 100.0% | 4 | yes |
| 5 | +33.5 pp | +17.0 pp | 100.0% | 4 | yes |
| 6 | +32.3 pp | +16.7 pp | 100.0% | 4 | yes |
| 7 | +31.7 pp | +15.5 pp | 100.0% | 4 | yes |
| 8 | +30.7 pp | +13.8 pp | 100.0% | 4 | yes |
| 9 | +28.4 pp | +11.9 pp | 100.0% | 4 | yes |
| 10 | +25.5 pp | +10.1 pp | 100.0% | 4 | yes |
| 11 | +23.4 pp | +9.4 pp | 100.0% | 4 | yes |
| 12 | +21.1 pp | +8.1 pp | 100.0% | 4 | yes |
| 13 | +18.0 pp | +7.1 pp | 100.0% | 4 | yes |
| 14 | +15.9 pp | +6.5 pp | 100.0% | 4 | yes |
| 15 | +11.0 pp | +4.7 pp | 100.0% | 4 | yes |

This pilot establishes routing information only. It is not a latency, transfer-feasibility, or cross-model result.
