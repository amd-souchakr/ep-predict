# H3 result: `h3-standard-small`

**Decision:** PILOT_DOES_NOT_SUPPORT

The fixed linear sidecar does not materially beat the transition table on the primary gate; use the simpler transition policy in H4 and stop learned-predictor work for this checkpoint.

## Domain-balanced decode results at K=16

| Δ | Policy | Selection coverage | Complete-token coverage | Candidate churn |
|---:|---|---:|---:|---:|
| 1 | domain_oracle | 49.3% | 3.0% | 0.0% |
| 1 | linear | 79.4% | 28.7% | 52.8% |
| 1 | static | 41.0% | 0.6% | 0.0% |
| 1 | transition | 79.0% | 24.1% | 42.8% |
| 2 | domain_oracle | 49.9% | 3.1% | 0.0% |
| 2 | linear | 79.3% | 28.8% | 52.6% |
| 2 | static | 41.3% | 0.6% | 0.0% |
| 2 | transition | 77.9% | 23.5% | 41.4% |
| 3 | domain_oracle | 50.5% | 3.4% | 0.0% |
| 3 | linear | 79.3% | 28.8% | 52.2% |
| 3 | static | 41.7% | 0.7% | 0.0% |
| 3 | transition | 76.8% | 22.2% | 40.4% |
| 4 | domain_oracle | 50.8% | 3.6% | 0.0% |
| 4 | linear | 78.7% | 28.6% | 51.9% |
| 4 | static | 41.6% | 0.7% | 0.0% |
| 4 | transition | 75.5% | 21.2% | 40.2% |
| 5 | domain_oracle | 51.0% | 3.8% | 0.0% |
| 5 | linear | 78.4% | 28.4% | 52.1% |
| 5 | static | 41.5% | 0.6% | 0.0% |
| 5 | transition | 74.1% | 19.9% | 39.7% |
| 6 | domain_oracle | 50.9% | 3.8% | 0.0% |
| 6 | linear | 77.8% | 27.9% | 52.3% |
| 6 | static | 41.3% | 0.6% | 0.0% |
| 6 | transition | 72.9% | 18.8% | 38.7% |
| 7 | domain_oracle | 50.2% | 3.9% | 0.0% |
| 7 | linear | 77.1% | 27.4% | 52.8% |
| 7 | static | 40.6% | 0.6% | 0.0% |
| 7 | transition | 72.0% | 17.8% | 38.8% |
| 8 | domain_oracle | 49.8% | 4.0% | 0.0% |
| 8 | linear | 76.5% | 27.4% | 53.4% |
| 8 | static | 39.9% | 0.4% | 0.0% |
| 8 | transition | 70.7% | 15.9% | 38.7% |
| 9 | domain_oracle | 49.4% | 3.9% | 0.0% |
| 9 | linear | 75.6% | 26.5% | 54.0% |
| 9 | static | 39.4% | 0.5% | 0.0% |
| 9 | transition | 68.7% | 14.1% | 38.4% |
| 10 | domain_oracle | 49.3% | 4.0% | 0.0% |
| 10 | linear | 74.4% | 25.1% | 54.7% |
| 10 | static | 38.9% | 0.6% | 0.0% |
| 10 | transition | 66.7% | 12.0% | 37.7% |
| 11 | domain_oracle | 49.5% | 4.2% | 0.0% |
| 11 | linear | 73.1% | 22.6% | 54.8% |
| 11 | static | 38.9% | 0.7% | 0.0% |
| 11 | transition | 64.7% | 10.5% | 37.2% |
| 12 | domain_oracle | 49.5% | 4.5% | 0.0% |
| 12 | linear | 72.0% | 21.9% | 55.3% |
| 12 | static | 38.2% | 0.8% | 0.0% |
| 12 | transition | 62.8% | 9.9% | 36.8% |
| 13 | domain_oracle | 48.9% | 3.7% | 0.0% |
| 13 | linear | 69.8% | 19.1% | 56.5% |
| 13 | static | 37.6% | 0.7% | 0.0% |
| 13 | transition | 60.2% | 7.9% | 35.5% |
| 14 | domain_oracle | 48.2% | 3.2% | 0.0% |
| 14 | linear | 70.4% | 20.4% | 58.4% |
| 14 | static | 37.0% | 0.3% | 0.0% |
| 14 | transition | 57.6% | 6.4% | 37.9% |
| 15 | domain_oracle | 48.2% | 1.9% | 0.0% |
| 15 | linear | 69.2% | 19.7% | 58.7% |
| 15 | static | 37.4% | 0.1% | 0.0% |
| 15 | transition | 53.8% | 4.6% | 34.0% |

## Preregistered primary gate

- Selection gain: +0.4 pp.
- Complete-token gain: +4.7 pp.
- Positive selection scopes: 56.7%.
- Positive complete-token scopes: 75.0%.
- Domains positive on both metrics: 2/4.

## Integrity

- Feature/route alignment: pass.
- Feature rows: 377,488.
- H2 transition scopes reproduced: 1008.
- Maximum H2 coverage difference: 0.

This is a single-checkpoint pilot. It does not establish physical transfer feasibility, latency improvement, or universal MoE behavior.
