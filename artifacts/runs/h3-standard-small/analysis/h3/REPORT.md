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
