# H4 oracle feasibility result

**Decision:** `PILOT_DOES_NOT_SUPPORT`

## Calibration

- Hook-free cached-token forward median: 10.229 ms.
- Effective inter-MoE-layer budget: 0.639 ms.
- Exact 12 MiB pinned-host transfer median: 0.524 ms.
- Fitted effective bandwidth: 24.14 GB/s.

## Frozen primary gate

| Δ | Deadline-feasible cold bytes | Oracle stall reduction |
|---:|---:|---:|
| 1 | 26.6% | 31.8% |
| 2 | 29.8% | 35.8% |
| 3 | 32.8% | 38.9% |

The gate requires both metrics to reach 50% for at least one short horizon at measured bandwidth and K=16.

## Descriptive feasibility boundary

| K | Δ | Bandwidth | Resident hits | On-time cold bytes | Stall reduction |
|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 1× | 40.0% | 24.1% | 29.3% |
| 16 | 3 | 1× | 56.8% | 32.8% | 38.9% |
| 32 | 3 | 1× | 79.6% | 55.5% | 61.8% |
| 16 | 9 | 1× | 53.4% | 58.7% | 61.8% |
| 16 | 1 | 2× | 55.5% | 50.0% | 58.1% |

The frozen compact-tier target fails, but the broader scan is not a universal physical impossibility: K=32 at Δ=3, K=16 at Δ=9, and K=16 with 2× measured bandwidth expose feasible oracle regions. These cells are descriptive and do not change the formal gate or trigger predictor replay.

## Interpretation boundary

This is a trace-driven, single-copy-engine oracle calculation. It establishes a calibrated feasibility region, not end-to-end speedup or overlap correctness in the live model.
