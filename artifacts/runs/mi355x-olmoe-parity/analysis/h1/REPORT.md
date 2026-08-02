# H1 result: `mi355x-olmoe-parity`

**Decision:** PILOT_DOES_NOT_SUPPORT

0 of 16 eligible prefill layers passed both the skew and stability thresholds (required fraction: 0.50).

## Integrity

- Records: 26752
- Requests: 16
- Observed layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
- Routing top-k values: [8]
- Schema versions: [1]

## Headline per-layer skew

| Layer | Gini | Top-8 coverage | Lift over uniform |
|---:|---:|---:|---:|
| 0 | 0.201 | 23.3% | 1.86× |
| 1 | 0.204 | 20.9% | 1.67× |
| 2 | 0.194 | 19.9% | 1.59× |
| 3 | 0.240 | 21.3% | 1.70× |
| 4 | 0.260 | 23.8% | 1.90× |
| 5 | 0.261 | 22.4% | 1.79× |
| 6 | 0.293 | 24.7% | 1.97× |
| 7 | 0.265 | 23.7% | 1.90× |
| 8 | 0.272 | 22.5% | 1.80× |
| 9 | 0.231 | 22.2% | 1.78× |
| 10 | 0.195 | 20.6% | 1.65× |
| 11 | 0.259 | 23.9% | 1.91× |
| 12 | 0.246 | 23.7% | 1.90× |
| 13 | 0.237 | 21.9% | 1.75× |
| 14 | 0.217 | 20.4% | 1.64× |
| 15 | 0.253 | 22.8% | 1.82× |

## Interpretation

The configured hot tier does not pass the model-wide pilot gate. Inspect domain-specific rows before deciding whether H1 is locally mixed or should be rejected for this testbed.

This is a workload-characterization result only. Hooked inference is not a latency measurement, and H1 does not establish that experts can be prefetched in time.
