# H1 result: `olmoe-base-c0-paired`

**Decision:** PILOT_DOES_NOT_SUPPORT

0 of 16 eligible prefill layers passed both the skew and stability thresholds (required fraction: 0.50).

## Integrity

- Records: 222688
- Requests: 128
- Observed layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
- Routing top-k values: [8]
- Schema versions: [1]

## Headline per-layer skew

| Layer | Gini | Top-8 coverage | Lift over uniform |
|---:|---:|---:|---:|
| 0 | 0.222 | 25.1% | 2.00× |
| 1 | 0.219 | 23.3% | 1.86× |
| 2 | 0.200 | 21.6% | 1.73× |
| 3 | 0.236 | 23.1% | 1.84× |
| 4 | 0.266 | 25.6% | 2.05× |
| 5 | 0.274 | 25.8% | 2.07× |
| 6 | 0.318 | 28.2% | 2.26× |
| 7 | 0.292 | 26.0% | 2.08× |
| 8 | 0.282 | 25.5% | 2.04× |
| 9 | 0.244 | 24.4% | 1.95× |
| 10 | 0.224 | 22.5% | 1.80× |
| 11 | 0.276 | 26.9% | 2.15× |
| 12 | 0.267 | 25.1% | 2.01× |
| 13 | 0.265 | 24.9% | 1.99× |
| 14 | 0.244 | 22.9% | 1.83× |
| 15 | 0.275 | 25.8% | 2.06× |

## Interpretation

The configured hot tier does not pass the model-wide pilot gate. Inspect domain-specific rows before deciding whether H1 is locally mixed or should be rejected for this testbed.

This is a workload-characterization result only. Hooked inference is not a latency measurement, and H1 does not establish that experts can be prefetched in time.
