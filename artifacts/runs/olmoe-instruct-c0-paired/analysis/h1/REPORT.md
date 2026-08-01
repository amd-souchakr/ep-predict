# H1 result: `olmoe-instruct-c0-paired`

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
| 0 | 0.202 | 23.7% | 1.89× |
| 1 | 0.231 | 23.7% | 1.90× |
| 2 | 0.221 | 23.0% | 1.84× |
| 3 | 0.260 | 23.8% | 1.90× |
| 4 | 0.288 | 26.4% | 2.11× |
| 5 | 0.282 | 26.1% | 2.09× |
| 6 | 0.327 | 29.4% | 2.35× |
| 7 | 0.304 | 26.1% | 2.09× |
| 8 | 0.296 | 26.0% | 2.08× |
| 9 | 0.253 | 24.8% | 1.98× |
| 10 | 0.228 | 23.0% | 1.84× |
| 11 | 0.284 | 27.5% | 2.20× |
| 12 | 0.279 | 25.5% | 2.04× |
| 13 | 0.275 | 25.3% | 2.03× |
| 14 | 0.248 | 23.4% | 1.87× |
| 15 | 0.254 | 25.1% | 2.01× |

## Interpretation

The configured hot tier does not pass the model-wide pilot gate. Inspect domain-specific rows before deciding whether H1 is locally mixed or should be rejected for this testbed.

This is a workload-characterization result only. Hooked inference is not a latency measurement, and H1 does not establish that experts can be prefetched in time.
