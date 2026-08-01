# H1 result: `h1-standard-small`

**Decision:** PILOT_DOES_NOT_SUPPORT

2 of 16 eligible decode layers passed both the skew and stability thresholds (required fraction: 0.50).

## Integrity

- Records: 377488
- Requests: 128
- Observed layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
- Routing top-k values: [8]
- Schema versions: [1]

## Headline per-layer skew

| Layer | Gini | Top-8 coverage | Lift over uniform |
|---:|---:|---:|---:|
| 0 | 0.186 | 22.4% | 1.79× |
| 1 | 0.183 | 21.9% | 1.75× |
| 2 | 0.180 | 20.4% | 1.63× |
| 3 | 0.260 | 23.7% | 1.89× |
| 4 | 0.266 | 26.3% | 2.10× |
| 5 | 0.271 | 27.1% | 2.17× |
| 6 | 0.327 | 29.9% | 2.39× |
| 7 | 0.312 | 27.5% | 2.20× |
| 8 | 0.287 | 25.6% | 2.05× |
| 9 | 0.272 | 26.7% | 2.14× |
| 10 | 0.224 | 23.0% | 1.84× |
| 11 | 0.265 | 25.6% | 2.05× |
| 12 | 0.242 | 24.0% | 1.92× |
| 13 | 0.226 | 22.5% | 1.80× |
| 14 | 0.200 | 20.4% | 1.63× |
| 15 | 0.199 | 20.9% | 1.67× |

## Interpretation

The configured hot tier does not pass the model-wide pilot gate. Inspect domain-specific rows before deciding whether H1 is locally mixed or should be rejected for this testbed.

This is a workload-characterization result only. Hooked inference is not a latency measurement, and H1 does not establish that experts can be prefetched in time.
