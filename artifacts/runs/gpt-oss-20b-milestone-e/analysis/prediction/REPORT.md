# GPT-OSS 20B Milestone E result

**Decision:** `PILOT_SUPPORTS_20B_ROUTE_PREDICTION`

The request-held-out split contains 96 training and 32 test requests. The primary point is decode K=8 (2× top-4 candidate amplification; 25% of experts). Transition is compared with the stronger of domain popularity and current-route copy separately for each metric.

| Δ | Selection (transition / comparator / gain, 95% CI) | Routed mass gain | Complete-route gain (95% CI) | Positive domains | Pass |
|---:|---:|---:|---:|---:|:---:|
| 1 | 86.3% / 68.1% / +18.2 pp [+17.4, +19.0] | +17.7 pp | +32.3 pp [+31.3, +33.1] | 4/4 | yes |
| 2 | 85.5% / 68.8% / +16.7 pp [+16.0, +17.6] | +16.5 pp | +30.0 pp [+28.9, +31.0] | 4/4 | yes |
| 3 | 84.4% / 68.8% / +15.5 pp [+14.8, +16.4] | +15.5 pp | +26.9 pp [+26.1, +27.6] | 4/4 | yes |

3/2 required short-horizon points passed.

This is route-set prediction evidence from one checkpoint and one workload. It is not a language-quality score, latency result, or substitute for the cancelled 120B comparison.
