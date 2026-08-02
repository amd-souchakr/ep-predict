# GPT-OSS 20B Milestone E result

**Overall decision:** `CONDITIONAL_PILOT_SUPPORT_WITH_TRACE_WEIGHT_EXCEPTION`

The collection retained 128 requests, 22,152 prompt tokens, and 2,048 decode tokens, yielding 580,800 complete token-layer records and 2,323,200 consumed ID/weight pairs. All executed expert IDs matched. The frozen trace gate formally failed: 6 independently reconstructed weights differed (0.000258% of pairs; maximum absolute error 0.001953125). The analysis below is explicitly post-hoc conditional evidence using the exact dispatch-consumed IDs and weights retained in the trace.

At the preregistered decode K=8 point, transition prediction is compared with the stronger of domain popularity and current-route copy:

| Δ | Selection coverage gain (95% CI) | Routed-mass gain | Complete-route gain (95% CI) | Pass |
|---:|---:|---:|---:|:---:|
| 1 | +18.2 pp [+17.4, +19.0] | +17.7 pp | +32.3 pp [+31.3, +33.1] | yes |
| 2 | +16.7 pp [+16.0, +17.6] | +16.5 pp | +30.0 pp [+28.9, +31.0] | yes |
| 3 | +15.5 pp [+14.8, +16.4] | +15.5 pp | +26.9 pp [+26.1, +27.6] | yes |

3 lookaheads passed; at least 2 were required.

The all-horizon K=8 curves, K=4/8/16 candidate-count comparison, and source-layer heatmap are descriptive. This milestone establishes held-out route-prediction behavior for GPT-OSS 20B only; it makes no language-quality, latency, 120B, or cross-model claim.
