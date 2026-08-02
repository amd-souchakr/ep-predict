# GPT-OSS 20B MTP-style route-head confirmation

**Decision:** `CONFIRMATION_PASS`

The frozen 276-head weighted+binary route predictor was evaluated without refitting on 64 fresh requests (16 per domain) with zero prompt or sample-ID overlap against the previous 128 requests.

| Δ | Learned selection | Transition | Difference | Learned complete | Transition complete | Domains positive | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 91.7% | 86.0% | +5.7 pp | 74.1% | 60.0% | 4/4 | yes |
| 2 | 91.1% | 85.4% | +5.8 pp | 73.3% | 59.6% | 4/4 | yes |
| 3 | 90.0% | 84.1% | +5.9 pp | 70.8% | 56.8% | 4/4 | yes |

3/2 required lookaheads passed.

The predictor has 574,080 FP32 parameters (2.19 MiB) and costs 2,048 MACs per forecast.

This confirms route-only expert-demand prediction on one checkpoint and four workload domains. It does not establish latency benefit, language quality, or the accuracy of a future hidden-state/jointly-trained head.
