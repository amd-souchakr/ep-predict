# GPT-OSS 20B Milestone F development result

**Decision:** `DEVELOPMENT_FAIL`

The fixed shared route MLP was fit on 96 retained requests and evaluated once on the 32-request development split. This split had already been inspected during Milestone E, so this result is not confirmatory.

| Δ | Learned selection | vs transition | vs cheap | Learned complete | vs transition | Domains positive | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 62.6% | -23.7 pp | -5.5 pp | 22.9% | -37.9 pp | 0/4 | no |
| 2 | 63.1% | -22.4 pp | -5.7 pp | 21.4% | -38.1 pp | 0/4 | no |
| 3 | 64.8% | -19.6 pp | -4.0 pp | 25.4% | -31.4 pp | 0/4 | no |

0/2 required lookaheads passed the unchanged gate.

## Compactness and audit checks

- Parameters: 5,864 (23,456 FP32 bytes).
- Forecast cost: 5,376 multiply-accumulates.
- Milestone E baseline reproduction: 17,664 metric values, maximum absolute difference 0; pass.

No cache state, cold-expert label, token text, domain label, hidden state, or development request entered model fitting. Fresh confirmation is required before this can support a confirmatory learned-predictor claim.

The conditional fresh run was not collected because this gate failed. A
post-gate check found only 61.1%, 61.6%, and 63.3% training-request decode K=8
selection coverage at Δ=1/2/3, so ordinary held-out generalization collapse
does not explain the failure. See `analysis/posthoc_train_decode.json`; this
diagnostic did not affect the gate.
