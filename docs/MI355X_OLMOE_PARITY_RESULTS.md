# MI355X OLMoE matched-workload baseline results

**Primary run:** `artifacts/runs/mi355x-olmoe-instruct-c0-paired`  
**Superseded tracer bullet:** `artifacts/runs/mi355x-olmoe-parity`  
**Platform:** one visible AMD Instinct MI355X (`gfx950`), ROCm 7.2  
**Evidence grade:** exact 128-request matched-workload derived-artifact
comparison  
**Milestone decision:** supports advancing to isolated MI355X H4 calibration
after human review; raw selected-ID parity remains untestable

## Plain-language conclusion

The noticeable H1 difference in the earlier 16-request plot was a sampling
artifact. After rerunning all 128 prompts with the same raw serialization,
generation configuration, H1 settings, and H2 request split as the NVIDIA C0
run, both H1 and H2 curves are effectively coincident.

The full MI355X run completed 222,688 token-layer records with zero router
mismatches. Mean top-8 popularity coverage differs from NVIDIA by only
+0.0011 percentage points, with a 0.0134-point layerwise mean absolute
difference. H2 transition selection gain differs by only 0.0402 points on
average across all 15 horizons.

The correct result is:

> **OLMoE's aggregate prefill routing geometry and conditional trajectory
> statistics reproduce on MI355X to within very small numerical differences.
> The original 16-request H1 offset was caused by the incomplete workload, not
> evidence of a platform-scale routing change.**

NVIDIA raw records remain unavailable, so this result cannot compute
token-layer selected-ID agreement or authorize record-level trace
interchangeability.

## Matched scope and integrity

The MI355X collection config is mechanically identical to
`c0-olmoe-instruct-collect.toml` after excluding only run identity, output
path, and platform-comparison metadata. The H2 config is likewise identical
after excluding analysis identity and trace path.

| Integrity check | Result |
|---|---:|
| Requests | 128 MI355X / 128 NVIDIA |
| Request keys and order | exact match |
| Prompt-file SHA-256 | exact match |
| Collection settings | exact match |
| H2 96/32 request split | exact match |
| Comparable input-token hashes | 126 |
| Comparable token-hash mismatches | 0 |
| MI355X input-token hashes recorded | 128/128 |
| MI355X routed token-layer records | 222,688 |
| Router validation mismatches | 0 |
| Requests with missing/extra router calls | 0 |
| Model geometry | exact report match |

Two old NVIDIA manifest entries do not retain input hashes because they were
recorded as already-complete requests by an earlier collector version. Their
request keys and shared prompt/config provenance match, but they cannot be
included in the 126-request token-hash check.

The project verifier separately confirmed one visible MI355X, BF16 compute,
pinned H2D transfer, ROCm 7.2, and the tiny OLMoE router-hook test. The full
checkpoint reproduces 16 routed layers, 64 experts/layer, top-8 routing, and
12,582,912 BF16 bytes/expert.

## H1 matched skew and hot-expert results

| Derived comparison | MI355X | NVIDIA | Cross-platform comparison |
|---|---:|---:|---:|
| Mean top-8 popularity coverage | 25.1758% | 25.1746% | +0.0011 pp |
| Layerwise top-8 mean absolute difference | — | — | 0.0134 pp |
| Layerwise top-8 maximum difference | — | — | 0.0305 pp |
| Layerwise top-8 Pearson/Spearman | — | — | 0.999962 / 1.000000 |
| Mean Gini | 0.264476 | 0.264493 | −0.000017 |
| Layerwise Gini mean absolute difference | — | — | 0.000161 |
| Layerwise Gini Pearson/Spearman | — | — | 0.999978 / 1.000000 |
| Mean top-8 hot-expert intersection | — | — | 7.875 of 8 |
| Mean top-8 hot-set Jaccard | — | — | 0.9722 |

Fourteen of 16 layers have exactly the same aggregate top-8 hot-expert set.
At layer 5, the eighth expert is 6 on MI355X and 52 on NVIDIA. At layer 14,
it is 11 versus 3. The first seven experts match in both cases. These are
marginal rank-boundary substitutions, not a changed layerwise skew regime.

Both platforms make the same H1 decision: zero of 16 prefill layers pass the
combined universal-hot-tier gate. The per-layer coverage shape, skew peaks,
and threshold crossings agree.

## H2 matched trajectory results

Both platforms use the same 96 training and 32 held-out requests, including
24/8 per domain, and the same K=8/16/32 and Δ=1..15 analysis.

| Derived comparison | Result |
|---|---:|
| Selection-gain Pearson/Spearman | 0.999989 / 1.000000 |
| Selection-gain mean absolute difference | 0.0402 pp |
| Selection-gain maximum difference | 0.0809 pp |
| Complete-route-gain Pearson/Spearman | 0.999922 / 1.000000 |
| Complete-route-gain mean absolute difference | 0.1008 pp |
| Complete-route-gain maximum difference | 0.2549 pp |
| Passing horizons | Δ=1..15 on both platforms |

At K=16, the MI355X transition selection gain is +39.34 points at Δ=1 and
+12.51 points at Δ=15. The NVIDIA values are +39.32 and +12.56 points. The
small residuals do not change any horizon decision, ordering, or scientific
interpretation.

## Interpretation

### Directly supported

1. The hook-visible OLMoE Transformers path is valid on one MI355X.
2. With identical requests and analysis splits, aggregate H1 skew and H2
   transition statistics reproduce extremely closely across CUDA/NVIDIA and
   ROCm/AMD execution.
3. The 16-request H1 discrepancy was caused by inadequate matched sampling.
4. Existing OLMoE conclusions about universal hotness and conditional depth
   structure are stable enough to anchor the AMD baseline phase.

### Still not established

- exact selected expert IDs for every NVIDIA/MI355X token-layer record;
- equivalence of selected routing weights or router-logit margins;
- decode-route parity;
- timing or latency parity;
- permission to relabel an old NVIDIA raw trace as measured AMD demand.

The tiny aggregate residuals are consistent with a small number of numerical
route-boundary changes, but the absent NVIDIA trace prevents localizing them.

## Decision and recommendation

Advance to Milestone B after researcher review: remeasure hook-free cached
decode timing and pinned host-to-device bandwidth in the isolated
`mi355x-h4-calibration` directory.

For H4 interpretation:

- The old NVIDIA trace may be replayed with MI355X timing only as a
  counterfactual hardware-only sensitivity.
- A measured AMD H4 demand replay still requires a new MI355X decode trace,
  because the present exact comparison is prefill-only with one generated
  token.
- Preserve the RTX calibration unchanged as historical evidence.

Do not begin GPT-OSS qualification until the MI355X H4 calibration result has
been reviewed.

## Artifacts

- Protocol: `MI355X_OLMOE_PARITY_PROTOCOL.md`
- Primary run definition/environment: `artifacts/runs/mi355x-olmoe-instruct-c0-paired/`
- H1 tables/report: `analysis/h1/`
- H2 tables/report: `analysis/h2/`
- Matched comparison tables/report: `analysis/platform_comparison/`
- Comparison PDF/PNG and figure manifest:
  `analysis/platform_comparison/figures/`

