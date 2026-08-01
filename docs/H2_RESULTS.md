# H2 pilot findings

## Outcome

The preregistered H2 gate is **supported** for all tested lookaheads:

| Decode, K=16 | Δ=1 | Δ=2 | Δ=3 |
|---|---:|---:|---:|
| Selection-coverage gain over static | +38.0 pp | +36.6 pp | +35.1 pp |
| Complete-token gain over static | +23.5 pp | +22.9 pp | +21.5 pp |
| Layer-domain scopes with positive selection gain | 60/60 | 56/56 | 52/52 |
| Domains with positive mean gain | 4/4 | 4/4 | 4/4 |
| Gate | Pass | Pass | Pass |

For this checkpoint, the current top-8 route contains strong information about
the top-8 route one to three MoE layers ahead. This justifies a lightweight
external predictor experiment after human review. It does not yet establish a
prefetch or memory-hierarchy speedup.

## Held-out design and integrity

- Reused the existing H1 traces; no inference was collected.
- Split 128 requests into 96 train and 32 test requests, with 24/8 per domain.
- The split is deterministic, stratified, and disjoint by request.
- Popularity and transition tables use train requests only.
- Coverage and churn use held-out requests only.
- Source and target routes are joined on the same request, phase, and token
  position.
- Every joined token has all 16 layers and top-8-of-64 routing.
- Prefill and decode remain separate; expert namespaces remain layer-local.

An independent recomputation for decode layer 0 to 1 at K=16 reproduced 59.1%
transition selection coverage versus 37.0% static coverage on 2,016 held-out
tokens. This was separate from the analysis implementation and supports the
alignment and metric audit.

## Capacity and complete-set behavior

Domain-balanced decode results for the strongest lookahead, Δ=1:

| K | Candidate amplification | Static selection | Transition selection | Static complete token | Transition complete token |
|---:|---:|---:|---:|---:|---:|
| 8 | 1x | 24.2% | 58.0% | 0.0% | 1.2% |
| 16 | 2x | 41.0% | 79.0% | 0.6% | 24.1% |
| 32 | 4x | 67.2% | 93.2% | 9.8% | 64.1% |

The ordinary selection metric is encouraging but the complete-token result is
the hardware-relevant warning. At K=8, predicting substantially more selected
experts still almost never covers the complete top-8 route. K=16 is a useful
middle point but covers the full route for only about one quarter of tokens.
K=32 reaches 64% complete-token coverage while proposing half of all experts.

Prefill is similar at Δ=1 and K=16: transition selection coverage is 79.0%
versus 40.4% static, and complete-token coverage is 26.7% versus 2.3%. The
signal is therefore not a decode-only artifact, although its operational use
will differ because prefill tokens execute in parallel.

## Which conditioning signal matters?

At decode Δ=1 and K=16:

| Baseline | Selection coverage | Complete-token coverage | Candidate replacement/token |
|---|---:|---:|---:|
| Static per-layer popularity | 41.0% | 0.6% | 0.0% |
| Domain-oracle popularity | 49.3% | 3.0% | 0.0% |
| Previous complete-request window | 48.6% | 3.0% | 0.1% |
| Current-route transition | 79.0% | 24.1% | 42.8% |

Domain metadata recovers a useful but much smaller part of the gap. The causal
previous-window set does not improve on the domain oracle, so this pilot does
not support frequent histogram-only adaptation beyond domain conditioning.
The dominant signal comes from the current token's route.

The train-only transition conditional entropy is about 10.6%, 10.0%, and 9.5%
lower than marginal target-layer entropy for decode Δ=1,2,3. Prefill reductions
are 11.0%, 10.4%, and 10.0%. These modest entropy reductions coexist with
large top-K coverage gains because ranking the candidate tail is more relevant
than reducing the full 64-way entropy uniformly.

## Domain and layer robustness

At decode Δ=1 and K=16, transition versus static selection gains are:

| Domain | Static | Transition | Gain | Transition complete token |
|---|---:|---:|---:|---:|
| Code | 57.5% | 83.1% | +25.6 pp | 33.3% |
| Mathematics | 44.1% | 82.6% | +38.5 pp | 29.5% |
| General prose | 27.1% | 74.5% | +47.4 pp | 15.4% |
| Conversation | 35.2% | 75.8% | +40.5 pp | 18.1% |

The transition gain is largest where static popularity is weakest. Across all
decode lookaheads, individual layer-domain selection gains range from +10.8
to +62.5 percentage points; all 168 eligible scopes are positive. This makes
it unlikely that the aggregate result is caused by one domain or a few layers,
though the pilot still uses one held-out split and has no request-bootstrap
confidence intervals.

## Placement implication: information is not movement

The K=16 transition candidates replace 42.8%, 41.4%, and 40.4% of slots per
decode token at Δ=1,2,3. Prefill replacement is 39.2%, 38.3%, and 38.0%.
Literal candidate residency would therefore request roughly six to seven
different slots per prediction.

This does not invalidate H2. It changes the likely use:

- prioritize or reserve transfers rather than blindly load every candidate;
- combine predictions across tokens into wave-level demand;
- use stable experts for residency and conditional candidates for scheduling;
- test whether Δ=1–3 supplies enough physical transfer time before optimizing
  a learned predictor;
- charge movement, cache occupancy, and false-prefetch bandwidth explicitly.

## Decision and recommendations

1. Record H2 as **pilot supported** for OLMoE. Routing-only conditional
   information strongly beats static and domain-only popularity.
2. The condition for trying a lightweight external predictor is met. Start
   with a linear/low-rank sidecar and require it to beat the transition table
   at the same K, not merely static popularity.
3. The simplified H2 figures were human-reviewed on 2026-08-01 and approved as
   clear evidence to advance to a minimal H3 proof/disproof experiment.
4. Before substantial predictor tuning, execute the minimum H4 oracle timing
   kill switch using unhooked layer time, exact 12 MiB expert bytes, and a
   measured PCIe transfer curve. Strong information has no latency value if
   the lookahead cannot hide transfer.
5. In H3/H5, report wave- and decode-step-complete coverage, not just the
   token metrics used for this information gate.
6. Repeat H1/H2 on one newer, more sparsely routed top-1/top-2 checkpoint
   before generalizing. OLMoE's top-8 route supplies eight source observations
   and creates unusually demanding complete-set behavior; both can differ
   substantially in a more sparse router.

## Figures

Generate the H2 figures with:

```bash
uv run ep-predict plot-h2 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h2-standard-small.toml
```

Outputs under `analysis/h2/figures`:

- `fig1_predictability_by_lookahead`: fraction of future experts found at
  n+1, n+2, and n+3;
- `fig2_complete_route_by_lookahead`: strict all-eight-experts coverage at
  n+1, n+2, and n+3.

The figure directory contains vector PDFs, 450-DPI PNGs, input hashes, and the
required human-review checklist.
