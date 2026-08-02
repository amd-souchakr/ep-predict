# GPT-OSS 20B Milestone F development result

**Run:** `gpt-oss-20b-milestone-f-development`

**Date:** 2026-08-01

**Decision:** `DEVELOPMENT_FAIL`

**Fresh confirmation:** not authorized and not collected

**Milestone G use:** prohibited as confirmatory learned-model evidence

> **Subsequent result:** this negative decision applies to the frozen shared
> 64-unit MLP. A separately frozen MTP-style layer-pair head architecture later
> passed development and 64-request fresh confirmation. See
> [GPT_OSS_MULTIHEAD_RESULTS.md](GPT_OSS_MULTIHEAD_RESULTS.md). The original
> failure and no-confirmation decision remain unchanged for the shared MLP.

## Result

The frozen 5,864-parameter shared route MLP did not preserve the transition
table's lookahead frontier. It passed zero of the three primary decode K=8
lookaheads; the protocol required at least two.

| Δ | Learned selection | Transition | Learned − transition | Strong cheap comparator | Learned − cheap | Learned complete | Transition complete | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 62.6% | 86.3% | −23.7 pp | 68.1% domain popularity | −5.5 pp | 22.9% | 60.7% | no |
| 2 | 63.1% | 85.5% | −22.4 pp | 68.8% domain popularity | −5.7 pp | 21.4% | 59.4% | no |
| 3 | 64.8% | 84.4% | −19.6 pp | 68.8% domain popularity | −4.0 pp | 25.4% | 56.8% | no |

The paired stratified-request 95% intervals exclude a close call. The learned
minus transition selection intervals are −24.4 to −22.9 pp, −23.1 to −21.8 pp,
and −20.2 to −18.9 pp. Learned minus cheap-comparator selection intervals are
also wholly negative: −6.1 to −5.0 pp, −6.2 to −5.2 pp, and −4.6 to −3.4 pp.
Every per-domain primary gain is negative.

## Frozen experiment

The configuration was materialized before fitting at
[`configs/experiment/gpt-oss-20b-milestone-f.toml`](../configs/experiment/gpt-oss-20b-milestone-f.toml).
Its run-time copy has SHA-256
`026fe84b769459ca84ebf3a21be25be5ee4a004362234eabc58aa99ec0896bfc`.

- Input: the dispatch-consumed weighted top-4 route as a sparse 32-vector,
  plus 8-d source-layer, 8-d target-layer, and 4-d phase embeddings.
- Network: one 64-unit exact-GELU hidden layer and 32 expert logits.
- Objective: unweighted multilabel BCE; AdamW at 0.001 with 0.0001 weight
  decay; 20 full shuffled epochs; final-epoch checkpoint; no validation,
  early stopping, scheduler, or development-driven choice.
- Fit: 96 requests, 18,126 tokens, 5,002,776 source-target examples per epoch,
  100,055,520 example presentations total.
- Development: 32 requests, 6,074 tokens, and 1,676,424 forecasts. Requests
  remained the bootstrap unit and domains received equal aggregate weight.
- Compactness: 5,864 FP32 parameters, 23,456 parameter bytes, 26,359-byte
  checkpoint, and 5,376 multiply-accumulates per forecast.

No cache state, resident/cold labels, domain label, token text, request ID,
hidden state, or future-token information entered the model.

## Frontier and calibration

Increasing K rescues absolute coverage only by spending most of the expert
space:

| Δ | K=4 selection / complete | K=8 | K=12 | K=16 |
|---:|---:|---:|---:|---:|
| 1 | 42.6% / 4.7% | 62.6% / 22.9% | 75.3% / 41.3% | 84.4% / 57.1% |
| 2 | 43.2% / 4.5% | 63.1% / 21.4% | 77.0% / 42.5% | 86.3% / 61.4% |
| 3 | 43.9% / 5.7% | 64.8% / 25.4% | 77.1% / 44.0% | 86.7% / 61.4% |

K=16 means four candidates per demanded expert and half of all experts. That
is not preservation of the transition frontier: the transition table already
reaches 84.4–86.3% selection at K=8.

The model is superficially well calibrated while ranking poorly. At primary
decode lookaheads its domain-balanced mean score is 0.126–0.127 against the
exact 0.125 positive frequency, Brier score is 0.0878–0.0895, and ten-bin ECE
is 1.26–1.38 pp. This is consistent with learning marginal expert frequencies
without learning enough conditional discrimination. Calibration alone is
therefore a dangerously weak success criterion for sparse top-k demand.

At K=8 the learned model beats route copy at all 23 decode horizons, but beats
domain popularity at none and transition at none. Across primary
domain/source-layer cells it beats the stronger cheap comparator in only
18/92, 12/88, and 24/84 cells for Δ=1/2/3. Its mean K=8 decode candidate
replacement fraction across adjacent source forecasts is 22.6%.

## Integrity and diagnosis

The implementation reproduced all 17,664 comparable Milestone E K=8
per-request baseline metric values with maximum absolute difference zero.
Source traces retain exact dispatch IDs and complete coverage. The six legacy
independent-weight deviations are within the frozen BF16-aware absolute and
frequency bounds; model inputs and routed-mass metrics use dispatch-consumed
weights.

A post-gate diagnostic evaluated the frozen checkpoint on the fitting
requests. Decode K=8 selection was 61.1%, 61.6%, and 63.3% for Δ=1/2/3, with
complete-route coverage 22.2%, 21.2%, and 24.9%. Training-request performance
is not better than development performance, so ordinary held-out
generalization collapse does not explain the result.

The likely failure class is underfitting or objective mismatch, not corrupted
labels. One 64-unit shared layer must represent many source-target conditional
maps, while naturally sampled decode examples are only 8.5% of fitting data
and unweighted BCE rewards good marginal probabilities. These are post-hoc
hypotheses, not established causes. Testing them would require a new
preregistered development cycle and independent validation data; silently
changing phase weights, loss, epochs, or width now would be test-set tuning.

## Decision

Do not collect the conditional 64-request confirmation. Retain the transition
table as the demonstrated lightweight GPT-OSS predictor and withdraw the
claim that this compact parameterized model preserves its frontier. Do not use
the failed learned frontier as confirmatory empirical input to Milestone G.
Any analytical continuation must either use the already-supported transition
frontier with an explicitly revised claim or wait for a separately authorized,
preregistered learned-model study with genuinely fresh development and
confirmation sets.

Artifacts, including the frozen TOML, checkpoint, request lists, training
history, per-request and per-layer tables, bootstrap intervals, calibration,
threshold frontier, churn, figures, report, and manifest, are under
[`artifacts/runs/gpt-oss-20b-milestone-f-development/`](../artifacts/runs/gpt-oss-20b-milestone-f-development/).
