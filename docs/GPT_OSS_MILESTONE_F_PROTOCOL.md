# GPT-OSS 20B Milestone F learned-lookahead protocol

**State:** protocol ready; freeze the implementation config before fitting
**Checkpoint:** `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`
**Primary evidence:** existing Milestone E dispatch traces
**New model inference:** none for development; a fresh confirmation trace is
conditional on the development gate
**Scope:** learned expert-demand prediction, not cache-policy implementation or
end-to-end performance

## Decisive question

Can a compact learned, route-only model predict the complete future top-4
expert demand of GPT-OSS 20B at useful lookaheads and tunable candidate budgets
on request-held-out data?

Milestone E already established that a fitted transition table is strong. This
milestone tests whether that structure can be represented by a small
parameterized predictor suitable for the paper's lookahead-prediction claim.

The predictor estimates total expert demand. It does not predict only cold
experts, observe residency, or make prefetch/eviction decisions. A conceptual
software manager may later subtract resident and in-flight experts from the
forecast.

## Evidence boundary

The 32-request Milestone E test split has been inspected. Reusing it for a new
model is development evidence, even though those requests remain excluded from
fitting. A confirmatory claim requires the architecture, weights, thresholds,
and analysis to be frozen before evaluating a fresh request set.

The six Milestone E independent-weight deviations do not affect expert-set
labels. Require exact dispatch-ID parity and complete token/layer coverage.
Treat independent weight agreement with a preregistered BF16-aware ULP rule;
use dispatch-consumed weights for routed-mass metrics.

## Prediction task

For token `t`, phase `p`, source layer `l`, and target layer `l + delta`, form
a 32-dimensional sparse route vector whose four nonzero entries are the
dispatch-consumed source-route weights. Predict 32 target-expert logits for the
complete target top-4 set.

The fixed primary learned model is a compact shared route MLP:

1. input: the 32-dimensional weighted source route, learned source-layer and
   target-layer embeddings, and a prefill/decode phase embedding;
2. one hidden layer with 64 units and GELU activation;
3. output: 32 expert logits;
4. objective: multi-label binary cross-entropy against the target top-4 set;
5. no hidden-state capture, token text, domain label, request ID, or future
   token information.

The exact initialization seed, embedding widths, optimizer, learning rate,
weight decay, epoch budget, and checkpoint rule must be materialized in the
Milestone F TOML before the first fit. Do not tune against the 32 development
requests.

## Comparators

Evaluate at identical candidate count `K`:

1. global target-layer popularity;
2. domain-conditioned target-layer popularity (strong but metadata-assisted);
3. current-route copy filled by global popularity;
4. the Milestone E source-expert transition table;
5. the shared learned route MLP.

The transition table is both a strong baseline and valid evidence that a
data-fitted predictor already works. The MLP need not beat it everywhere, but
must show that a compact parameterized model preserves most of its frontier.

## Split and training discipline

- Preserve the Milestone E 96/32 request split for the development result.
- Fit only on the 96 training requests; no token or route from a development
  request may affect weights, normalization, stopping, or thresholds.
- Separate prefill and decode metrics, but one phase-conditioned model may be
  shared.
- Treat requests, not tokens or layer pairs, as the uncertainty unit.
- Record model parameter count, serialized bytes, and multiply-adds per
  forecast.

If the development gate passes, freeze the complete pipeline and collect 64
previously unused requests, 16 per domain, using the Milestone E inference and
trace protocol. The frozen predictor is evaluated without refitting. That run
is the confirmation; the reused 32-request result remains a development pilot.

## Candidate budgets and metrics

Evaluate `K = 4, 8, 12, 16`, corresponding to `K/4 = 1x, 2x, 3x, 4x`
candidate amplification and `K/32 = 12.5%, 25%, 37.5%, 50%` candidate-set
fractions. Also report a score-threshold frontier with variable candidates per
token, but do not use it to rescue a failed fixed-K gate.

For every phase, request, domain, source layer, target layer, and lookahead,
report:

- expert-selection coverage and candidate precision;
- dispatch-consumed routed-mass coverage;
- exact complete-top-4 coverage;
- useful candidate amplification;
- calibration/reliability of expert scores;
- candidate churn across adjacent source layers;
- parameter count, model bytes, and forecast operation count.

Aggregate within request first, then domains equally. Use paired stratified
request bootstrap intervals. Long-horizon summaries must retain source-target
cells or explicitly show the changing eligible-layer mixture.

## Development gate

Primary scope: decode, `K=8`, `delta in {1, 2, 3}`.

The learned model supports the next step only if at least two lookaheads meet
all of the following:

1. selection coverage is at least 82% and complete-top-4 coverage is at least
   50%;
2. selection coverage is no worse than the transition table by more than 3
   percentage points;
3. complete-route coverage is no worse than the transition table by more than
   5 percentage points;
4. it beats the stronger of domain popularity and route copy by at least 10
   selection-coverage points;
5. selection gain over that cheap comparator is positive in all four domains.

This is a compact-predictor sufficiency/noninferiority gate, not a claim that a
neural predictor must dominate a well-estimated table. The all-horizon and
variable-K scans are descriptive.

## Decision

- **Pass development:** freeze the predictor and analysis, run the fresh
  64-request confirmation, and require the same gate before calling the learned
  result confirmatory. Then use the confirmed predictor frontier as the
  empirical input to Milestone G.
- **Fail learned model, transition remains strong:** retain the transition
  table as the demonstrated lightweight predictor and narrow the paper's
  parameterized-model claim; do not tune a larger network automatically.
- **Fail both on confirmation:** withdraw the broad GPT-OSS lookahead claim and
  do not use the development frontier as empirical Milestone G evidence.

## Required outputs

- frozen configuration and exact train/development request lists;
- model weights, training history, parameter/operation accounting, and hashes;
- per-request, per-layer/horizon, and domain-balanced metrics;
- fixed-K coverage/amplification figure and score-threshold frontier;
- transition-versus-learned comparison with bootstrap intervals;
- concise decision report and durable artifact manifest.

No cache-manager implementation, asynchronous transfer prototype, router
modification, language-quality claim, or 120B download is part of Milestone F.
