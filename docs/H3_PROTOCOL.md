# H3 protocol: linear hidden-state future-route prediction

## Decisive question

H3 asks:

> Can one lightweight linear multilabel sidecar per phase, source layer, and
> lookahead beat the H2 routing-transition table at the same candidate
> capacity on held-out requests?

This is a proof/disproof pilot, not a predictor optimization study. Only one
feature representation and one fixed training recipe are allowed. There is no
MLP, hyperparameter sweep, projection-size ablation, domain-specific head, or
post-hoc threshold tuning.

## Frozen scope

- Evidence grade: pilot.
- Model and revision: the pinned OLMoE-1B-7B checkpoint used for H1/H2.
- Workload: the same revision-pinned balanced 128 requests.
- Generation: greedy, batch one, prompt cap 384, decode cap 64.
- Routing: top-8 of 64 experts in 16 MoE layers.
- Phases: prefill and decode are trained and reported separately.
- Lookahead: target layer \(l+\Delta\), for
  \(\Delta\in\{1,2,3\}\).
- Candidate capacity: \(K\in\{8,16,32\}\).
- Split: the exact H2 96/32 request split, with 24/8 requests per domain.

All fitting, including feature standardization, uses training requests only.
Coverage is evaluated only on held-out requests. Domains remain separate in
the metric table; aggregate values are domain-balanced layer-domain means.

## Hook-only feature collection

No model or Transformers source is changed. The existing explicit router
forward hook reads positional input zero—the exact hidden-state tensor passed
into the router module—and immediately computes:

\[
x_l = h_l R,\qquad
R_{ij}\in\{-1,+1\}/\sqrt{128}.
\]

The fixed Rademacher projection has dimension 128 and seed 31. One projection
matrix is shared by all layers and requests. Its SHA-256 hash, input/output
dimensions, seed, scale, hook point, and storage dtype are recorded in the run
manifest. Only projected float16 features are persisted; full hidden states
and router logits are not stored.

Each request writes an atomic compressed NumPy shard containing numeric arrays
only (`allow_pickle=False`). Phase, position, layer, token ID, and record
ordering are stored beside the feature matrix and must exactly match the
routing JSONL shard before training. Collection remains request-resumable.
Hooked runs make no timing claim.

## Fixed predictor

For each `(phase, source layer, delta)`, fit exactly one affine
128-to-64 multilabel logistic-regression head:

\[
\hat y = W\,\mathrm{standardize}(x_l)+b.
\]

The target is a 64-dimensional multi-hot vector for the eight experts selected
at layer \(l+\Delta\). Rank logits and take the top \(K\) experts; do not tune a
decision threshold.

Frozen training recipe:

- train-only per-feature mean and standard deviation;
- zero-variance standard deviations replaced by one;
- BCE-with-logits loss;
- constant positive-class weight 7, matching 8 positives and 56 negatives;
- AdamW, learning rate 0.01, weight decay 0.0001;
- 30 epochs, batch size 1024, seed 23;
- deterministic sorted samples and no validation-based early stopping.

This intentionally tests whether a basic linear readout has incremental value.
If it fails, H3 stops for this checkpoint rather than escalating immediately
to an MLP or tuning sweep.

## Comparators

All comparators are refit on the H3 training requests, phase by phase:

1. static target-layer popularity across all training domains;
2. domain-oracle target-layer popularity using the known request domain;
3. the H2 transition rule
   \(P(E_{l+\Delta}\mid E_l)\), with the identical scoring and fallback rule
   from `docs/H2_PROTOCOL.md`;
4. the linear projected-hidden-state sidecar.

Recomputed H3 transition results must agree with H2 within numerical
roundoff. This detects route, split, or join drift.

## Metrics

For every `(phase, domain, source layer, delta, K, policy)` report:

- selection coverage: fraction of the actual top-8 experts in the candidate
  set;
- complete-token coverage: fraction of tokens whose entire top-8 set is
  covered;
- candidate amplification: \(K/8\);
- candidate churn: fraction of candidate slots replaced between consecutive
  held-out predictions in causal replay order;
- held-out token count.

Candidate amplification is fixed by capacity and is not a precision claim.
Wave-complete and decode-step-complete coverage begin with H4/H5, when batching
and physical replay semantics are defined.

## Preregistered go/no-go gate

The sole primary gate is **decode, \(K=16,\Delta=1\)**, comparing `linear`
against `transition` over paired eligible `(domain, source layer)` scopes.
H3 passes only if every condition holds:

- domain-balanced mean selection-coverage gain is at least +3 percentage
  points;
- domain-balanced mean complete-token-coverage gain is at least +2 percentage
  points;
- selection coverage improves in at least 75% of layer-domain scopes;
- complete-token coverage improves in at least 75% of layer-domain scopes;
- both mean metrics improve in at least 3 of 4 domains.

The thresholds are prototype engineering thresholds, not significance claims.
\(\Delta=2,3\), other capacities, and prefill are secondary evidence and
cannot rescue a failed primary gate.

## Decision

- **Pass:** record H3 as pilot-supported and proceed directly to the minimum H4
  hardware-feasibility study before predictor optimization.
- **Fail:** reject the need for a learned predictor for this checkpoint. Use
  the simpler H2 transition policy in H4. Do not start MLP or tuning work.

Neither outcome establishes transfer feasibility, latency improvement,
cross-model generality, or that the base model learned to manage hardware
resources.

## Required artifacts and review

Under `artifacts/runs/h3-standard-small/` retain:

- frozen run definition and manifest;
- model and projection reports;
- request routing and projected-feature shards;
- the copied/validated split definition;
- predictor weights and train-only normalizers in non-pickle NPZ;
- metric, summary, training, gate, and integrity tables;
- a short result report;
- one or two simple PDF/450-DPI PNG decision figures plus input hashes.

After figures are generated, pause for human review and record exactly one next
action before beginning H4.
