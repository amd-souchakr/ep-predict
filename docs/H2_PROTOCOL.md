# H2 protocol: routing-conditioned future expert demand

## Purpose

H2 is an information gate:

> Does current or recent routing state predict future expert demand better
> than held-out per-layer marginal popularity?

This trace-only pilot reuses `artifacts/runs/h1-standard-small`; it performs no
new inference, does not use hidden states, and does not train a learned model.
It decides whether an external H3 predictor is worth implementing.

## Evidence grade and scope

- Grade: pilot.
- Model: the pinned OLMoE-1B-7B checkpoint used for H1.
- Workload: the same 128 balanced requests.
- Routing: top-8 of 64 experts at each of 16 MoE layers.
- Phases: prefill and decode are evaluated separately.
- Domains: code, conversation, general prose, and mathematics remain separate.
- Lookahead: target layer \(l+\Delta\), for \(\Delta \in \{1,2,3\}\).
- Candidate capacities: \(K \in \{8,16,32\}\).

The trace is sufficient because every record is keyed by request, phase,
token position, and layer. Source and target routes are joined only on the
same request, phase, and token position. The source route at layer \(l\) is
available before the target route at layer \(l+\Delta\).

## Leakage control and split

Use a deterministic stratified request split with seed 17:

- 24 train requests and 8 test requests per domain;
- split by request, never by token;
- fit all popularity and transition tables on train requests only;
- evaluate all reported coverage metrics on test requests only.

The exact request IDs and sample IDs are written to `split.json`. Domains are
kept separate in primary tables. Aggregate rows are domain-balanced means, not
token-weighted mixtures.

## Fixed baselines

At every target layer, phase, and candidate capacity:

1. **Static per-layer popularity:** top-\(K\) target experts from all training
   requests. This is the primary comparator.
2. **Domain-oracle popularity:** top-\(K\) target experts from training
   requests of the test request's known domain. "Oracle" refers only to the
   domain label; it never sees held-out target routes.
3. **Previous-window hot set:** top-\(K\) target experts in the immediately
   preceding held-out replay window for the same domain, phase, and target
   layer. Windows target at least 128 tokens but boundaries occur only between
   complete requests, so prefill tokens from one parallel forward never
   predict other tokens in that forward. The first window falls back to the
   domain-oracle set. A window is used only after all its requests are in the
   past, so the policy is causal.
4. **Layer transition table:** estimate
   \(P(E_{l+\Delta}=e' \mid E_l=e)\) on training requests. For a source token's
   top-8 route, score each target expert by the mean of its conditional
   probabilities across the eight selected source experts; select the
   top-\(K\), breaking ties by expert ID. Unseen source rows fall back to the
   target-layer marginal distribution.

The transition table is deliberately not domain-conditioned: this isolates
the incremental value of the current route from the already measured value of
domain metadata.

## Metrics

For each `(phase, domain, source layer, delta, capacity, baseline)` report:

- **selection coverage:** mean fraction of the actual top-8 target experts in
  the candidate set;
- **complete-token coverage:** fraction of tokens for which all eight target
  experts are candidates;
- **candidate amplification:** \(K/8\), equal to 1x, 2x, or 4x here;
- **useful amplification:** candidates issued per covered expert selection,
  exposing waste at a fixed capacity;
- **hot-set churn:** mean fraction of candidate slots replaced between
  consecutive held-out predictions in causal replay order;
- count of held-out token-layer pairs.

Also report train-only marginal entropy, transition conditional entropy, and
normalized entropy reduction for every `(phase, source layer, delta)`.
Entropy is descriptive; held-out coverage decides the gate.

## Preregistered decision gate

The primary scope is **decode at \(K=16\)**, comparing the transition table
with static per-layer popularity. For each lookahead, compute paired gains for
every eligible `(domain, source layer)` scope and domain-balanced means.

A lookahead passes only if all are true:

- mean selection-coverage gain is at least +3 percentage points;
- mean complete-token-coverage gain is at least +2 percentage points;
- selection coverage improves in at least 75% of eligible scopes;
- mean selection coverage improves in at least 3 of 4 domains.

H2 supports advancing to a lightweight external H3 predictor if at least one
lookahead passes. If recent-window locality improves but the transition gate
does not, continue only with adaptive residency/window policies, not a
skip-layer hidden-state predictor. If neither improves materially, stop H3
predictor work for this checkpoint and advance the oracle hardware-feasibility
kill switch.

The thresholds are prototype engineering thresholds, not claims of
statistical significance.

## Required artifacts

Write under `artifacts/runs/h1-standard-small/analysis/h2/`:

- `split.json`;
- `metrics.csv`;
- `entropy.csv`;
- `summary.csv`;
- `gate.json` and `summary.json`;
- `REPORT.md`;
- one to three scripted PDF/PNG decision figures and an input-hash manifest.

After plots are generated, pause for human visual review before H3.

## Interpretation boundaries

A positive H2 result means that a frozen OLMoE route contains useful
short-horizon information for an external policy. It does not show that:

- the base model learned to manage hardware resources;
- weights can be transferred before demand;
- prediction improves latency;
- the effect generalizes beyond this top-8-of-64 checkpoint.

If H2 is positive, repeat H1/H2 on one newer, more sparsely routed top-1/top-2
MoE before making a general architectural claim.
