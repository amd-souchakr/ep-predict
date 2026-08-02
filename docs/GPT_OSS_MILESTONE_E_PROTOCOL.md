# GPT-OSS 20B held-out prediction-quality protocol

**Frozen:** 2026-08-01, before Milestone E inference
**State:** authorized after researcher verification of Milestone D
**Checkpoint:** `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`
**Scope:** 20B routing analysis and route-prediction quality; no 120B download,
language-quality claim, timing claim, or learned hidden-state predictor

**Post-run terminology correction:** K is the number of nominated prediction
candidates. K/E is the candidate-set fraction, not resident capacity. This
experiment does not contain a residency variable or cache replay; the
correction changes no frozen K value, prediction, metric, threshold, or gate.

## Why Milestone E changed

The former plan made the approximately 60.8 GiB GPT-OSS 120B checkpoint the
next comparison. The researcher has rejected that experiment because local
disk capacity is insufficient. Milestone E therefore asks a narrower question
that the qualified 20B path can answer directly:

> On held-out requests, how much future top-4 expert demand can a route-only
> transition predictor cover, at what candidate amplification and candidate-set
> fraction, and does it beat strong cheap baselines?

This is a within-model prediction-quality test. It cannot establish
cross-model transfer or GPT-OSS 120B behavior.

## Frozen collection

- Use all 128 records in the revision-pinned `h1-standard-small` prompt file:
  32 each from code, conversation, general text, and math.
- Verify the prompt-file SHA-256 before loading model weights.
- Render each prompt with the checkpoint chat template, fixed current date
  `2026-08-01`, and generation prompt. Reject rather than truncate any rendered
  input longer than 512 tokens; the preflight maximum is 429.
- Run greedy batch-one inference with seed `20260801`, native MXFP4, exactly
  one visible MI355X, and 16 retained output tokens per request.
- Feed every retained token through a cached-token forward, including the last
  terminal trace-only forward, so all 16 decode tokens have complete routes.
- Reuse the Milestone C/D dispatch observer. Require 24 layers, 32 experts,
  top-4 routing, complete token-layer coverage, and zero independent-router to
  consumed-dispatch ID or selected-weight mismatches at `1e-6`.
- Do not repeat all 128 requests: Milestone D already established exact
  same-process determinism. This milestone spends the budget on independent
  requests instead.

## Frozen held-out analysis

- Split by request with seed 17, stratified 24 train / 8 test per domain. No
  token from a test request may train popularity or transition tables.
- Evaluate prefill and decode separately at K=4/8/16 and every layer lookahead
  Δ=1--23. Report K/top-k amplification and K/E candidate-set fraction.
- Compare four fixed policies:
  1. global target-layer popularity;
  2. domain-conditioned target-layer popularity;
  3. current-route copy, filled to K by global target-layer popularity;
  4. a global source-expert→target-expert transition table.
- Report expert-selection coverage, selected routed-mass coverage, exact
  complete-route coverage, and useful candidate amplification. Selected
  weights are consumed dispatch weights and sum to the routed top-4 mass.
- Aggregate each test request first, then average requests within domain and
  domains equally. Use 2,000 stratified request-level bootstrap resamples for
  paired uncertainty; tokens and layer pairs are not independent samples.

## Preregistered gate

The primary point is decode K=8: 2× candidate amplification and a candidate
set spanning 25% of the 32-expert namespace. For each Δ in {1,2,3}, compare
transition prediction with the better of domain popularity and current-route
copy. A lookahead passes only if:

1. mean selection-coverage gain is at least +3 percentage points and its 95%
   bootstrap confidence interval is strictly positive;
2. mean complete-route gain is at least +2 points and its 95% interval is
   nonnegative; and
3. selection gain is positive in at least three of four domains.

Milestone E supports short-horizon route prediction only if at least two of
the three lookaheads pass. The all-horizon scan is descriptive and cannot
rescue a failed primary gate. No H3 predictor training, latency replay, or
120B comparison follows automatically.

## Required outputs

- standard per-request trace shards and retained outputs;
- model inspection, run definition, coverage/parity integrity tables;
- frozen split, per-scope metrics, per-request metrics, domain-balanced
  horizon summary, bootstrap gate table, and machine-readable decision;
- one horizon-quality figure and one source-layer gain heatmap in PDF and
  450-DPI PNG;
- result report and hashed durable-artifact manifest.
