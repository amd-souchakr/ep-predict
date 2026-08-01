# Experiment log

Keep one short entry per meaningful run. Do not log routine command retries.
The immutable run manifest and metrics remain the source of truth.

## Template

### `<run-id>` — `<YYYY-MM-DD>`

- Hypothesis:
- Question:
- Config:
- Trace/run artifact:
- Result:
- Integrity checks:
- Decision:
- One next action:

## Runs

### `h1-standard-small-dataset` — 2026-07-31

- Hypothesis: H1 preparation.
- Question: Can a small balanced standard workload be materialized reproducibly?
- Config: `configs/dataset/h1-standard-small.toml`.
- Artifact: `artifacts/datasets/h1-standard-small/manifest.json`.
- Result: 128 unique prompts, 32 each from WikiText-2, GSM8K, HumanEval,
  and MT-Bench; SHA-256
  `4e08d5b4753ed1d06e6922359ea63249d4e4d215c9bc5081204533adf369fcf1`.
- Integrity checks: pinned revisions, balanced domains, source-code whitespace
  retained, answers/solutions excluded.
- Decision: use this for H1; retain authored prompts only as a hook smoke test.
- One next action: collect a two-request OLMoE trace on the CUDA host.

### `h1-standard-small` — 2026-07-31

- Hypothesis: H1 — expert popularity is skewed and operationally stable.
- Question: Does an 8-of-64 fast tier have at least 2x uniform coverage and
  stable expert identity across at least half of OLMoE's layers?
- Config: `configs/experiment/h1-standard-small.toml`.
- Trace/run artifact: `artifacts/runs/h1-standard-small`; 128 requests,
  377,488 records, and 3,019,904 expert selections.
- Result: the mixed decode tier covered 24.2% of selections on average.
  Seven layers met the skew threshold, but only layers 6 and 9 also met the
  stability thresholds. Code and math were much stronger than conversation.
- Integrity checks: 128 unique request traces, all 16 routers observed,
  top-k consistently 8, schema/layer/expert-ID checks passed, and mismatch
  failures enabled during collection.
- Decision: `PILOT_DOES_NOT_SUPPORT` for a universal mixed-workload tier;
  record the broader result as mixed because domain-conditioned routing is
  strong. See `docs/H1_RESULTS.md`.
- Figures: capacity/coverage, skew/stability, and domain-shift figures generated
  as PDF and 450-DPI PNG under
  `artifacts/runs/h1-standard-small/analysis/h1/figures`; input hashes are in
  `figure_manifest.json`.
- Human visual review: completed 2026-08-01; the mixed interpretation stands.
- One next action: review the H1 figures with the human, then preregister H2
  conditional-locality baselines if the interpretation still holds.

### `h2-standard-small` — 2026-08-01

- Hypothesis: H2 — current/recent routing predicts future expert demand better
  than per-layer marginal popularity.
- Question: On held-out requests, do routing transition tables at Δ=1/2/3
  improve selection and complete-token coverage at the same candidate budget?
- Config: `configs/experiment/h2-standard-small.toml`; preregistration in
  `docs/H2_PROTOCOL.md`.
- Trace/run artifact: reused `artifacts/runs/h1-standard-small`; no new
  inference. Analysis is under `analysis/h2`.
- Result: `PILOT_SUPPORT`. At decode K=16, transition tables beat static
  popularity by +38.0/+36.6/+35.1 pp selection coverage and
  +23.5/+22.9/+21.5 pp complete-token coverage for Δ=1/2/3. All 168 eligible
  layer-domain comparisons were positive.
- Integrity checks: deterministic disjoint 96/32 request split; 24/8 per
  domain; same-request/phase/token source-target joins; top-8 and 16-layer
  completeness; an independent layer-0-to-1 recomputation agreed.
- Decision: routing-only conditional information is strong enough to justify a
  lightweight external predictor after visual review. This is not a latency
  result: K=16 transition candidates replace about 40–43% of slots per token.
- Figures: capacity/coverage, layer-domain gain, and coverage/churn figures
  generated as PDF and 450-DPI PNG with hashed inputs.
- Human visual review: pending.
- One next action: review the H2 figures, then run the minimum H4 oracle timing
  kill switch before substantial H3 predictor tuning.
