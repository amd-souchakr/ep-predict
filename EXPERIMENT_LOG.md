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
- Figures: two primary lookahead plots show expert-selection coverage and
  complete-top-8 coverage at n+1/n+2/n+3; detailed capacity, layer/domain, and
  churn plots remain supplementary. PDF and 450-DPI PNG inputs are hashed.
- Human visual review: completed 2026-08-01. The simplified figures make the
  conclusion clear: selection coverage falls only from 79.0% at n+1 to 76.8%
  at n+3, while complete-top-8 coverage falls from 24.1% to 22.2%.
- Decision after review: advance to a minimal H3 proof/disproof experiment.
  Require a lightweight predictor to beat the H2 transition table at the same
  candidate budget; do not begin MLP, optimization, or ablation work first.
- One next action: preregister the H3 held-out gate and compact hook-captured
  feature format. Run the H4 physical kill switch before substantial predictor
  tuning.

### `h3-standard-small` — 2026-08-01

- Hypothesis: H3 — a lightweight linear sidecar can recover future-expert
  candidate sets better than routing transitions at equal capacity.
- Question: Does one fixed 128-to-64 linear multilabel head per phase,
  source layer, and lookahead materially improve both selection and complete
  top-8 coverage on held-out requests?
- Config: `configs/experiment/h3-standard-small.toml`; preregistration in
  `docs/H3_PROTOCOL.md`.
- Trace/run artifact: `artifacts/runs/h3-standard-small`; 128 requests,
  377,488 routing records, and 377,488 aligned projected router inputs.
- Result: `PILOT_DOES_NOT_SUPPORT`. At the primary decode K=16, Δ=1 gate,
  linear versus transition gained only +0.4 pp selection coverage but +4.7 pp
  complete-token coverage. Selection improved in 34/60 layer-domain scopes;
  only code and math improved both domain means. Linear candidate churn was
  52.8% versus 42.8% for transition.
- Integrity checks: zero router mismatches; 128/128 trace and feature shards;
  deterministic 128-dimensional projection; finite features; exact
  feature/route row alignment; exact H2 96/32 split; all 1,008 H2 transition
  scopes reproduced with maximum coverage difference 0.0; 21 tests pass.
- Decision: do not tune the predictor or add an MLP for this checkpoint. The
  hidden state has complementary complete-set information, but the fixed
  linear policy is not broadly superior enough to replace transition tables.
- Figures: two PDF/450-DPI PNG figures show lookahead behavior and primary
  domain consistency under `analysis/h3/figures`; input/output hashes are in
  `figure_manifest.json`.
- Human visual review: pending.
- One next action: after figure review, use the simpler H2 transition policy in
  the minimum H4 hardware-feasibility study.

### `h23-extended-horizon` — 2026-08-01

- Hypothesis: post-hoc H2/H3 characterization; formal gates unchanged.
- Question: How do transition and fixed linear predictor coverage change over
  every valid source-target layer pair through Δ=15?
- Config: `configs/experiment/h23-extended-horizon.toml`; protocol in
  `docs/H23_EXTENDED_HORIZON_PROTOCOL.md`.
- Trace/run artifact: reused `artifacts/runs/h3-standard-small`; no new
  inference. Analysis is under `analysis/h23_extended_horizon`.
- Result: routing remains predictable to the final layer, but source-layer
  regime dominates. At decode K=16, transition selection/complete coverage
  falls from 79.0%/24.1% at Δ=1 to 53.8%/4.6% at Δ=15; linear changes from
  79.4%/28.7% to 69.2%/19.7%. Linear wins selection in 100/120 and complete
  coverage in 112/120 domain-balanced source-target pairs.
- Layer insight: linear mean selection gain is +14.7 pp from source layer 0
  and +11.0 pp from layer 1, but becomes negative from source layer 10 onward.
  It is valuable for early long-range planning, not as a universal transition
  replacement.
- Integrity checks: all 120 valid pairs per phase; 240 fixed heads; preserved
  96/32 split; all original H2 transition scopes reproduced exactly.
- Cost: linear churn reaches 58.7% at Δ=15 versus 34.0% for transition.
- Decision: retain the formal H3 failure. In H4, keep the oracle-first physical
  gate and scan issue point explicitly; carry both existing candidate streams
  without MLP or predictor tuning.
- Figures: full-horizon curve and triangular source-target gain heatmap
  generated as PDF/450-DPI PNG with hashed inputs.
- Human visual review: pending.
- One next action: review the extended figures, then preregister the minimum H4
  issue-point/bandwidth/capacity feasibility scan.
