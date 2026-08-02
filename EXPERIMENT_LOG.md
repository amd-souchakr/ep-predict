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
- Human visual review: completed 2026-08-01. The formal H3 failure stands; the
  extended scan below narrows, but does not replace, that decision.
- One next action: run the minimum oracle-first H4 hardware-feasibility study.

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
- Human visual review: completed 2026-08-01. The researcher accepted the
  changing-layer-count caveat and the early-versus-late source-layer regime.
- Protocol lesson: keep one simple early gate, then prefer cheap broad post-hoc
  analysis over a complex preregistered hypothesis tree. Treat the current
  held-out split as discovery data for any newly found hybrid policy.
- One next action: preregister the minimum H4 oracle
  issue-point/bandwidth/capacity feasibility scan.

### `h4-oracle` — 2026-08-01

- Hypothesis: H4 — perfect future knowledge can move exact 12 MiB experts
  early enough to eliminate a meaningful fraction of cold-expert decode stall.
- Config: `configs/experiment/h4-oracle.toml`; preregistration in
  `docs/H4_PROTOCOL.md`.
- Calibration: four hook-free cached-token chains produced 80 decode-forward
  samples. Median forward time was 10.229 ms, or a 0.639 ms effective interval
  across 16 MoE layers. The exact 12 MiB pinned-host copy median was 0.524 ms
  and the fitted effective bandwidth was 24.14 GB/s.
- Trace replay: reused the immutable 128-request H1 trace; no routing
  collection, model modification, inference-library modification, or timing
  hook. Scanned K=8/16/32, Δ=1/2/3/6/9/12/15, and 0.5×/1×/2× measured
  bandwidth with one serialized copy engine.
- Formal result: `PILOT_DOES_NOT_SUPPORT`. At measured bandwidth and K=16,
  the best frozen short horizon was Δ=3: 32.8% deadline-feasible cold bytes
  and 38.9% oracle stall reduction, below both 50% thresholds. 74.5% of
  eligible synchronous waves still stalled.
- Miss diagnosis: at K=16, Δ=3, only 832/360,247 cold occurrences were
  compulsory first uses; 359,415 were capacity-eviction misses. Repeated
  movement, not startup, dominates.
- Post-gate scan: the physical mechanism is not universally impossible.
  K=32, Δ=3 reaches 55.5% timely cold bytes and 61.8% stall reduction;
  K=16 reaches 61.8% at Δ=9; and K=16, Δ=1 reaches 58.1% stall reduction at
  2× bandwidth. These descriptive cells do not rewrite the formal gate.
- Decision: do not overlay, retrain, or tune transition/linear policies.
  First complete human figure review. If the K=32 boundary remains relevant,
  validate only that cell with a minimal concurrent copy/compute mechanism
  check.
- Figures: one oracle bandwidth/lookahead/capacity heatmap and one
  measured-bandwidth stall-reduction curve, as PDF and 450-DPI PNG with
  hashed inputs.
- Human visual review: completed 2026-08-01. The formal result stands; the
  researcher accepted first-order modeling for subsequent co-design window
  analysis rather than requiring timing-fidelity work first.
- Post-hoc synthesis: a co-design regime map combines cold-service headroom
  \(\Delta T_{\text{layer}}/(\bar N_{\text{cold}}T_{\text{copy}})\) with
  complete top-8 prediction coverage. It separates both-limited,
  transfer-limited, prediction-limited, and candidate regions. K=32, Δ=3–6 is
  the only region where both existing predictors exceed 50% complete coverage
  and the trace-driven oracle passes. This does not change the formal H4 gate
  or establish profitability.

### Next-experiment plan update — 2026-08-01

- Decision: performance-model fidelity is not the current research focus.
  Advance with first-order analytical viability and profitability windows;
  defer concurrent-overlap validation until a policy region justifies it.
- H5-A: sweep assumed complete-route coverage, 1×/2×/4× candidate
  amplification, K=8/16/32, Δ=1–15, and 0.25×–4× normalized cold bandwidth
  using existing trace-derived demand and reuse.
- H5-B: invert the sweep to report minimum complete coverage and maximum
  amplification required for useful benefit.
- Proposed screen to freeze before execution: ≥25% modeled stall reduction,
  ≥50% oracle recovery, and ≤2× predicted/useful transferred bytes.
- H5-C: reconstruct existing transition and linear candidates at four
  representative boundary/control cells; report cold-set coverage, false/late
  bytes, expected stall reduction, and oracle recovery without retraining.
- Visualization: one categorical profitability phase diagram and one
  inverse-requirement curve; actual policies are overlaid or summarized in a
  compact actual-versus-required table.
- Insight mining: identify empty windows, boundary/crossover locations,
  capacity–bandwidth–lookahead substitution, active constraints, and durable
  dimensionless quantities after the frozen gate.
- Conditional later work: H6 mechanism competition, one small H7
  predictable-routing training intervention with loss/load-balance controls,
  and one C1 top-1/top-2 sparse-model confirmation before general claims.
- Full plan: `docs/NEXT_EXPERIMENTS.md`.

### `h5-first-order` — 2026-08-01

- Hypothesis: H5 — predictor quality and cold-path resources jointly define an
  analytically profitable region, and an existing policy may recover a useful
  share of that region.
- Config: `configs/experiment/h5-first-order.toml`; preregistration in
  `docs/H5_PROTOCOL.md`.
- Inputs: immutable H1 decode trace, H4 calibration, and existing H2/H3 split,
  transition statistics, features, and fixed linear heads. No inference,
  training, model change, or inference-library change.
- Controlled result: 22,618/68,175 assumption cells pass the frozen ≥25%
  modeled stall-reduction, ≥50% oracle-recovery, and ≤2× transfer-amplification
  screen. Nonempty inverse requirements range from 25% to 50% complete
  cold-set coverage.
- Existing-policy result: none of eight representative held-out placements
  passes. K=32 policies cover 67–81% of complete cold sets but require
  6.3–6.7× transferred candidate bytes per useful cold byte. K=16, Δ=9
  reaches only 28.9% transition and 42.4% linear complete cold-set coverage,
  with 3.8× and 3.4× amplification.
- Interpretation: H5 supports a first-order co-design opportunity but rejects
  treating the raw K-wide prediction as a transfer list. The limiting
  mechanism is selective admission, not more predictor capacity.
- Figures: categorical profitability diagram and inverse prediction
  requirement curve generated as PDF and 450-DPI PNG with hashed inputs under
  `analysis/h5/figures`.
- Human visual review: pending.
- One next action: after review, compare one simple prediction-guided
  admission/residency score against reactive and static/domain baselines on
  the same artifacts. Keep H7, C1, predictor tuning, and new inference
  deferred.

### `h5-admission` — 2026-08-01

- Hypothesis: post-hoc H5 mechanism diagnosis; formal H5 gate unchanged.
- Question: do existing scores separate useful cold experts from useless
  nonresident candidates enough to reach A≤2× without collapsing complete
  cold-set coverage?
- Config/protocol: `configs/experiment/h5-admission.toml` and
  `docs/H5_ADMISSION_PROTOCOL.md`.
- Inputs: unchanged transition tables, fixed linear heads, 96/32 split, and
  K=32 LRU residency. All 64 target expert IDs are scored; no inference or
  retraining.
- Separation: useful-versus-useless AUROC is 0.850/0.803 for transition and
  0.883/0.861 for linear at Δ=3/9.
- Boundary: at A≤2×, linear preserves only 28.4%/22.8% complete cold-set
  coverage. Reaching C≥50% requires A=3.0×/3.3×. Transition requires
  A=4.0×/5.0× and has no 2× crossing at Δ=9.
- Decision: the ranking contains substantial signal but one scalar threshold
  is insufficient. This is evidence for a targeted cost-sensitive admission
  calibrator/head, not a generic future-expert MLP accuracy sweep.
- Figures: admission frontier and useful/useless score distributions generated
  as PDF and 450-DPI PNG with hashed inputs under
  `analysis/h5/admission/figures`.
- Human visual review: pending.
- One next action at the time: test placement value before predictor
  escalation. The subsequent H6 residency result supersedes the proposed
  cost-sensitive admission model.

### `h6-residency` — 2026-08-01

- Hypothesis: H6 — existing trajectory prediction improves on-demand expert
  residency at equal fast-tier capacity and runtime movement budget.
- Config/protocol: `configs/experiment/h6-residency.toml` and
  `docs/H6_PROTOCOL.md`.
- Inputs: existing H3 routes/features, preserved 96/32 split, frozen
  transition tables and linear heads, and the measured exact 12 MiB expert
  size. No inference, predictor training, model download, or library change.
- Policies: static popularity, domain popularity, reactive LRU,
  transition-guided residency, linear-guided residency, and an equal-budget
  exact-next-use oracle. Prediction may admit only an actually demanded miss;
  one insertion is allowed per target-layer wave.
- Formal result: `PILOT_DOES_NOT_SUPPORT`. At decode K=16, Δ=3, transition and
  linear are 3.9/2.5 pp worse in expert-stall reduction and 0.7/0.6 pp worse
  in complete-set hits than the strongest matched static/domain/LRU baseline.
  Only 5/52 and 3/52 layer-domain scopes improve on both metrics; no domain or
  layer is positive on average.
- Primary metrics: residual cold demand is 48.1% for LRU, 50.2% transition,
  48.8% linear, and 31.2% oracle. Complete resident-set hits are 2.8%, 3.4%,
  3.5%, and 11.8%. Later-use movement efficiency is 69.7%, 60.0%, 66.3%, and
  94.7%.
- Broad scan: no domain-balanced decode cell improves both headline metrics at
  K=16 or K=32. Prefill contains a few weak middle-layer cells, but none gains
  at least 2 pp on both metrics.
- Interpretation: same-token prediction down network depth is not a temporal
  reuse predictor across tokens. The oracle gap shows residency has headroom;
  the existing information is misaligned with this mechanism.
- Figure: one triangular source-layer × lookahead heatmap for K=8/16/32,
  saved as PDF and 450-DPI PNG with hashed inputs under
  `analysis/h6/figures`.
- Human visual review: completed through the subsequent checkerboard/oracle
  interpretation review; the negative gate stands.
- One next action: review the H6 heatmap, then close this placement mechanism.
  Do not fit an admission head, tune an MLP, start H7, collect confirmation,
  or download a second model without a new explicitly approved hypothesis.

### `c0-base-instruct-trajectories` — 2026-08-01

- Hypothesis: C0 — Base→Instruct post-training materially changes matched-token
  routing-trajectory predictability.
- Config/protocol: `configs/experiment/c0-posttraining-trajectory.toml` and
  `docs/C0_PROTOCOL.md`; endpoint runs are
  `olmoe-base-c0-paired` and `olmoe-instruct-c0-paired`.
- Checkpoints: Base revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`
  and Instruct revision `caada7d7b70f4b852b14108479e0812223a8794f`.
- Inputs: the same 128 standard-small prompts under forced raw serialization
  and one prefill forward per request. All 13,918 input tokens and 222,688
  token-layer records match exactly across checkpoints.
- Split: preserved 96/32 request split, 24/8 per domain. Popularity and
  transition tables use train requests only.
- Formal result:
  `PILOT_DOES_NOT_SUPPORT_POSTTRAINING_PREDICTABILITY_EFFECT`. At prefill
  layer 0→15 and K=16, transition-over-static selection gain is +11.0 pp for
  Base and +12.6 pp for Instruct. The +1.6 pp change is below the frozen 5 pp
  threshold.
- Trajectory result: 89.7% selection agreement, 82.6% route Jaccard, 35.7%
  exact top-8 equality, 84.6% K=16 hot-set Jaccard, and 0.0029 nats
  popularity JS divergence across held-out layers/domains.
- Policy transfer: cross-checkpoint transition penalties are generally below
  one point through Δ=12; the symmetric primary layer-0→15 penalty is 1.6 pp.
- Interpretation: structured routing already exists in Base. SFT+DPO+RLVR make
  frequent small expert substitutions while preserving the trajectory
  scaffold, skew, and predictability.
- Figures: fixed-source predictability and matched-route agreement saved as
  PDF/450-DPI PNG with hashed inputs under
  `artifacts/runs/olmoe-c0-base-instruct/analysis/c0/figures`.
- Programmatic visual inspection: completed; axes, units, fixed-source
  semantics, and headline values agree. Researcher review is pending.
- Decision: do not download SFT or DPO after the endpoint gate failure.
- One next action: human review of the two C0 figures; retain C1 sparse-model
  transfer as the higher-value future confirmation requiring explicit
  permission.

### `ax-future-predictor-architecture` — 2026-08-01

- Track: AX — assumption-driven predictive expert-memory architecture
  exploration.
- Status: `COMPLETE_PENDING_HUMAN_REVIEW`; AX1–AX3 ran without new inference
  or training.
- Protocol/config:
  `docs/ARCHITECTURE_EXPLORATION_PROTOCOL.md` and
  `configs/experiment/ax-future-predictor-architecture.toml`.
- Objective: derive quantitative capacity, bandwidth, latency, reliability,
  granularity, and SLO regions for a future MTP-style routing gate. The goal is
  architectural exploration, not wall-clock benefit on current OLMoE.
- Evidence contract: keep measured calibration, trace-derived demand, assumed
  future-predictor quality, and hypothetical hardware inputs separate in every
  output.
- Predictor sweep: wave-complete cold-set coverage 50–99.9% and
  predicted/useful byte amplification 1–8×, with correlated wave misses.
- Anchor integrity: the 12 MiB transfer fit agrees within 0.000006 ms and the
  archived H5 K=16/32 headroom cells reproduce exactly.
- AX1 result: at measured PCIe and assumed C=99%, A=1.5×, the best wave-local
  projections improve P99 TPOT by 33.8%, 35.3%, and 39.3% versus reactive
  offload at K=8/16/32. Absolute P99 is 48.03/43.27/30.71 ms versus the
  10.23 ms all-resident measured reference.
- Tail sensitivity: selected FCFS queue replay gives 65.23 ms stall at K=16,
  Δ=9, C=99%, A=1.5× versus 33.04 ms wave-local. At 64 GB/s, K=16, Δ=3, the
  corresponding queue P99 falls to 20.91 ms but remains nontrivial.
- AX2 result: for trace-derived K=16 demand, whole experts, A=1×, and one lane,
  minimum mean bandwidth is 71.3/22.8/11.6/8.2 GB/s at Δ=1/3/6/9.
  Lookahead and amplification act approximately as reciprocal/linear
  bandwidth levers; complete-set coverage independently controls cold-path
  tail incidence.
- AX3 result: a top-8 layer requires 96 MiB of whole-expert payload. Rolling
  double buffering needs 192 MiB at A=1× and 384 MiB at A=2×. The frozen
  SRAM range has 1,429/7,200 physically feasible factorized cells; A=4× and
  A=8× have no capacity-feasible whole-expert cell at ≤512 MiB.
- Figures: profitability phase map, memory–P99 Pareto, and inverse
  bandwidth/lookahead curve saved as PDF/450-DPI PNG with hashed inputs under
  `artifacts/runs/h1-standard-small/analysis/architecture/figures`.
- Claim boundary: CPU/pooled-memory prefetch may enable larger models and beat
  reactive offload, but cannot be called faster than an otherwise identical
  all-HBM model. Future-router sweep points are projections, not evidence that
  the current checkpoint achieves them.
- Interpretation: a quantitative co-design window exists, but queueing,
  complete-wave reliability, and false-positive occupancy are architectural
  requirements rather than predictor footnotes. Whole-expert SRAM is
  capacity-limited for top-8 routing; top-1/top-2 or selective sub-expert
  staging would move the bound more than a small link-speed increase.
- Human review: pending.
- One next action: review the three figures and select at most one optional
  live calibration point only if it would change the conclusion.

### `ax4-deadline-degradation` — 2026-08-01

- Track: AX4 — deadline-bounded graceful expert degradation.
- Status: `COMPLETE_PENDING_HUMAN_REVIEW`; formal analytical gate passed.
- Protocol/config: `docs/DEADLINE_DEGRADATION_PROTOCOL.md` and
  `configs/experiment/ax4-deadline-degradation.toml`.
- Decisive question: can a fixed layer commit deadline remove cold-transfer
  waiting from low-batch TPOT while retaining a plausible P99 missing-routed-
  mass contract for future availability-trained models?
- Inputs: 128,176 retained decode layer-waves with selected IDs/weights, AX1
  queue/residency state, and H4 timing. No new inference, training, model, or
  library change.
- Weight semantics: OLMoE softmaxes over all 64 experts, takes top-8, and does
  not renormalize. Raw selected weights sum to 0.406 on average (P99 0.717).
  Primary missing mass is normalized within top-8 and is explicitly a future
  architecture/training contract.
- Policies: reactive exact, predictive exact-wait, deadline null,
  deadline-renormalized, deadline shared-residual, and mass-priority oracle.
- Replay: 756 factorized wave-local scenarios, 2,304 physical-bound cells,
  and 19 trace-ordered FCFS boundary candidates; every deadline policy has
  exactly zero post-commit transfer wait.
- Formal gate: at least 50% expert offload, ≥25% P99 TPOT improvement over
  reactive exact, ≤1.5× all-local TPOT with fallback, P99 missing mass ≤20%,
  and full fallback ≤1% across at least two domains and two layer bands.
- Formal result: `SUPPORTED_ANALYTICALLY_UNDER_ASSUMPTIONS`. K=8 at 256 GB/s,
  C=99%, A=1.5×, Δ=1, and one-layer slack passes with only 1.5 GiB resident,
  10.5 GiB offloaded, 11.25 ms bounded TPOT, 88.9 token/s, 0.93% degraded
  waves, effectively zero P99 wave missing mass, and zero full fallback. The
  same-hierarchy reactive P99 is 16.41 ms, so improvement is 31.4%/1.46×.
- Boundary: K=32 at 128 GB/s passes with 7.2% P99 missing mass. K=16 at
  128 GB/s narrowly misses at 20.4%. Measured 24.14 GB/s PCIe fails at every
  capacity with 100%/100%/81% P99 missing mass for K=8/16/32.
- Frozen headline reproduced: 10.23 ms all-local plus 10% overhead gives
  11.25 ms and 88.9 token/s, 83.2% lower TPOT and 5.94× throughput versus the
  earlier 66.83 ms K=16 reactive-PCIe projection. This is not the
  same-hardware comparison used by the formal gate.
- Large-model context: 9,216 labeled geometry projections span 64–512
  experts/layer, top-1/2/4/8, 32–96 MoE layers, 4–64 MiB objects, and batch
  1/2/4/8. They are sensitivities, not cross-model evidence.
- HW proposal: always-resident fallback plane, optional residual-expert plane,
  deadline-aware DMA, atomic commit bitmap, bounded renormalizer, speculative
  traffic isolation, and missing-mass/fallback telemetry.
- Figures: latency–quality frontier, capacity–throughput–degradation Pareto,
  and FCFS bandwidth/erasure phase map are retained as PDF/450-DPI PNG with
  hashed inputs under
  `artifacts/runs/h1-standard-small/analysis/ax4_deadline_degradation/figures`.
- Claim boundary: passing identifies an erasure-robustness target worth
  training for. It cannot show current-model quality under missing experts.
- One next action: human review. Do not start training, new inference, or a
  new checkpoint without explicit permission.

### `mi355x-olmoe-parity` — 2026-08-01

- Milestone: AMD-A — qualify pinned OLMoE and its hook-visible routing path on
  one MI355X, then compare derived trajectory trends with retained NVIDIA
  artifacts.
- Scope amendment: NVIDIA request-level traces were not committed and cannot
  be restored. Record-level selected-ID parity and trace interchangeability
  are therefore untestable. The comparison uses the nested 16-request MI355X
  prefix versus preserved 128-request NVIDIA H1/H2 CSV/JSON outputs.
- Platform: one visible AMD Instinct MI355X, `gfx950:sramecc+:xnack-`, ROCm
  7.2, PyTorch `2.11.0+rocm7.2`, Transformers 5.14.1, BF16 SDPA.
- Model: pinned Instruct revision
  `caada7d7b70f4b852b14108479e0812223a8794f`; inspection reproduces 16
  routers, 64 experts/layer, top-8, and 12,582,912 bytes/expert.
- Integrity: all 16 requests completed with one prefill forward each; 26,752
  routed token-layer records, 16/16 input hashes, zero router mismatches, zero
  missing/extra router-call counts, and complete 16-layer trace integrity.
- H1: both runs fail the model-wide prefill hot-tier gate. MI355X mean top-8
  coverage is 22.37% versus 25.17% NVIDIA. Layerwise top-8 Pearson/Spearman is
  0.839/0.841; Gini is 0.947/0.959. Mean top-8 expert intersection is 4.94/8
  with Jaccard 0.456.
- H2: transition-over-static selection-gain curves are nearly identical over
  Δ=1..15 (Pearson/Spearman 0.999/1.000; 0.86 pp mean absolute difference).
  Complete-route-gain correlation is 0.996/1.000. MI355X passes the
  descriptive gate through Δ=14; Δ=15 misses only the 2% complete-gain
  threshold at 1.967% on the four-request test set.
- Interpretation: MI355X Transformers execution and router instrumentation are
  qualified, and the historical skew/trajectory conclusions are
  qualitatively retained. No observed difference can be attributed solely to
  platform because request counts and H2 splits differ.
- Figure: one two-panel H1 layer/H2 horizon comparison retained as PDF and
  450-DPI PNG with hashed inputs under
  `analysis/platform_comparison/figures/`; programmatic and visual inspection
  completed.
- Result: supports advancing to isolated MI355X H4 calibration after human
  review, while keeping all routing traces platform-scoped. Any replay of the
  old NVIDIA demand trace with AMD timing is a counterfactual sensitivity, not
  a measured AMD workload.
- One next action: researcher review of the comparison figure and result. Do
  not start H4 calibration or GPT-OSS qualification first.

### `mi355x-olmoe-instruct-c0-paired` — 2026-08-01

- Milestone: AMD-A matched-workload confirmation, requested after the
  16-request comparison showed a visible H1 absolute-skew offset.
- Collection config: exact counterpart of
  `c0-olmoe-instruct-collect.toml`; only run identity/output and platform
  metadata differ. All 128 revision-pinned prompts use raw serialization,
  384-token truncation, greedy one-token generation, and BF16 SDPA.
- H2 config: exact counterpart of `c0-olmoe-instruct-h2.toml`; same split
  seed, 8 held-out requests/domain, 96/32 split, K=8/16/32, and Δ=1..15.
- Scope integrity: request keys/order, prompt SHA-256, collection settings,
  and H2 split match. All 126 NVIDIA manifest token hashes that remain
  available match MI355X; two older already-complete entries lack hashes.
- Run integrity: 128/128 requests, 222,688 prefill token-layer records,
  128/128 MI355X input hashes, zero router mismatches, zero bad router-call
  counts, and exact inspected model geometry.
- H1 result: the visible 16-request offset disappears. Mean top-8 coverage is
  25.1758% MI355X versus 25.1746% NVIDIA; layerwise MAE is 0.0134 pp, maximum
  difference 0.0305 pp, and Pearson/Spearman 0.999962/1.0. Mean Gini differs
  by −0.000017. Fourteen layers have identical top-8 hot sets; layers 5 and
  14 differ only at rank eight, giving mean intersection 7.875/8 and Jaccard
  0.9722.
- H2 result: selection-gain Pearson/Spearman is 0.999989/1.0 with 0.0402 pp
  MAE; complete-route gain is 0.999922/1.0 with 0.1008 pp MAE. Every horizon
  Δ=1..15 passes on both platforms.
- Interpretation: the earlier H1 discrepancy was an incomplete-workload
  artifact. Aggregate OLMoE skew and trajectory statistics reproduce nearly
  exactly across the NVIDIA/CUDA and AMD/ROCm execution paths. Raw selected-ID
  parity remains unavailable because NVIDIA trace shards were not retained.
- Figure: overlaid H1/H2 curves plus MI355X−NVIDIA residual panels, retained
  as PDF/450-DPI PNG with hashed inputs under the matched run's
  `analysis/platform_comparison/figures/`; visual inspection completed.
- Result: supports advancing to isolated MI355X H4 calibration after human
  review. Do not call the old NVIDIA trace measured AMD demand; use it only as
  a counterfactual hardware sensitivity until a new MI355X decode trace exists.

### `mi355x-h4-calibration` — 2026-08-01

- Milestone: AMD-B — hook-free MI355X timing calibration plus H4 oracle replay
  on a new, platform-scoped standard decode trace.
- Demand run: `mi355x-h4-decode`, matching the 128-request, 384-prompt-token,
  greedy 64-token H1 workload. It contains 8,012 decode tokens, 128,192 decode
  layer-waves, and zero router validation mismatches.
- Platform: one visible AMD Instinct MI355X, `gfx950`, ROCm 7.2, PyTorch 2.11,
  and the pinned BF16 OLMoE Instruct checkpoint. Timing installed no hooks.
- Calibration: 15.638 ms median cached-token forward, 0.977 ms effective
  inter-MoE interval, 0.285 ms exact 12 MiB H2D event time, 0.300 ms wall
  completion, and 44.69 GB/s fitted pinned H2D bandwidth.
- RTX comparison: the measured MI testbed forward is 52.9% slower, while the
  exact copy is 45.6% faster and fitted bandwidth is 85.2% higher. Given the
  hardware capability, the forward gap is treated as a software-stack or
  small-batch kernel artifact, not an inherent MI355X property. Copies per
  effective layer interval increase from 1.22 to 3.43.
- Formal result: `PILOT_SUPPORTS`. At K=16, delta 1/2/3, on-time cold bytes are
  68.8/78.7/83.9% and oracle stall reduction is 74.8/82.7/86.5%; every frozen
  short horizon passes both 50% thresholds.
- Attribution: RTX timing on MI demand reproduces the old 32.8%/38.9% delta-3
  result. The faster MI copy alone passes delta 2 and 3; the extra measured
  testbed forward slack alone does not pass but expands the combined regime.
- Boundary: this is a physical oracle ceiling, not predictor sufficiency,
  concurrent overlap validation, execution from copied state, or end-to-end
  speedup. Delta 3 still leaves 20.4% of waves with some modeled stall.
- Figures: a plain-language full-grid map and a two-metric measured-link trend
  chart retain every capacity/lookahead point as PDF/450-DPI PNG with hashed
  inputs. Domain jargon was removed and the frozen decision region is marked.
- Human visual review: completed 2026-08-01 with requested revisions. The
  researcher accepted the regime-space purpose, classified the slower forward
  as a likely Transformers/PyTorch/ROCm testbed artifact rather than inherent
  MI355X behavior, and requested simpler figures.
- Review revision: the original dense views were replaced by a plain-language
  63-cell grid and a two-panel measured-link chart. All capacities,
  bandwidths, lookaheads, and both gate metrics remain visible; no relevant
  trend was averaged away.
- Decision: Milestone B is complete with a narrowed platform-stack claim.
  Advance the queue to Milestone C planning, but do not infer wall-clock
  benefit or begin GPT-OSS inference from the oracle result.
- One next action: freeze and execute the smallest GPT-OSS Transformers
  router/dispatch qualification, then stop for review before the 20B tracer
  bullet.

### `gpt-oss-20b-milestone-c` — 2026-08-01

- Milestone: C — GPT-OSS Transformers router/dispatch qualification, not a
  routing-distribution experiment.
- Provenance: `openai/gpt-oss-20b` and bundled tokenizer at revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`; Transformers 5.14.1,
  PyTorch 2.11.0+rocm7.2, kernels 0.15.2, and GPT-OSS Triton-kernel snapshot
  `9655fcf7d0f638bec4a82f6f1a70014f0aa8cfb0` on one MI355X.
- Path result: native MXFP4 replaces the MLP forward, computes router logits
  inline, and bypasses `GptOssTopKRouter.forward`. Ordinary router hooks fire
  zero times. Expert-dispatch pre-hooks observe the exact `RoutingData` and
  gather/scatter tensors passed to the two fused grouped GEMMs.
- Integrity: 24/24 routed layers covered; 6 tokens × 24 layers × top-4 = 576
  consumed ID/weight pairs; zero ID mismatches, zero weight mismatches at
  `1e-6`, and maximum absolute weight error 0.0.
- Semantics: 32 experts/layer, top-4, no shared expert, selected-top-k softmax,
  descending-logit router order, ascending-ID then expert-major dispatch order,
  and BF16 compute inputs.
- Storage: 13,236,480 serialized bytes/expert and 13,253,760 loaded
  bytes/expert. The 17,280-byte difference is exactly two BF16 bias vectors
  materialized in FP32.
- Decision: `QUALIFIED`. The covered MXFP4 bypass is admissible for later
  tracing. This establishes no routing distribution, model output, timing, or
  120B claim.
- Figure: one two-panel coverage/parity figure retained as PDF and 450-DPI PNG
  with hashed inputs/outputs under `analysis`-independent run `figures/`.
- One next action: researcher reviews the report and figure. Do not begin
  Milestone D without explicit approval.

### `gpt-oss-20b-milestone-d` — 2026-08-01

- Milestone: D — small end-to-end GPT-OSS 20B tracer bullet, not the
  routing-distribution comparison.
- Authorization: the researcher explicitly requested Milestone D, satisfying
  the Milestone C review/advance stop condition.
- Frozen workload: two local prompts, checkpoint chat template with fixed
  current date, batch one, greedy eight-token output, and a terminal cached
  forward so every retained output token has routing coverage. The complete
  workload was repeated once without reloading the model.
- Provenance: the same pinned 20B checkpoint, Transformers 5.14.1, native
  MXFP4 path, PyTorch 2.11.0+rocm7.2, and one visible MI355X used by Milestone
  C. The run definition hashes the config, checkpoint config, tokenizer, and
  safetensors index and links the qualified Milestone C evidence.
- Coverage: 161 prompt plus 16 generated tokens; 4,248/4,248 token-layer
  records across 24 layers; zero missing, unexpected, duplicate, bad-phase,
  or bad-top-k records.
- Dispatch integrity: 16,992 consumed `(expert_id, weight)` pairs; zero ID or
  selected-weight mismatches versus independent router calculation and 0.0
  maximum absolute selected-weight error.
- Determinism: repeat input and generated token IDs match exactly; all route
  IDs and selected weights match with 0.0 maximum absolute error.
- Outputs: both eight-token generations are retained but end inside the model's
  analysis channel. They qualify output capture and determinism, not answer
  quality.
- Artifacts: two standard trace shards, inspection, output JSONL, integrity
  JSON/CSV, compact phase/layer routing table, two PDF/450-DPI PNG figures,
  report, and a SHA-256 manifest.
- Decision: `QUALIFIED`. The end-to-end GPT-OSS 20B tracing workflow is
  reproducible and complete. No routing-distribution, domain, performance,
  quality, or 120B claim follows from this convenience workload.
- One next action: researcher reviews Milestone D. Do not begin Milestone E
  without explicit approval.

### `gpt-oss-20b-milestone-e` — 2026-08-01

- Milestone: E — request-held-out GPT-OSS 20B route-prediction quality. The
  planned 120B comparison was cancelled because available disk is
  insufficient; no 120B result is implied.
- Frozen design: all 128 revision-pinned standard prompts, 32 per domain;
  greedy batch-one inference; 16 traced decode tokens; 96/32 stratified
  train/test split; K=4/8/16; all Δ=1--23; prefill/decode separated.
- Collection: 22,152 prompt plus 2,048 decode tokens; 580,800/580,800 complete
  token-layer records; 2,323,200 exact dispatch-consumed ID/weight pairs.
- Frozen trace gate: `FAILED`. Every executed expert ID matches, but 6
  independently reconstructed weights exceed `1e-6`; maximum absolute error
  is 0.001953125. These sparse BF16-sized deviations are consistent
  with native fused Triton versus independent PyTorch softmax numerics. The
  threshold is not weakened post hoc.
- Analysis status: conditional post-hoc evidence is admissible for set metrics
  because IDs are exact and for routed-mass metrics because stored weights are
  the dispatch-consumed values.
- Primary prediction result at decode K=8: transition selection coverage is
  86.3%/85.5%/84.4% at Δ=1/2/3, gains of +18.2/+16.7/+15.5 pp over the
  stronger of domain popularity and route copy. Complete-route gains are
  +32.3/+30.0/+26.9 pp. All request-bootstrap intervals exclude zero and all
  four domains improve; 3/3 gate points pass.
- Candidate-count qualification: K=4 achieves 63--67% selection but only
  18--20% complete-route coverage; K=16 reaches 85--87% complete routes with
  a candidate set spanning 50% of the expert namespace and 4× candidate
  amplification. K is not resident capacity; no residency state is modeled.
- Overall decision: `CONDITIONAL_PILOT_SUPPORT_WITH_TRACE_WEIGHT_EXCEPTION`.
  This is one-model internal route-prediction evidence, not language quality,
  timing, prefetch profitability, or a controlled cross-model comparison.
- Artifacts: protocol/config, 128 trace shards, exact outputs and integrity
  tables, split and scope/request/horizon/bootstrap tables, three PDF/450-DPI
  figures, reports, and hashed manifest under
  `artifacts/runs/gpt-oss-20b-milestone-e/`.
- One next action: researcher reviews the result and chooses a bounded 20B
  resource-contract replay or closes the GPT-OSS track.
