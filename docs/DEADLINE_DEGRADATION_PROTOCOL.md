# AX4: Deadline-bounded graceful expert degradation

**Status:** complete; formal analytical gate passed, human review pending  
**Track:** assumption-driven WL/SW/HW co-design  
**Immediate experiment:** analysis-only replay of retained weighted decode routes  
**New inference, training, model download, or library modification:** none  
**Primary config:** `configs/experiment/ax4-deadline-degradation.toml`

**Canonical result:**
`artifacts/runs/h1-standard-small/analysis/ax4_deadline_degradation/REPORT.md`

## Decisive question

Can a low-batch MoE decode system enforce a tight token deadline by refusing
to wait for late expert weights, while keeping the resulting missing routed
mass small enough to define a plausible robustness target for future
availability-trained models?

AX4 changes the failure contract established by AX1–AX3:

- exact execution stalls when any demanded cold expert is late;
- deadline-degraded execution commits at a fixed deadline, runs every
  available contribution, and replaces or omits late contributions;
- prediction and transfer quality affect output fidelity, but cannot extend
  the committed transfer deadline.

The intended setting is batch-1 through batch-4 autoregressive decode for
large models with many sparsely routed experts and a strict TPOT SLO. Batch 8
is retained only as a sensitivity. The architectural objective is to increase
admitted model capacity or throughput relative to reactive offload at the same
HBM capacity—not to claim exact equivalence with all-resident execution.

## Evidence and claim boundary

Every result must retain the AX evidence labels:

1. **Measured:** OLMoE expert size, hook-free decode timing, and pinned-host
   transfer calibration.
2. **Trace-derived:** selected expert IDs, selected expert weights, cold
   demand, layer/domain order, and queue burstiness.
3. **Assumed predictor:** future-route coverage, candidate amplification,
   confidence or importance ordering, and correlated misses.
4. **Assumed robustness:** the model tolerates bounded missing routed mass
   after availability-conditioned training.
5. **Hypothetical hardware:** deadline enforcement, transfer isolation,
   fallback capacity, bandwidth, concurrency, and larger-model geometry.

The immediate experiment can establish latency, capacity, traffic, and
missing-mass contracts. It cannot establish language quality, loss
preservation, or that current OLMoE tolerates expert erasure. Those require a
later training/inference experiment and explicit permission.

## Definitions

For token or synchronous batch wave \(t\) at MoE layer \(l\):

- \(S_{t,l}\): experts selected by the ordinary router;
- \(a_i\): selected-expert contribution weight normalized over \(S_{t,l}\);
- \(R_{t,l}\): experts already resident;
- \(F_{t,l}(d)\): predicted transfers completed by commit deadline \(d\);
- \(A_{t,l}=S_{t,l}\cap(R_{t,l}\cup F_{t,l}(d))\): available selected experts;
- \(M_{t,l}=S_{t,l}\setminus A_{t,l}\): missing selected experts;
- delivered routed mass
  \(q_{t,l}=\sum_{i\in A_{t,l}}a_i\);
- missing routed mass
  \(m_{t,l}=1-q_{t,l}\);
- full-fallback wave: \(q_{t,l}=0\);
- deadline erasure: a selected expert omitted because it was not available at
  commit, regardless of whether its transfer later completes.

The recorded OLMoE selected weights do not sum to one in every trace record.
Before replay, reproduce the model's actual downstream weighting semantics.
The primary architecture metric then normalizes within the selected top-k so
that \(m\in[0,1]\) is interpretable as a fraction of routed contribution.
Preserve raw-weight results as a secondary integrity check.

## Degradation policies

Evaluate only three simple policies:

1. **Null residual:** omit missing contributions without renormalization.
2. **Present renormalization:** renormalize available selected weights; if
   delivered mass is below the frozen floor, use the fallback-only path.
3. **Shared residual fallback:** compute an always-resident shared expert once
   and treat routed experts as optional residual refinements.

The preferred future model form is:

\[
y_l = F_l(x) + \sum_{i\in S_l} a_i \Delta E_{l,i}(x).
\]

\(F_l\) is always resident and provisioned for the full batch. Late
\(\Delta E_i\) contributions are erased at commit. Do not model one shared
expert invocation per missing expert.

Identity or null fallback provides the strongest timing bound. Shared fallback
is expected to provide the better quality contract but must be replicated or
otherwise provisioned for the worst case in which every token takes it.

## Latency and throughput bounds

With a nonblocking commit deadline and isolated speculative traffic:

\[
T_{\mathrm{step}} \le
T_{\mathrm{fixed}} +
\sum_l
\left(
T_{\mathrm{router},l}+
T_{\mathrm{fallback},l}+
T_{\mathrm{optional},l}(k_{\max})+
T_{\mathrm{merge},l}+
T_{\mathrm{scheduler},l}
\right).
\]

Cold-transfer completion time is intentionally absent. A transfer that misses
commit cannot delay dispatch.

This is a hard bound only when local compute, fallback throughput, metadata
work, and memory/DMA interference have worst-case reservations. The immediate
study reports:

- a trace-calibrated P99 bound using measured local timing;
- a deterministic analytical bound parameterized by explicit local-compute
  and fallback allowances.

For a synchronous batch producing \(B\) tokens per decode step:

\[
\mathrm{throughput}_{\min} =
\frac{1000B}{T_{\mathrm{step,bound}}}
\quad\text{tokens/s}.
\]

This is a projection until batch-dependent local compute is measured. Report
batch 1 as primary, batch 2/4 as low-batch sensitivities, and batch 8
separately.

Speculative transfers must use bounded credits or a QoS-isolated channel.
Otherwise late false-positive traffic may contend with local expert compute
and invalidate the TPOT bound.

## Output-perturbation bounds

Let \(\|E_i(x)\|\le B_l\). Then:

- null residual:

  \[
  \|\hat y_l-y_l\|\le m_lB_l;
  \]

- present renormalization, when \(q_l>0\):

  \[
  \|\hat y_l-y_l\|\le 2m_lB_l;
  \]

- shared fallback with
  \(D_l=\max_i\|E_i(x)-F_l(x)\|\):

  \[
  \|\hat y_l-y_l\|\le m_lD_l.
  \]

For shared-plus-residual experts, the tighter form is:

\[
\|\hat y_l-y_l\|
\le
\sum_{i\in M_l}a_i\|\Delta E_{l,i}(x)\|.
\]

AX4 reports normalized bounds \(m\), \(2m\), and an assumed \(mD/B\) sweep.
It does not translate them into perplexity or task accuracy. End-to-end
Lipschitz composition may be reported as a mathematical upper bound, but is
expected to be too loose for the headline.

## Preregistered predictions

### P1 — Deadline conversion

A hard commit policy removes transfer queue length from the critical-path TPOT
bound. Compared with the same reactive hierarchy, P99 TPOT should improve
substantially wherever AX1 observed large cold stalls.

The frozen current-testbed prediction is deliberately concrete. The measured
all-local decode anchor is 10.23 ms. If fallback, commit, and merging add 10%,
the deadline bound is 11.25 ms, or 88.9 batch-1 tokens/s. This is 83.2% below
the 66.83 ms K=16 reactive-offload P99 projection. It is also 85.1% below the
75.46 ms selected K=16, Δ=9 predictive FCFS projection. These are
preregistered analytical predictions, not measured degraded execution.
At equal batch size, the corresponding throughput projection is 5.94× the
reactive-offload value of 15.0 tokens/s.

At a future 20 ms bounded TPOT, the ideal synchronous throughput projection is
50/100/200 tokens/s for batch 1/2/4. AX4 must report actual batch-scaling
assumptions beside these values.

### P2 — Mass coverage replaces exact-set coverage

Complete cold-set coverage will cease to be the decisive latency metric.
Several incomplete waves should retain high delivered routing mass because
the omitted expert has a small selected weight.

### P3 — Importance-aware admission

Scheduling predicted experts by expected routed mass or bounded distortion per
byte should dominate ID-count coverage at the same movement budget. This is
an oracle/sensitivity result until a future routing head emits calibrated
importance.

### P4 — Reliability becomes a quality contract

Correlated false negatives should no longer create an unbounded tail, but they
will create bursts of missing mass and fallback load. P99 missing mass—not
mean recall—will determine the training target.

### P5 — Routing sparsity changes the Pareto

Top-1/top-2 geometry should sharply reduce transfer volume and staging
capacity, but a single miss more often invokes full fallback. The net result
is a latency/quality trade rather than an unconditional advantage.

## AX4-A — Immediate weighted deadline replay

### Inputs

- retained H1 decode trace with selected IDs and weights;
- AX1 K=8/16/32 LRU cold sets and queue order;
- measured 12 MiB transfer curve and 10.23 ms OLMoE local-decode anchor;
- frozen AX future-predictor coverage/amplification profiles.

No routing trace, hidden state, model inference, or predictor training is
collected.

### Replay

For each token and layer:

1. establish residency and prediction issue time exactly as in AX1;
2. schedule candidates through the bounded transfer queue;
3. at the target commit deadline, freeze the available expert bitmap;
4. never wait for a late expert;
5. compute delivered and missing normalized routed mass;
6. charge fallback compute, renormalization, metadata, and any nonisolated
   traffic allowances explicitly;
7. aggregate layer values into token P50/P95/P99 and worst observed values.

Do not discard incomplete waves. They are the primary evidence.

### Baselines

Compare at identical HBM residency, cold-tier bandwidth, and movement budget:

1. reactive exact offload;
2. predictive exact offload that waits for all selected experts;
3. deadline-null;
4. deadline-renormalized;
5. deadline-shared-residual;
6. mass-priority oracle deadline scheduler;
7. all-resident timing/capacity reference.

The current transition/linear policies are optional descriptive markers and
must not delay the primary assumed-predictor envelope.

### Primary sweep

- decode, batch 1;
- K=8, 16, 32;
- lookahead \(\Delta=1,3,6,9\);
- measured 24.14 GB/s plus 64, 128, and 256 GB/s;
- wave-complete predictor coverage 90%, 95%, 99%, and 99.9%;
- predicted/useful amplification 1, 1.5, 2, and 4×;
- commit slack 0, 0.25, 0.5, and 1.0 local MoE-layer intervals;
- missing-mass thresholds
  \(\tau=0,0.05,0.10,0.20,0.40,1.0\);
- deterministic correlated miss seed 17.

Use factorized inverse analysis instead of a needlessly large full Cartesian
table. Run trace-ordered FCFS replay only for boundary and Pareto candidates.

### Primary metrics

- modeled and bounded P50/P95/P99 TPOT;
- batch-normalized tokens/s;
- transfer-induced stall after commit, which must be zero by construction;
- delivered and missing routed mass;
- \(P(m>\tau)\) for each frozen \(\tau\);
- full-fallback wave/token fraction;
- maximum consecutive degraded layers;
- useful, false, cancelled, and late bytes;
- fallback invocations and worst-case fallback throughput;
- HBM expert capacity retained and offloaded;
- normalized null, renormalization, and shared-residual perturbation bounds;
- improvement over reactive and exact-wait predictive offload.

Keep phase, domain, layer band, and request separate before aggregation.

### Primary architecture-candidate gate

AX4-A supports a **plausible degradation contract**, not model quality, only
if at least one configuration:

1. keeps at most half the experts per layer resident;
2. reduces modeled P99 TPOT by at least 25% versus reactive exact offload at
   identical K, bandwidth, and movement budget, equivalently improving
   same-batch throughput by at least 1.33×;
3. stays within 1.5× the all-local TPOT anchor after an explicit fallback
   allowance;
4. has P99 missing routed mass at most 20%;
5. has full-fallback waves at most 1%;
6. repeats across at least two domains and two layer bands.

Also report the complete Pareto frontier so a near miss is not hidden by the
gate. Passing does not show that language quality is acceptable; it establishes
a concrete erasure-robustness target worth training for.

If no configuration satisfies even \(\tau=0.40\) under mass-priority oracle
admission, stop: the current route-weight structure does not support this
mechanism at the studied hierarchy.

## AX4-B — Large sparse-model projection

Use AX4-A's delivered-mass distributions as a workload shape, then project
clearly labeled model geometries:

- 64/128/256/512 experts per layer;
- resident fractions 1/8, 1/4, and 1/2, corresponding to idealized 8×, 4×,
  and 2× expert-capacity expansion before fallback overhead;
- top-1/2/4/8 routing;
- 32/64/96 MoE layers;
- 4/12/32/64 MiB expert transfer objects;
- batch 1/2/4, with batch 8 shown separately;
- 10/20/30/50 ms TPOT targets;
- HBM, pooled/CXL, and host-class bandwidth sensitivities.

Truncating OLMoE's top-8 route to top-1/2/4 is a rank sensitivity, not evidence
for an actual sparse checkpoint. Larger expert counts change storage capacity
only; they do not create new demand statistics.

For each geometry report:

\[
\text{expert capacity} = LNS,
\qquad
\text{resident capacity} = LKS + \text{fallback bytes},
\]

bounded TPOT, projected tokens/s, missing-mass contract, fallback throughput,
and the maximum offloaded capacity satisfying the gate.

## Hardware architecture proposal

AX4 evaluates a **deadline-elastic expert execution engine**:

1. **Always-resident fallback plane.** Shared, identity, or null path has
   reserved compute and storage; shared experts are replicated for worst-case
   batch demand.
2. **Optional refinement plane.** Routed experts in HBM/SRAM contribute when
   available but cannot block commit.
3. **Deadline-aware DMA scheduler.** Each request carries expert/object ID,
   target layer, deadline, expected routed mass or distortion reduction,
   bytes, confidence, and cancellation generation.
4. **Commit bitmap.** At a fixed layer deadline hardware atomically publishes
   available contributions to dispatch.
5. **Bounded renormalizer/merger.** Present weights are merged under the
   selected policy with a fallback floor for zero delivered mass.
6. **Traffic isolation.** Speculative copies have bounded credits and cannot
   consume reserved local-compute or fallback bandwidth.
7. **Degradation telemetry.** Hardware exposes delivered mass, erased mass,
   late/cancelled bytes, fallback load, and deadline counters to the runtime
   and SLO controller.
8. **Two-timescale control.** Long-horizon heads plan pooled-memory→HBM
   residency; short-horizon heads schedule HBM→SRAM refinements; the ordinary
   router confirms final weights.

This resembles approximate computing with an explicit semantic erasure
budget. The runtime may tighten or relax \(\tau\) per request priority, TPOT
SLO, or current memory pressure.

## Required figures

Create at most three:

1. **Deadline quality–latency frontier:** P99 TPOT versus P99 missing routed
   mass for reactive, exact-wait, deadline, and oracle policies.
2. **Capacity–throughput–degradation Pareto:** HBM expert capacity/offloaded
   capacity versus low-batch tokens/s, colored by missing-mass contract.
3. **HW phase map:** bandwidth/headroom versus tolerated missing mass, with
   regions labeled exact, graceful, fallback-dominated, and infeasible.

Every figure must distinguish measured anchors, trace replay, assumed
robustness, and hypothetical large-model points.

## Deferred training validation

Only after AX4-A passes or identifies a near-boundary contract:

- train with correlated deadline-derived expert erasures;
- use full-model distillation/consistency plus degraded LM loss;
- optimize a tail objective such as CVaR, not only mean dropout loss;
- measure ordinary loss, downstream quality, load balance, fallback
  saturation, and exact-mode quality;
- compare null, renormalized, and shared-residual forms with one fixed recipe.

This is a new model mechanism and requires explicit researcher permission.
Until then, “the model will learn to tolerate misses” remains an assumption.

## Human review and stop rule

After the three AX4 figures:

- verify actual selected-weight semantics;
- verify transfer stall is exactly zero after commit;
- inspect whether the Pareto is broad or driven by one domain/layer;
- record the smallest plausible \(\tau\) contract;
- decide whether the next action is training validation, sparse-checkpoint
  confirmation, or stopping the mechanism.

Do not start training, download a model, or build live hardware calibration
before this review.
