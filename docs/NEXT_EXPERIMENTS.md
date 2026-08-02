# Next experiments: learned lookahead prediction and analytical regime placement

**Updated:** 2026-08-01  
**Current next action:** materialize the exact Milestone F training TOML, then
implement the frozen-data predictor from
[GPT_OSS_MILESTONE_F_PROTOCOL.md](GPT_OSS_MILESTONE_F_PROTOCOL.md); do not
collect fresh traces until its development gate is applied
**Publication spine:** demonstrate compact lookahead expert-demand prediction
on existing unmodified MoE models, then place the measured predictor frontier
in an analytical workload/memory-system regime map
**Operating rule:** the predictor estimates total future expert demand. A
conceptual software manager may consume its scored forecasts, but building a
cache manager is not a research deliverable. Keep prediction candidate count
K independent from resident capacity R; K/E is candidate-set fraction,
whereas R/E is resident fraction.

This plan separates four questions that should not be conflated:

1. Is cross-layer route structure present on more than one public MoE
   architecture?
2. Can a compact learned model convert current routes into scored future
   expert demand with a tunable coverage/amplification frontier?
3. Where does that empirical frontier fall under swept workload, capacity,
   bandwidth, latency, and lead-time assumptions?
4. Could future co-designed training improve predictability without harming
   quality or load balance? This final question is future work, not a claim of
   the current paper.

OLMoE H2 and GPT-OSS Milestone E answer the first question conditionally: both
show held-out cross-layer structure, but they do not establish universal MoE
behavior or a controlled cross-model effect size. Milestone F addresses the
second question. Milestone G adapts the completed AX/H5 analytical machinery
to the measured GPT-OSS frontier. Training a predictability objective is
retained only as a future co-design hypothesis.

## Experiment list

| ID | Question | New inference/training? | Status |
|---|---|---:|---|
| AMD-A | Does OLMoE retain its derived routing trends on MI355X? | Yes | Complete and reviewed; aggregate trends retained |
| AMD-B | Does measured MI355X timing pass the unchanged whole-expert H4 oracle gate? | Yes for demand and timing; replay analytical | Complete and reviewed; gate passes with narrowed testbed interpretation |
| AMD-C | Can Transformers expose GPT-OSS router outputs proven identical to actual dispatch? | Configuration/tiny path first; 20B only as needed | Qualified; reviewed and advanced |
| AMD-D | Does the qualified GPT-OSS 20B path produce a complete tracer-bullet artifact chain? | Yes | Qualified and reviewed |
| AMD-E | On held-out GPT-OSS 20B requests, do transition tables beat strong cheap route baselines? | Yes | Conditional support: 3/3 prediction gates pass; frozen trace-weight gate fails on 6/2,323,200 pairs; K is candidate count and residency is unmodeled |
| AMD-F | Can a compact learned route model preserve the GPT-OSS lookahead coverage/amplification frontier? | No for development; fresh confirmation only after pass | Protocol ready; exact training TOML and implementation are next |
| AMD-G | Under what workload and memory-system regimes is the measured GPT-OSS predictor analytically profitable? | No | Planned after F fixes the empirical frontier; no cache-manager implementation |
| AMD-120B | How does 120B routing change normalized contracts? | Would require a new checkpoint | Cancelled: insufficient disk; no result claimed |
| AX1 | What model-capacity and TPOT envelope can future predictive host/pooled-memory prefetch provide? | No | Complete; projected region exists, review pending |
| AX2 | How do bandwidth, latency, coverage, amplification, and transfer granularity divide the design space? | No | Complete; inverse bounds and phase map generated |
| AX3 | What local-HBM and rolling-SRAM capacities suit a multi-horizon three-tier hierarchy? | No | Complete; physical staging envelope generated |
| AX4 | Can hard-deadline expert erasure produce a tight low-batch TPOT bound with a plausible quality-robustness contract? | No for completed replay; later training requires permission | Complete; analytical gate passes only in a high-bandwidth mass-priority regime, review pending |
| H5-A | What prediction × hardware combinations create a first-order profitability window? | No | Complete; region exists |
| H5-B | What minimum predictor quality is required at each capacity, bandwidth, and lookahead? | No; derived from H5-A | Complete |
| H5-C | How much analytical oracle benefit do the existing transition and linear streams recover? | No retraining; reconstruct existing predictions | Complete; raw streams fail traffic gate |
| H5-D | Do existing scores separate useful from useless cold candidates? | No | Complete; signal present, shared threshold insufficient |
| H6 | Does prediction-guided on-demand residency beat static/domain/LRU at equal capacity and movement budget? | No | Complete; frozen gate failed |
| H7 | Can a routing-predictability objective improve the quality/resource frontier without harming loss or load balance? | Yes; model training | Future work outside the current paper; motivated, not tested |
| C0 | Does Base→Instruct post-training materially change matched-token trajectory predictability? | Yes; two endpoint traces | Complete; frozen stage-effect gate failed |
| C1 | Does the trajectory/co-design result transfer to one newer top-1/top-2 checkpoint? | Yes; one trace collection | Deferred; explicit permission required |

Detailed timing validation, concurrent-copy microbenchmarks, live cache work,
new model downloads, and predictability-aware base-model training remain
deferred. The fixed Milestone F route MLP is now active because it tests total
lookahead demand on an already qualified model; it is not an attempt to rescue
the H3 replacement gate or H6 residency policy.

## Completed GPT-OSS sequence -- Milestones C through E

### Decisive question

Can the pinned Hugging Face Transformers GPT-OSS implementation be instrumented
so the captured router IDs and weights are demonstrably the values consumed by
expert dispatch, including any MXFP4 or custom-kernel path?

This qualification comes before H1/H2 collection. A plausible-looking hook is
not enough: if the implementation reorders, filters, reconstructs, or bypasses
router outputs after the hooked module, the resulting trace is not valid
workload evidence.

### Minimum execution sequence

1. Pin exact Transformers, `gpt-oss-20b`, and tokenizer revisions.
2. Inspect configuration and source to map router output to the dispatch
   consumer; document the model-specific path rather than generalizing the
   OLMoE adapter.
3. Start with configuration-only or tiny synthetic execution. Use 20B weights
   only when required to exercise the real MXFP4/kernel path; do not download
   120B in this milestone.
4. Compare hook-captured selected IDs and weights with the tensors consumed by
   dispatch for every tested token and layer. Any mismatch or unobservable
   kernel bypass fails qualification.
5. Record expert tensor shapes, stored and loaded bytes, storage/compute dtype,
   shared experts, top-k ordering, selected-weight normalization, router count,
   and experts per layer.
6. Keep vLLM out of qualification because ordinary module visibility is the
   invariant under test.

### Exit gate and outputs

Pass only with zero hook-to-dispatch ID mismatches, complete router-call
coverage, pinned provenance, and explicit accounting for quantized/custom
kernel behavior. Produce one compact model-path report, an integrity table,
and a go/no-go decision. Stop there for review: Milestone D, not C, is the
small GPT-OSS 20B routing tracer bullet.

The value of Milestone C is epistemic rather than statistical. It prevents an
expensive 20B/120B collection from producing traces that look reasonable but
do not represent executed expert demand. Milestone D then exercised that
qualified path end to end: 4,248/4,248 token-layer records and all 16,992
dispatch-consumed pairs are complete and exact, and the immediate repeat is
identical. The outputs, trace shards, tables, figures, and hash manifest are
retained under `artifacts/runs/gpt-oss-20b-milestone-d/`. Milestone E then
used all 128 frozen standard requests for a 96/32 request-held-out prediction
test. At decode K=8, transition selection coverage is 86.3%/85.5%/84.4% at
Δ=1/2/3 and all three bootstrap gates pass against the stronger cheap
baseline. The overall result is conditional because six rare BF16-scale
independent weight deviations fail the frozen trace threshold; all executed
IDs match and exact dispatch weights are retained. See
[GPT_OSS_MILESTONE_E_RESULTS.md](GPT_OSS_MILESTONE_E_RESULTS.md).

AX4 is the immediate exception to the exact-execution contract, not to the
no-training rule. It asks whether late experts can be converted from a latency
failure into bounded routed-mass erasure. Current traces establish the
resource/erasure contract; they cannot establish model quality under erasure.

C0 adds a within-family control, not a new placement mechanism. Base and
Instruct preserve 89.7% of expert selections and differ by only +1.6 points on
the frozen long-range conditional-predictability metric. Do not spend two more
checkpoint downloads on SFT/DPO stage localization. The next generalization
experiment, if explicitly approved later, should change routing architecture
rather than add another OLMoE post-training stage.

## AMD-F — Compact learned GPT-OSS lookahead predictor

The protocol-ready design is
[GPT_OSS_MILESTONE_F_PROTOCOL.md](GPT_OSS_MILESTONE_F_PROTOCOL.md).

### Claim being tested

A small learned model, using only the current token's weighted top-4 route plus
layer/horizon/phase context, can emit scored demand forecasts for all 32
experts and preserve most of the strong transition-table frontier.

The predictor estimates total target-layer demand for every token. It is not
trained against cold labels and does not observe a cache. At runtime, a
conceptual manager could suppress resident and in-flight experts, aggregate
forecasts across tokens, and choose movements. That manager is outside the
experimental deliverable.

### Minimum implementation

1. Materialize the weighted 32-dimensional source-route vector and top-4
   target labels from existing Milestone E traces.
2. Implement one shared 64-hidden-unit route MLP with source-layer,
   target-layer, and phase embeddings.
3. Freeze the optimizer, seed, training budget, and checkpoint rule in TOML
   before fitting.
4. Preserve the 96/32 request split and compare global/domain popularity,
   route copy, transition, and learned scores at K=4/8/12/16.
5. Apply the decode K=8, delta 1--3 absolute-quality and transition
   noninferiority gate without tuning on the development requests.
6. Report request-bootstrap uncertainty, domain/layer breadth, score
   calibration, parameter bytes, and forecast operations.

The existing 32 test requests are development data after Milestone E review.
If the gate passes, freeze the complete pipeline before a 64-request fresh
confirmation required for the paper's confirmatory learned-predictor claim. If
it fails while the transition table remains strong,
retain the transition predictor and narrow the learned-model claim rather than
escalating network size.

## AMD-G — GPT-OSS analytical workload/memory-system regimes

The dependent plan is
[GPT_OSS_MILESTONE_G_PLAN.md](GPT_OSS_MILESTONE_G_PLAN.md). Freeze it only
after Milestone F fixes the empirical coverage, precision, calibration, and
candidate-amplification frontier.

Adapt the existing H4/H5/AX machinery to GPT-OSS's measured 24-layer,
32-expert, top-4 geometry and 12.640 MiB loaded expert size. Sweep candidate
count K independently from resident capacity R, then vary bandwidth, startup
latency, transfer concurrency, staging capacity, available lookahead slack,
batch/concurrent streams, prefill/decode mix, domain persistence, and demand
union size.

The decisive comparison is predictive versus reactive hierarchy at equal R.
Report useful/false/late/missed bytes, complete cold-demand coverage, required
staging, inverse bandwidth, and modeled mean/P95/P99 service. An all-resident
point is a capacity/performance reference, not an equal-capacity baseline.

No production cache manager, live asynchronous transfer path, or end-to-end
speedup is required. Resident/in-flight suppression and eviction controls are
analytical bookkeeping. Every output must distinguish measured,
trace-derived, assumed, and hypothetical inputs.

## Publication claim ladder

The planned evidence supports three deliberately different claim levels:

1. **Empirical:** existing OLMoE and GPT-OSS checkpoints contain structured
   cross-layer demand, and lightweight predictors achieve a measured held-out
   coverage/amplification frontier.
2. **Analytical:** under explicit workload and memory-system assumptions, the
   measured GPT-OSS points enter or fail to enter defined profitable regions.
3. **Future work:** routing could be co-designed with a multi-horizon
   predictability objective or auxiliary loss to move this frontier. The
   current paper does not test training, language quality, specialization, or
   load-balance tradeoffs.

## AX architecture-exploration sequence

### Assumption boundary

Use measured and trace-derived OLMoE demand as the workload anchor, but sweep
hypothetical future-router quality independently:

- complete cold-set coverage
  \(C\in\{0.50,0.75,0.90,0.95,0.99,0.999\}\);
- predicted/useful byte amplification
  \(A\in\{1,1.25,1.5,2,4,8\}\);
- correlated wave misses rather than independent expert-label corruption.

Every output must label measured inputs, trace-derived inputs, assumed
predictor behavior, and hypothetical hardware parameters separately.

### AX1 — Capacity-first predictive offload

Extend the existing H4/H5 replay over K=8/16/32, selected lookaheads through
Δ=15, and cold-tier bandwidths from 16–512 GB/s. Report HBM bytes retained,
maximum offloaded expert capacity, useful/false/late/missed bytes, and
mean/P95/P99 slowdown relative to the **reactive hierarchy**.

CPU-memory prefetch is a capacity-enabling mechanism. Do not compare it as a
speedup over an otherwise identical all-HBM model.

### AX2 — Reliability and interconnect regimes

Add startup latency, transfer concurrency, and 12/4/1/0.25 MiB transfer
objects. Derive \(\beta_{\min}\), \(C_{\min}\), \(A_{\max}\), and
\(S_{\max}\). Classify bandwidth-, latency-, reliability-, capacity-, and
SLO-limited regions. Use normalized unique demand U=1/2/4/8 only as a clearly
labeled sensitivity for future top-1/top-2 routing.

### AX3 — Predictive three-tier hierarchy

Model pooled or host memory → local HBM → rolling software-managed SRAM.
Long-horizon heads plan HBM placement; short-horizon heads plan SRAM staging;
the ordinary router confirms demand. Sweep 32–512 MiB global SRAM capacity
with double buffering, not a persistent per-layer expert cache.

### Required synthesis

Produce at most:

1. complete coverage versus cold-service-headroom profitability map;
2. fast-tier capacity versus P99 TPOT Pareto frontier;
3. minimum bandwidth versus lookahead inverse-design curve.

The result is quantitative bounds and architecture regimes, not a measured
wall-clock benefit. Live validation is optional and follows only after a
representative point is selected.

### Completed AX result and review gate

The canonical result and three figures are under
`artifacts/runs/h1-standard-small/analysis/architecture/`. The immediate
decision is not another sweep:

1. review the profitability phase map, memory–P99 Pareto, and inverse
   bandwidth/lookahead curve;
2. accept or reject the wave-local model as a useful architecture envelope in
   light of the more pessimistic selected FCFS queue points;
3. select at most one calibration point only if measuring live asynchronous
   behavior would change the architectural conclusion;
4. otherwise preserve C1 top-1/top-2 confirmation and H7 co-designed training
   as future work; neither is on the current Milestone F/G critical path.

## AX4 — Deadline-bounded graceful degradation

The frozen protocol is
[DEADLINE_DEGRADATION_PROTOCOL.md](DEADLINE_DEGRADATION_PROTOCOL.md), with
configuration in
`configs/experiment/ax4-deadline-degradation.toml`.

### Immediate question

At batch-1 decode, can a hard layer commit deadline remove cold-transfer
queueing from the critical path while keeping P99 missing normalized routed
mass small enough to define a credible future training target?

### Completed result

OLMoE uses probabilities from a 64-way softmax after top-8 selection without
renormalizing them. AX4 therefore reports normalized-within-top-8 mass as the
architecture contract and preserves absolute missing router probability as a
secondary result.

The trace-ordered FCFS boundary gives:

- measured 24.14 GB/s PCIe fails at K=8/16/32 with 100%/100%/81% P99 missing
  normalized mass;
- K=32 at 128 GB/s passes with 7.2% P99 missing mass;
- K=8 and K=16 at 256 GB/s pass with effectively zero P99 wave mass;
- K=16 at 128 GB/s is a sharp near miss at 20.4%;
- K=32 at 256 GB/s delivers the mass but fails the 25% benefit threshold
  because reactive offload is already too fast.

The gate requires at least 50% expert offload, ≥25% P99 TPOT improvement over
reactive exact offload, ≤1.5× all-local TPOT including fallback allowance,
P99 missing mass ≤20%, and ≤1% full-fallback waves across multiple domains
and layer bands. It passes at K=8, 256 GB/s, C=99%, A=1.5×, Δ=1, one-layer
slack, and mass-priority ordering: 1.5 GiB is resident, 10.5 GiB is offloaded,
bounded TPOT is 11.25 ms, and the same-hierarchy reactive comparison is
16.41 ms. Only 0.93% of waves are degraded and none takes full fallback.
Passing identifies a robustness target worth training for; it does not
validate quality.

Concrete frozen prediction: the 10.23 ms measured local anchor plus 10%
fallback/commit allowance gives an 11.25 ms deadline cap and 88.9 batch-1
tokens/s, versus the existing 66.83 ms and 15.0 tokens/s K=16 reactive P99
projection. This is a 5.94× same-batch throughput projection. Whether that
large resource gain is useful is decided by the resulting P99 missing routed
mass, not assumed in advance.

The aligned hardware proposal has an always-resident shared/identity/null
plane, optional routed residual experts, fixed commit bitmaps, deadline-aware
mass-per-byte scheduling, bounded speculative credits, and missing-mass
telemetry. Transfers that miss commit cannot delay dispatch.

The immediate next step is human review of the three figures and evidence
boundary. Do not start erasure training, inference collection, or a new model
without explicit permission.

## H5-A — Controlled prediction × hardware sweep

### Question

For the observed OLMoE cold-demand and reuse structure, which combinations of
prediction quality, speculative traffic, fast-tier capacity, lookahead, and
cold-path bandwidth produce useful expected benefit?

### First-order model

Use the dimensionless cold-service pressure:

\[
\rho =
\frac{
\bar N_{\text{candidate}} S_{\text{expert}}
}{
B_{\text{cold}}\Delta T_{\text{layer}}
}.
\]

Reuse existing trace-derived demand, LRU hit/cold rates, exact 12 MiB expert
size, and the effective layer interval. Do not add a more detailed performance
model for this experiment.

Sweep:

- complete-route coverage from 0% to 100%;
- candidate amplification \(A\in\{1,2,4\}\);
- per-layer capacity \(K\in\{8,16,32\}\);
- lookahead \(\Delta=1\ldots15\);
- normalized cold bandwidth \(0.25\times,0.5\times,1\times,2\times,4\times\).

Treat expert size and bandwidth as the same first-order \(S/B\) control. Add a
separate expert-granularity axis only if a result cannot be explained by that
normalized ratio.

### Proposed screening gate

Call a point analytically profitable only if it achieves all three:

- at least 25% modeled reactive-stall reduction;
- at least 50% oracle recovery;
- no more than 2× predicted/useful transfer bytes.

These are prototype screening thresholds, not end-to-end speedup claims.
Freeze them in the H5 protocol before running the sweep.

### Outputs

- `h5_design_points.csv`: every assumption cell and category;
- `h5_windows.csv`: qualifying horizon range/count by capacity, bandwidth, and
  predictor assumption;
- expected stall reduction, oracle recovery, useful/false bytes, and
  amplification;
- one categorical phase diagram over physical headroom and complete coverage,
  with amplification as compact panels.

## H5-B — Inverse predictor requirements

### Question

What minimum complete-route coverage and maximum candidate amplification are
required to cross the H5-A screening gate?

Compute:

\[
C_{\min}(K,\Delta,B,A)
\]

and, where useful:

\[
A_{\max}(K,\Delta,B,C).
\]

This converts the hardware sweep into an ML and training target. The headline
output is one curve of minimum required complete coverage versus lookahead for
K=8/16/32, with bandwidth shown as a small number of line styles or panels.

Report empty windows explicitly. Do not hide a requirement above 100% or a
candidate amplification below the demanded top-k.

## H5-C — Existing-policy placement and replay

### Question

At representative points on the H5 surface, how much oracle benefit do the
existing untuned transition and linear candidate streams recover after false
candidate bytes are charged?

Reconstruct candidates from existing H2/H3 artifacts. Do not retrain, tune, or
add an MLP.

Primary representative cells:

- K=32, Δ=1: prediction-good/physics-limited control;
- K=32, Δ=3: current boundary candidate;
- K=32, Δ=9: long-range linear advantage;
- K=16, Δ=9: oracle-feasible/prediction-limited control.

Report cold-set rather than only total-route coverage:

- useful, false, and late candidate bytes;
- cold-expert and complete-cold-set coverage;
- expected stall reduction;
- oracle recovery;
- candidate amplification and churn.

The visualization should place actual policies on the H5-A phase diagram and
show a small actual-versus-required table. Do not create a separate dashboard.

## H6 — Mechanism competition

H6 is complete. It compared static popularity, domain popularity, reactive
LRU, transition-guided residency, linear-guided residency, and an equal-budget
next-use oracle at K=8/16/32. Prediction could admit only an actually demanded
miss; no broad candidate prefetch was allowed.

At the frozen decode K=16, Δ=3 gate, transition and linear lose 3.9 and 2.5 pp
of expert-stall reduction and 0.7 and 0.6 pp of complete-set hits relative to
the strongest matched simple baseline. The oracle remains strong, but existing
depth-trajectory scores do not predict temporal reuse well enough to select
residency.

Decision: stop this placement mechanism after human figure review. H6 rejects
using same-token depth scores as a temporal-reuse residency controller; it does
not reject learning the original cross-layer demand task on GPT-OSS. AMD-F is
therefore a different prediction experiment, not an attempt to rescue H6.

## H7 — Future co-designed predictability objective

H1–H6 and AMD-C--F observe normally trained models. They can motivate, but
cannot demonstrate, that future routing should include a multi-horizon
predictability objective or auxiliary loss.

The paper may state the hypothesis that jointly optimizing validation quality,
load balance, specialization, and future-route predictability could move the
measured coverage/amplification frontier. Testing that Pareto tuple requires
model training and is explicitly future work. H7 is no longer an active
experiment in the current agenda and must not be inferred from analytical
Milestone G results.

## C1 — Sparse-model transfer check

Before generalizing beyond OLMoE, repeat only the decisive trace and analytical
steps on one checkpoint with top-1/top-2 routing and more sparsity:

- routing integrity and exact expert bytes;
- H1/H2 trajectory structure;
- the H5-A normalized co-design map;
- existing simple transition baseline;
- a linear sidecar only if transitions leave a meaningful information gap.

The purpose is to test whether OLMoE top-8 demand makes complete-set coverage
and cold-service pressure unusually harsh.

## Insight mining after each major experiment

After applying the frozen gate, use cheap post-hoc analysis to extract:

- boundary locations and empty opportunity windows;
- capacity–bandwidth–lookahead substitution rates;
- whether mean headroom disagrees with trace-driven tail behavior;
- the source layers and domains that create or destroy a viable region;
- whether complete coverage, speculative traffic, or physics is limiting;
- dimensionless quantities that transfer across hardware assumptions;
- negative results that redirect prediction toward residency, replication, or
  activation movement.

Label each conclusion as directly supported, analytical inference, or
speculation. Update `docs/FOUNDATIONAL_INSIGHTS.md` only when a result changes
the durable thesis.

## Visualization standard for the next phase

Use at most two primary figures per major experiment:

1. a categorical co-design phase diagram;
2. an inverse-design curve showing the predictor quality required for a chosen
   benefit threshold.

Use a compact actual-versus-required table for H5-C. Every plot must name:

- prediction metric and set semantics;
- physical normalization;
- capacity and candidate-budget semantics;
- profitability assumptions;
- whether the result is measured, trace-driven, or analytical.

Complete the Milestone F development review before collecting fresh
confirmation or freezing Milestone G. H7 remains future work; C1 remains a
separate optional external-validity experiment.
