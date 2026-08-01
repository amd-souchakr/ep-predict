# Foundational insights and publication thesis

## Purpose and curation policy

This document preserves the durable ideas, perspective shifts, negative
results, and publication-level interpretations produced by the project. It is
not an experiment log and should not be updated after every run.

Update it only when at least one of the following occurs:

- a major hypothesis gate materially changes the architectural thesis;
- a broad post-hoc analysis reveals a new organizing principle;
- H4–H6 connects prediction to physical timing or policy benefit;
- a second model confirms or contradicts a claimed general principle;
- a result changes the strongest defensible paper claim.

Routine metrics, commands, implementation changes, and transient next steps
belong in `EXPERIMENT_LOG.md`, `STATUS.md`, and the per-hypothesis reports.

**Evidence snapshot:** 2026-08-01, after H1–H6 and the all-layer H2/H3 scan.

---

## Headline thesis

> **MoE routing is a structured trajectory through expert space.**

Global expert popularity is an incomplete description of sparse-model
execution. Even when a workload does not have a sufficiently stable universal
hot set, the token's path through layer-local expert spaces can remain strongly
predictable.

The emerging architectural view is:

> Early hidden state supports long-range resource planning; later routing
> decisions support short-range correction. A hierarchical runtime can
> progressively refine placement, replication, or scheduling as execution
> reveals more information.

This is presently an OLMoE pilot result, not a universal MoE law. H4 found
that physical actionability is conditional: the preregistered compact tier and
short horizon failed, while more residency, lead time, or bandwidth exposed an
oracle-feasible region.

---

## Empirical foundation

The following measurements anchor the interpretation.

### Static hotness is insufficient globally

- OLMoE routes top-8 of 64 experts across 16 MoE layers.
- A mixed-workload static top-8 tier passed the complete H1 gate in only 2/16
  layers.
- Domain-conditioned routing distributions were much more distinct and useful
  than one workload-agnostic placement.

### Current routing predicts nearby future demand

At decode \(K=16\), the H2 transition table achieved:

| Lookahead | Selection coverage | Complete top-8 coverage |
|---:|---:|---:|
| \(n+1\) | 79.0% | 24.1% |
| \(n+2\) | 77.9% | 23.5% |
| \(n+3\) | 76.8% | 22.2% |

All 168 original layer-domain comparisons improved over static popularity.

### A linear sidecar is not a universal replacement

At the formal H3 primary gate—decode \(K=16,\Delta=1\)—the fixed linear
hidden-state sidecar achieved 79.4% selection and 28.7% complete coverage,
versus 79.0% and 24.1% for transitions. It failed the global replacement gate
because selection gain and cross-domain consistency were insufficient, while
candidate churn rose from 42.8% to 52.8%.

### Early hidden states dominate at long range

The all-layer scan changed the interpretation without changing the formal H3
decision:

| Decode, K=16 | Transition | Linear hidden state |
|---|---:|---:|
| Layer 0→15 selection | 53.8% | 69.2% |
| Layer 0→15 complete top-8 | 4.6% | 19.7% |

Across all 120 domain-balanced source-target pairs:

- linear beats transition in 100/120 pairs for selection coverage;
- linear beats transition in 112/120 pairs for complete-token coverage;
- average selection gain is +14.7 points from source layer 0 and +11.0 points
  from source layer 1;
- selection gain becomes negative from approximately source layer 10 onward.

### Distance alone does not explain predictability

Holding source layer 0 fixed, linear selection coverage is:

- 69.9% for target layer 1;
- 80.0% for target layer 9;
- 69.2% for target layer 15.

Prediction does not monotonically degrade with layer distance. Source-target
identity and target-layer routing structure matter at least as much as
\(\Delta\).

### H4 separates information from physical actionability

- Hook-free cached-token forward: 10.229 ms median, or 0.639 ms per effective
  MoE-layer interval.
- Exact 12 MiB pinned-host expert copy: 0.524 ms median at 24.14 GB/s.
- At measured bandwidth, \(K=16,\Delta=3\): 32.8% of cold bytes timely and
  38.9% oracle stall reduction.
- At measured bandwidth, \(K=32,\Delta=3\): 55.5% timely and 61.8% stall
  reduction.

Perfect prediction cannot compensate for an overloaded transfer queue. It
becomes actionable only after residency has reduced the cold set, the issue
point supplies enough lead time, or hardware supplies more bandwidth.

### H5 separates prediction from transfer admission

On held-out decode requests at K=32, the existing policies cover 67–81% of
complete residual cold sets. Yet transferring every nonresident candidate
costs 6.3–6.7× useful cold bytes, so none passes the frozen 2× traffic screen.

This is a major semantic correction: a candidate set is an information
envelope, not an action list. Broad trajectory prediction can be valuable even
when literal candidate prefetch is not, provided a later admission policy
converts uncertainty into selective movement or residency.

The score distributions confirm that this is not simply a signal-absence
failure. For the linear sidecar, useful-versus-unused nonresident scores have
JS divergence of 0.381 bits at \(\Delta=3\) and 0.332 bits at \(\Delta=9\),
with AUROC 0.883 and 0.861. But only 7–8% of scored nonresident IDs are useful.
At 50% complete cold-set coverage, even the linear ranking still needs
3.0–3.3× transferred/useful bytes. The rare-event base rate and set-completion
requirement—not only distribution overlap—control admission profitability.

### H6 separates depth trajectory from temporal reuse

At held-out decode \(K=16,\Delta=3\), reactive LRU leaves 48.1% residual cold
expert demand. Transition- and linear-guided residency leave 50.2% and 48.8%,
while an equal-movement-budget next-use oracle reaches 31.2%.

The oracle establishes a real residency opportunity, but the existing
predictors do not recover it. They predict a target layer for the same token;
residency needs to predict reuse by later tokens. Linear and transition
movements earn later hits only 66.3% and 60.0% of the time, versus 69.7% for
LRU and 94.7% for the oracle.

This negative result changes the architectural interpretation: trajectory
information is not a generic cache-control signal. The conditioning axis must
match the mechanism's reuse axis.

---

## Foundational principles

### 1. Routing is a trajectory, not a histogram

Popularity describes occupancy. Prediction describes motion.

A nearly flat or unstable global histogram does not imply unpredictable
execution. Tokens can follow regular conditional paths through expert space
without producing one globally useful hot set. Hardware that observes only
frequency can miss structure visible in transitions and hidden state.

**Paper-level implication:** characterize MoE workloads using expert-flow
trajectories and source-target predictability, not only expert-popularity
histograms.

### 2. Predictability and cacheability are different properties

The linear sidecar retains substantial long-range coverage while changing more
than half of its candidate set per token. Demand can therefore be predictable
but mobile.

- Cacheability requires stability and reuse.
- Schedulability requires advance information.

Prediction may be more valuable for prioritizing transfers, reserving
bandwidth, planning replication, or ordering work than for blindly loading
every candidate.

### 3. Router decisions are useful but lossy telemetry

An early top-8 route is a severe quantization of a 2,048-dimensional hidden
state. It reports the local execution choice but discards semantic information
that remains relevant to later routing.

The evidence is consistent with an information-bottleneck interpretation:

- hidden state is a richer long-range planning signal;
- the current route becomes a better locally sufficient signal near the
  target;
- neither signal is globally dominant.

This is an interpretation, not yet a causal or information-theoretic proof.

### 4. Network depth is a control horizon

The early-linear/late-transition regime resembles a two-stage controller:

1. **Feed-forward planning:** an early hidden state forms a coarse long-range
   resource plan.
2. **Feedback correction:** later routing observations refine or correct that
   plan as the deadline approaches.

The natural policy is hybrid rather than winner-take-all. Predictor choice
should depend on source layer, target layer, lead time, confidence, and
resource cost.

### 5. Layer-pair identity is more fundamental than lookahead distance

A scalar \(\Delta\) hides the model's internal regimes. The correct object is a
directed source-target predictability graph whose nodes are MoE layers and
whose edges carry coverage, confidence, churn, and available lead time.

Some distant targets are easier to predict than nearby targets. This may
reflect differences in routing entropy, specialization, or the extent to which
a target layer expresses semantics already present in the early residual
stream.

### 6. Complete-set coverage is a distinct structured objective

Nearly identical marginal selection coverage can produce meaningfully
different complete top-8 coverage. The linear sidecar's primary H3 result
demonstrated this directly.

Hardware waits for demanded sets, not independent labels. Predictor evaluation
must therefore preserve correlated misses and report token-, wave-, and
eventually decode-step-complete coverage.

Ordinary recall is insufficient for architectural conclusions.

### 7. Candidate capacity can create a reliability threshold

For H2 transition prediction at \(n+1\):

- \(K=8\): 58.0% selection, 1.2% complete coverage;
- \(K=16\): 79.0% selection, 24.1% complete coverage;
- \(K=32\): 93.2% selection, 64.1% complete coverage.

Capacity does not translate smoothly into safe execution. A fast tier can be
large enough for respectable average recall yet too small to prevent nearly
universal partial misses. Hardware capacity should be studied as a reliability
regime transition, not only a hit-rate curve.

### 8. Routing behavior is conditional, not a scalar model property

The evidence depends jointly on:

\[
\text{checkpoint}
\times \text{source-target layer pair}
\times \text{domain}
\times \text{phase}
\times \text{candidate budget}.
\]

Statements such as “this model has skew” or “MoE routing is predictable” are
usually too coarse. Conditional structure weakens universal static policies
but strengthens the case for adaptive resource control.

### 9. The memory hierarchy may encode confidence as well as latency

A predictive hierarchy can be organized by how knowledge evolves:

- persistent tier for stable/domain-hot experts;
- planned tier for early, long-lead predictions;
- immediate tier for later, higher-confidence corrections;
- reactive path for residual misses.

This reframes fast, medium, and slow storage as physical representations of
different certainty and deadline regimes—not merely different access times.

### 10. The model may expose an accidental resource-control surface

A fixed 128-dimensional random projection of a 2,048-dimensional router input
retains strong long-range routing information. One possible explanation is
that future-routing intent occupies a low-dimensional or redundant subspace.

If confirmed, a compact external control plane may be extractable without
modifying or retraining the base model. The sidecar would interpret latent
computational intent for resource planning.

This is provocative but preliminary: only one projection size, seed, model,
and workload have been tested.

### 11. Prediction and residency are complements, not substitutes

At \(K=16,\Delta=3\), the average eligible wave still needs 3.46 cold expert
copies. Three layer intervals provide 1.92 ms of nominal lead time, while those
copies require about 1.81 ms only when the queue is empty. Sustained arrivals
create backlog and make 67.2% of cold bytes late.

At \(K=32\), residency removes about four-fifths of demand before prediction
acts. The oracle then crosses the physical gate at the same lookahead and
bandwidth. A hierarchy is therefore not justified by prediction replacing
capacity; prediction makes finite capacity more effective after capacity has
already reduced transfer pressure.

### 12. A useful hardware unit is experts transferable per layer interval

For the measured system:

\[
\frac{T_{\text{layer interval}}}{T_{\text{12 MiB copy}}}
=
\frac{0.639}{0.524}
\approx 1.22
\]

This dimensionless exchange rate connects model execution, expert granularity,
and interconnect bandwidth directly. Compare it with cold experts per wave,
not raw GB/s alone. It is a compact first-order screen for whether a proposed
issue point can drain demand faster than demand is created.

### 13. Prediction should target the residual cold set, then separate belief from action

Complete route coverage is the wrong final metric once a fast tier already
holds part of the route. The operational target is:

\[
D_{\mathrm{cold}} = D_{\mathrm{route}} - D_{\mathrm{resident}}.
\]

At K=32, this conditioning raises complete-set coverage into the 67–81% range.
But the same experiment shows why prediction and action must remain separate:
false nonresident candidates dominate bytes unless an admission mechanism
filters them.

The resulting control stack has three distinct objects:

1. a broad belief over future experts;
2. a residency state that removes already-satisfied demand;
3. a resource-aware admission decision that commits bytes.

Conflating these objects makes a good predictor look like a bad architecture,
or a bad transfer policy look like a prediction failure.

### 14. Useful ranking is not sufficient admission

The H5 score analysis finds useful-versus-useless AUROC of 0.883 at Δ=3 and
0.861 at Δ=9 for the linear sidecar. The latent signal is therefore not weak.
Yet a shared threshold needs about 3.0–3.3× transfer amplification to preserve
50% complete cold-set coverage.

Set completion amplifies modest score overlap: several useful experts must all
survive admission, while many more useless expert IDs each have a chance to
cross the threshold. Admission must therefore be trained and evaluated as a
resource-constrained set decision, not inferred from pairwise ranking quality
alone.

### 15. Architectural value is a three-gate intersection

The experiments separate three necessary properties:

1. **Information:** future demand is predictable.
2. **Service:** the hierarchy can act before the deadline.
3. **Selectivity:** acting on predictions does not waste excessive bytes or
   capacity.

H2/H3 support the information gate. H4 exposes conditional service regions.
H5 finds an analytical intersection but shows that the unchanged candidate
streams miss its selectivity boundary.

No single accuracy, bandwidth, or cache-hit metric establishes architectural
value. A system is profitable only in the intersection of all three regions.
This prevents two symmetric errors: dismissing useful trajectory information
because one transfer policy fails, and claiming a viable hierarchy because a
predictor or oracle looks strong in isolation.

### 16. Prediction must match the mechanism's time axis

The project now exposes two different prediction problems:

\[
\text{depth: }P(E_{\ell+\Delta,t}\mid \text{state}_{\ell,t}),
\qquad
\text{time: }P(E_{\ell,t+\tau}\mid \text{history}).
\]

H2/H3 establish strong depth prediction. H6 shows that feeding those scores
into a temporal residency policy does not beat simple caching, despite a
substantial oracle gap.

This is more than a failed heuristic. It is a workload–mechanism alignment
principle: advance information has architectural value only when it predicts
the future event that consumes the managed resource. A depth trajectory can
schedule within-token transfers; cache retention needs cross-token reuse;
replication needs cross-request or cross-device demand. These tasks may share
features, but one cannot be substituted for another without evidence.

---

## Perspective shifts produced by the experiments

| Initial framing | Evidence-driven framing |
|---|---|
| Find globally hot experts | Model conditional expert-flow trajectories |
| Predictability implies caching value | Predictability may enable scheduling even when caching is weak |
| One predictor should beat another globally | Use different signals at different control horizons |
| Accuracy should decay with layer distance | Source-target identity dominates a simple distance law |
| Marginal expert recall measures success | Complete demanded-set coverage controls stalls |
| More capacity gives proportionally better behavior | Complete coverage can exhibit threshold-like regimes |
| Fast memory tiers differ only by speed | Tiers can represent confidence, lead time, and commitment |
| Prediction can compensate for a small cache | Residency must first reduce cold demand below transfer service rate |
| A predictor candidate list is a prefetch list | Prediction is a belief envelope; admission commits scarce bytes |
| High AUROC or visible score divergence implies efficient admission | Rare-event base rate and complete-set survival determine byte efficiency |
| Complete route coverage is the placement target | Complete residual-cold-set coverage is the operational target |
| Quote bandwidth in GB/s | Compare experts transferable per layer interval with cold experts per wave |
| One good metric establishes architectural value | Information, physical service, and selectivity must pass together |
| Future-expert prediction is a generic cache signal | The predictor's axis must match the mechanism: depth for within-token action, time for reuse |
| A failed primary gate ends the idea | A failed global policy can expose a valuable conditional regime |
| Preregister every important interaction | Freeze one decision, then use cheap post-hoc scans for discovery |

---

## Hard-earned research lessons

### Negative gates should narrow claims, not trigger complexity

H1 rejected universal static placement but revealed domain structure. H3
rejected universal linear replacement but revealed early-layer long-range
value. Neither failure justified an MLP or a tuning sweep.

### Preregistration protects decisions, not discovery

The formal H3 decision remains valid. The post-hoc source-target scan did not
rewrite it; it found a different, narrower architectural claim.

Simple gates plus broad inexpensive analysis were more productive than a large
early hypothesis tree would have been.

### Aggregation can hide the architectural regime

The \(\Delta=1\) global mean averaged fifteen source layers and obscured the
strong layer-0/layer-1 behavior. Any aggregate should be paired with a compact
heterogeneity view when layer or domain interactions are plausible.

### Accuracy before physics is only workload evidence

H4 confirms why predictor accuracy alone was insufficient. One exact expert
can arrive within one effective layer interval, but several continuously
arriving cold experts overload the serialized copy path. Deadline-feasible
bytes and residual wave stall—not isolated transfer latency—decide
actionability.

### High churn changes the likely mechanism

The observed candidate replacement rates argue against literal per-token
loading. Candidate unions, reuse, wave aggregation, priorities, and residency
must be modeled before interpreting prediction as data movement.

### A repeatedly inspected test split becomes discovery data

The current 32 held-out requests supported valid frozen gates, but subsequent
post-hoc policy discovery means they cannot independently confirm the new
hybrid policy. Fresh requests are required only if H4 makes confirmation worth
the cost.

### A policy bottleneck does not automatically justify model training

H5 identified excessive speculative movement, not a failure of the language
model or proof that routing should be reshaped. Turning score separation into
a base-model training objective would introduce a different hypothesis about
loss, load balance, and routing controllability.

The next aligned question is whether existing trajectory information improves
residency or replication over simple policies. Co-training predictable routing
remains later work and should not be used to rescue an unproven placement
mechanism.

### A strong oracle gap can coexist with the wrong predictor

H6's next-use oracle materially beats LRU, so residency is not intrinsically
empty. Yet the depth predictors fail. Oracle headroom identifies an opportunity
for information; it does not prove that the information already collected is
the right information.

Before tuning a model, write the exact conditional event the controller needs
to predict. This check would distinguish same-token layer lookahead from
cross-token reuse and prevents optimizing an impressive but causally misaligned
metric.

---

## Provocative hypotheses for later work

These are not established results.

### Sparse top-1/top-2 routers may strengthen the sidecar case

OLMoE's top-8 route gives transition tables eight source observations while
making complete-set coverage unusually demanding. A top-1/top-2 model may
provide weaker route telemetry but require much smaller complete sets. Hidden
state could become more valuable relative to transitions.

### Routing may reveal a latent computational program

Predictable expert trajectories may be an observable projection of a token's
internal computational plan. Early residual state may encode not only meaning,
but an approximate future sequence of specialized transformations.

### The best output may be a resource action, not an expert label

A future controller might predict:

- reserve bandwidth;
- retain or evict an expert;
- replicate to another tier;
- move an activation instead of weights;
- prefetch a tile or quantized fragment;
- defer commitment until later feedback.

Direct resource-action prediction could eventually dominate independent
expert-label prediction, but only after H4/H5 define the relevant cost.

### Whole-expert movement may fail while the principle survives

Even if 12 MiB experts are physically too large for just-in-time movement,
trajectory prediction may still benefit:

- slow-timescale replication;
- fast-tier residency planning;
- expert-tile or matrix-fragment movement;
- token dispatch to resident experts;
- bandwidth and queue reservation;
- workload admission and batching.

The predictive-control thesis is broader than whole-expert PCIe prefetch.

---

## Unanswered foundational questions

1. Does the descriptive \(K=32,\Delta=3\) oracle region survive measured
   concurrent copy/compute contention?
2. Can prediction be aggregated across requests or token waves so that reuse
   offsets candidate churn?
3. Can a direct temporal-reuse predictor, request-level aggregation, or
   replication objective exploit the H6 oracle gap without becoming a new
   high-complexity project?
4. Is future-routing information genuinely low-dimensional, or did one random
   projection happen to work well?
5. Does the early-linear/late-transition regime reproduce on fresh requests?
6. Does it generalize to a newer top-1/top-2 MoE checkpoint?
7. Are middle layers easier to predict because of lower router entropy,
   semantic specialization, residual-stream geometry, or another mechanism?
8. Is whole-expert movement ever preferable to moving activations toward
   resident experts?
9. Should the hierarchy optimize strict latency, throughput, bandwidth,
   replication quality, or different objectives in different regimes?
10. Can routing or control behavior eventually be co-trained without harming
    model quality?

---

## Strongest defensible claims today

### Directly supported for the pinned OLMoE workload

- Global static hotness is insufficient, while conditional routing structure
  is strong.
- Current routes predict nearby future expert demand far better than marginal
  popularity.
- A fixed projected-hidden-state linear readout retains substantial
  long-range routing information.
- Linear prediction is strongest from early source layers; transitions are
  preferable in several late-layer regimes.
- Complete-set coverage and candidate churn materially change the
  interpretation of ordinary selection coverage.
- On the measured platform, the compact \(K=16,\Delta=1\)–3 whole-expert
  oracle target is physically insufficient.
- A larger \(K=32,\Delta=3\) oracle region exists in the analytical replay.
- A controlled first-order profitability region exists, but the unchanged
  transition and linear candidate streams do not enter it.
- The current linear ranking contains real useful-versus-unused separation,
  but needs roughly 3.0–3.3× transferred/useful bytes to preserve 50%
  complete cold-set coverage.
- Existing transition/linear depth scores do not beat static/domain/LRU
  on-demand residency at equal capacity and movement budget.
- A strong equal-budget next-use oracle gap remains: at decode K=16, Δ=3,
  residual cold demand is 31.2% for oracle versus 48.1% for LRU.

### Plausible architectural inference

- A hybrid early-planning/late-correction controller is more appropriate than
  one universal prediction policy.
- Predictive information may help within-token scheduling even when literal
  prefetch is too expensive; residency requires a separate temporal-reuse
  signal.
- A memory hierarchy could be organized around certainty and deadline as well
  as speed and capacity.
- Experts transferable per layer interval is a useful co-design quantity, and
  prediction becomes actionable only after residency reduces cold demand.

### Not yet supported

- The analytical \(K=32,\Delta=3\) region survives real concurrent
  copy/compute contention.
- Prediction reduces end-to-end latency or TPOT.
- The current prediction-guided transfer policy is profitable.
- The current depth predictors improve on-demand expert residency.
- The result generalizes beyond one top-8 OLMoE checkpoint.
- The base model learned to manage hardware resources.
- Making routing more predictable would preserve model loss and load balance.
- Whole-expert movement beats activation movement or additional local memory.

---

## Candidate publication framing

### Possible title

**MoE Routing Is a Structured Trajectory Through Expert Space**

### Possible subtitle

**Early Hidden States Enable Long-Range Expert-Demand Planning While Later
Routes Provide Short-Range Correction**

### Candidate contributions under the current evidence

1. Show that static expert popularity misses predictable conditional
   trajectories.
2. Identify a source-layer-dependent crossover between hidden-state and
   transition prediction.
3. Demonstrate why complete-set coverage and candidate churn—not ordinary
   recall—govern architectural usefulness.
4. Map the capacity–lead-time–bandwidth region in which predictive information
   is physically actionable.
5. Separate predictive belief, residency state, and byte-committing admission,
   showing why information, service, and selectivity are independent gates.
6. Show that depth-trajectory predictability and temporal cache reuse are
   distinct, and that policy value requires matching the prediction axis to
   the resource-consumption axis.

The defensible contribution is a workload and co-design boundary, not a
profitable end-to-end prefetch implementation. Strong long-range routing
information can coexist with insufficient physical time or excessive
speculative traffic; this redirects the mechanism toward residency,
replication, scheduling, or finer-grained movement without weakening the
central trajectory result.

---

## Revision history

- **2026-08-01:** Initial synthesis after H1, H2, H3, and the complete
  source-target horizon analysis. H4 physical feasibility remains open.
- **2026-08-01:** H4 added the capacity–lead-time–bandwidth boundary,
  established the experts-per-layer-interval screening quantity, and showed
  that prediction and residency are complements.
- **2026-08-01:** H5 separated predictive belief from transfer admission,
  established information–service–selectivity as three independent gates, and
  showed how rare-event base rates make strong ranking insufficient for
  profitable movement.
- **2026-08-01:** H6 separated within-token depth prediction from cross-token
  reuse prediction. Existing depth scores failed equal-budget residency despite
  a strong next-use oracle ceiling, establishing the prediction-axis and
  mechanism-axis alignment principle.
