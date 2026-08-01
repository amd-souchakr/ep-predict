# H6 protocol: prediction-guided residency

**Frozen:** 2026-08-01  
**Evidence grade:** trace-driven pilot  
**New inference or training:** none

## Decisive question

At the same per-layer fast-tier capacity and runtime movement budget, does
prediction-guided admission and eviction materially reduce resident-set misses
relative to static popularity, domain-conditioned popularity, and reactive
LRU across multiple layers and domains?

This tests placement value, not predictor accuracy and not just-in-time
prefetch. A merely predicted expert is never transferred. Prediction can only
decide which **actually demanded miss** is admitted and which resident is
evicted.

## Frozen inputs and split

- H1 routing trace: `artifacts/runs/h1-standard-small`
- H3 aligned projected features:
  `artifacts/runs/h3-standard-small`
- Existing H2/H3 96/32 request split and fixed all-horizon linear heads
- Exact expert size: 12 MiB
- Capacities: 8, 16, and 32 experts per MoE layer
- Lookahead: every valid \(\Delta=1,\ldots,15\)
- Prefill and decode evaluated independently
- Domains evaluated independently on their eight held-out requests

The previously held-out requests are development data after H2–H5 post-hoc
inspection. H6 remains a pilot; a positive gate requires fresh confirmation.

## Replay semantics

Each phase/domain/source-target scope starts from the training-set static
per-layer hot set. The domain baseline instead starts from its
training-derived domain hot set. Initial population is reported separately
from runtime movement.

Dynamic policies may insert at most one expert after each target-layer demand
wave. This is approximately the measured budget of one 12 MiB copy per
effective inter-MoE interval. The budget is a cap, not a requirement to spend
bytes.

Policies:

1. **Static:** training-set per-layer top-\(K\); no runtime insertion.
2. **Domain:** training-set domain/layer top-\(K\); no runtime insertion.
3. **LRU:** admit at most one actually missed expert and evict the least
   recently used resident.
4. **Transition-guided:** exponentially smooth the frozen transition score
   vector; admit the highest-scored actual miss only when it outranks the
   lowest-scored resident.
5. **Linear-guided:** identical policy using the frozen H3 linear logits.
6. **Oracle ceiling:** under the same on-demand-only and one-insertion budget,
   use exact next use to choose admission and Belady eviction.

The fixed smoothing update is

\[
b_t = 0.75 b_{t-1} + 0.25\,z(s_t),
\]

where \(s_t\) is the current transition probability or linear-logit vector.
This converts token-level trajectory evidence into a slowly changing
residency belief. It is not tuned or swept.

An inserted expert is counted as a useful residency movement only if it serves
at least one later resident hit before eviction or the end of the scope.
Otherwise its bytes are counted as wasted for residency, even though the
original cold demand still required service.

## Metrics

Report for every phase, domain, source layer, target layer, lookahead,
capacity, and policy:

- residual cold-expert demand and expert hit fraction;
- complete resident-set hit coverage (fraction of waves with zero cold
  experts);
- useful and wasted runtime movement bytes;
- insertions, evictions, and normalized residency churn;
- first-order expert-work and synchronous-wave stall reduction;
- recovery of the oracle improvement over LRU.

The first-order model assigns one unit of cold service to each missed expert
and one synchronous-wave stall to any wave containing a miss. It is a
trace-driven placement comparison, not an end-to-end latency forecast.

## Frozen gate

Primary scope: held-out decode, \(K=16,\Delta=3\).

A transition- or linear-guided policy passes only if, relative to the strongest
of static, domain, and LRU in every matched layer/domain scope, it achieves:

- at least +2 percentage points mean expert-work stall reduction;
- at least +2 percentage points mean complete-set hit coverage;
- positive gains on both metrics in at least 50% of scopes;
- positive mean gains on both metrics in at least 3 domains and 4 layers.

One candidate policy passing is sufficient for H6 pilot support. The broader
phase/layer/horizon/capacity scan is descriptive and cannot rewrite this gate.

## Decision

- **Pass:** pause for figure review, then request permission for a small fresh
  confirmation workload. Do not download another model yet.
- **Fail:** conclude that routing trajectories are scientifically predictable
  but not useful for this on-demand residency mechanism on the current
  checkpoint. Defer predictor training, router modification, H7, and detailed
  timing work.

## Visualization

Generate one three-panel triangular heatmap (one panel per capacity) showing
the best prediction-guided complete-set hit gain over the strongest simple
baseline by source layer and lookahead. Save PDF, 450-DPI PNG, input hashes,
and a human-review checklist.
