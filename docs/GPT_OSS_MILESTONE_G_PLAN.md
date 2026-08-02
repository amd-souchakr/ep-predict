# GPT-OSS 20B Milestone G action-value plan

**State:** protocol-ready; exact TOML and source hashes are the next freeze

**New model inference or cache implementation:** none

**Evidence grade:** trace-driven analytical study

**Primary decision:** whether the confirmed forecast contains enough timely,
selective information to improve expert service over reactive movement at the
same resident capacity and transfer budget

## Decisive question

Given the frozen GPT-OSS 20B lookahead scores and realized routes, where does
early expert-demand information have positive action value after residency,
false candidates, finite bandwidth, startup latency, queueing, deadlines, and
staging are charged?

This is deliberately stricter than another predictor evaluation. Milestone F
closed the information gate for one checkpoint and four domains. Milestone G
tests the intersection of information, service capacity, and transfer
selectivity. It may find that a good forecast is not actionable under
whole-expert movement; that would be a substantive architecture result, not a
predictor failure.

## Frozen empirical inputs

- exact 64-request confirmation order, phases, route IDs, and route weights;
- exact per-expert scores from the frozen 574,080-parameter weighted+binary
  layer-pair checkpoint, with no refit;
- 24 routed layers, 32 experts/layer, top-4 routing, no shared expert;
- 12.640 MiB loaded bytes per whole expert and approximately 404 MiB/layer;
- candidate counts `K = 4, 8, 12, 16` plus score-threshold operating points;
- lookaheads through delta 23, with delta 1--3 labeled confirmatory and longer
  horizons labeled exploratory.

At K=8, selection/complete-route coverage is 91.7%/74.1% at delta 1,
87.2%/65.5% at delta 12, and 84.7%/62.8% at delta 23. At delta 23, K=12
recovers 91.5% selection and 77.6% complete-route coverage at 50% more
candidates. Use the exact scores behind these summaries; aggregate coverage
cannot represent which candidates are resident, useful, late, or correlated.

Keep four quantities distinct:

- `K`: experts nominated by the predictor;
- `R`: experts resident at the target layer;
- `M`: transfers admitted from the nonresident prediction set;
- `U`: realized unique residual cold experts.

`K/32` is a belief-envelope size. It is neither resident fraction nor a
command to transfer K experts.

## Decision contract

Compare predictive service with reactive service at equal `R`, expert
granularity, link parameters, and transfer-admission budget. Retain an oracle
with the same resource contract to distinguish insufficient information from
insufficient service capacity. The all-resident case is a reference, not the
equal-capacity baseline.

Use the existing H5 screen as the default preregistered success contract:

- at least 25% reduction in modeled residual expert-service stall versus
  reactive service;
- at least 50% recovery of the equal-resource oracle stall opportunity;
- no more than 2.0x transferred/useful bytes;
- no increase in resident capacity and no hidden priority or bandwidth
  advantage.

Freeze these thresholds, the exact source artifacts, and the scenario axes in
TOML before replay. Report the full Pareto even if no point passes.

## Stage G0 — exact action accounting

For every token, source-target pair, K, R, and admission rule:

1. form realized target demand;
2. subtract resident and already-in-flight experts;
3. deduplicate predictions against the same state;
4. rank or threshold the remaining scored candidates;
5. label admitted bytes as useful, false, late, or redundant;
6. label uncovered residual demand as missed/reactive;
7. preserve complete residual-set and wave-level outcomes.

This stage answers whether candidate amplification survives operational
suppression. Raw K/top-4 amplification is only a warning signal: at K=8 and
91.7% selection it is already about 2.18 nominated candidates per useful
selection before resident suppression. Exact replay, not this ratio, decides
the traffic gate.

## Stage G1 — first-order regime pruning

Screen the design space using dimensionless quantities before queue replay:

\[
q = \frac{B T_{layer}}{S_e}, \qquad
\rho = \frac{\lambda_{expert} S_e}{B}, \qquad
\chi = \frac{A}{\Delta}.
\]

Here `q` is whole experts transferable per layer interval, `rho` is offered
transfer load, and `chi` is the amplification paid per unit lookahead. Include
startup latency, finite transfer concurrency, and staging slots in the
feasibility test. Prune cells that are unambiguously overloaded, miss their
deadline even with an empty queue, exceed staging capacity, or are trivially
dominated.

Start with normalized service rates so the result is portable. Factorize the
sweep rather than constructing a blind Cartesian product:

- `R = 0, 4, 8, 16, 24, 32`;
- detailed horizons `delta = 1, 2, 3, 6, 12, 18, 23`;
- `K = 4, 8, 12, 16` and selected score thresholds;
- service rates spanning 0.25, 0.5, 1, 2, and 4 whole experts per layer
  interval;
- startup ratios and transfer concurrency sufficient to expose serialization;
- staging capacities of 4, 8, 16, and 32 whole-expert slots.

Prefill, concurrent streams, and request-arrival sweeps are secondary axes.
Add them only after the batch-one decode anchor identifies a boundary worth
stress-testing.

## Stage G2 — trace-ordered deadline replay

Replay all feasible cells plus a narrow band on each side of the first-order
boundary. Model transfer issue, startup, service, in-flight deduplication,
staging occupancy, completion, and target-layer deadlines in request order.
At minimum compare:

- reactive on-demand service;
- learned score-priority admission;
- equal-resource demand oracle;
- one cheap transition comparator where it clarifies the value of learning.

FCFS is the conservative default. Add earliest-deadline-first or
probability-mass-per-byte only if it answers whether a failed cell is a queue
discipline failure rather than a forecast failure. Do not tune a large policy
family on the confirmation trace.

Report complete residual-cold-set coverage; useful, false, redundant, late,
and missed bytes; staging occupancy; queue utilization; mean/P95/P99 modeled
stall; deadline-miss incidence; and oracle opportunity recovery. Preserve
correlated misses—independent Bernoulli reconstruction is invalid for a
top-4 demand set whose slowest missing expert gates execution.

## Stage G3 — inverse hardware requirements

For every robustly useful predictor point, solve for:

- minimum effective bandwidth or experts per layer interval;
- maximum whole-expert transfer size at fixed bandwidth;
- minimum usable lookahead after issue and compute overhead;
- required transfer concurrency and staging capacity;
- sensitivity to workload concurrency and demand union.

The point is not to announce one preferred interconnect. It is to reveal
whether the measured information favors longer-horizon whole-expert movement,
short-horizon tiles, bandwidth reservation, or no movement at all.

## Primary outputs

1. **Action-value phase map:** learned pass/fail and oracle headroom over
   resident fraction and normalized service rate, with the best K/lookahead
   operating point in each cell.
2. **Capacity/traffic/stall Pareto:** HBM retained versus transferred/useful
   bytes and modeled P99 stall, comparing learned, reactive, and oracle.
3. **Inverse lookahead curve:** minimum bandwidth or maximum transferable
   expert bytes versus lookahead for selected admission budgets.

Produce a fourth workload-sensitivity view only if concurrency or prefill
changes the qualitative conclusion.

## Evidence and stopping rules

Every reported input is labeled measured, trace-derived, assumed, or
hypothetical. Without GPT-specific unhooked layer timing, report normalized
slack and inverse requirements rather than a measured MI355X speedup.

Stop after the three primary outputs and review. Measure GPT-specific timing
or concurrent copy/compute only if a conclusion changes within the plausible
timing interval. Build a live movement prototype only if a coherent region
passes the frozen action-value contract. Do not respond to an analytical miss
by collecting more training tokens: forecast accuracy, service capacity, and
admission efficiency are different failure modes.

## Claim boundary

A positive region shows where this frozen predictor would satisfy the stated
analytical contract. It does not show that current GPT-OSS runs faster, that a
particular cache policy is optimal, or that jointly training future route
heads preserves language quality and load balance. A negative whole-expert
result does not refute routing predictability; it bounds what this information
can buy under the swept mechanism.
