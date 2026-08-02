# GPT-OSS 20B Milestone G analytical regime plan

**State:** planned; freeze only after Milestone F fixes the empirical predictor
frontier
**New model inference or cache implementation:** none
**Evidence grade:** trace-driven analytical study

## Decisive question

Given the measured GPT-OSS 20B lookahead prediction frontier, under what
workload and memory-system conditions can predictive expert movement improve a
hierarchical memory system relative to reactive service at the same resident
capacity?

Milestone G is not a software cache-manager experiment. A cache manager is a
conceptual consumer of scored demand forecasts. The analysis charges every
useful, false, late, and missed movement under explicit capacity, bandwidth,
latency, and lead-time assumptions.

## Empirical inputs

- 24 routed layers, 32 experts/layer, top-4 routing, no shared expert;
- 12.640 MiB loaded bytes per expert and approximately 404 MiB/layer;
- request-ordered prefill/decode demand and expert weights from Milestone E;
- the confirmed Milestone F coverage/precision/calibration frontier over
  candidate count `K` and lookahead `delta`;
- measured or trace-derived skew, overlap, demand-union size, and burstiness.

Keep prediction candidate count `K`, resident capacity `R`, and actual transfer
budget distinct. `K/32` is candidate-set fraction; `R/32` is resident fraction.

## Analytical sweep

Sweep or factorize:

- `K = 4, 8, 12, 16` and score-threshold operating points;
- independent resident capacities `R = 0, 4, 8, 16, 24, 32`;
- lookahead and available compute slack;
- cold-tier bandwidth, startup latency, transfer concurrency, and staging
  capacity;
- whole-expert and factorized transfer sizes;
- batch-one decode, concurrent decode streams, prefill/decode mix, request
  arrival rate, domain persistence, and demand-union size.

For a predicted nonresident set `P`, charge transfer time as

\[
T_{move}=N(P)L + \frac{|P|S_e}{B},
\]

where `L` is per-transfer startup latency, `S_e` is expert bytes, and `B` is
effective bandwidth. A prediction is timely only if this service completes
within the swept target-layer slack. Deduplicate resident and already-in-flight
experts before charging traffic.

The resident-set abstraction may use static, domain, LRU, and oracle controls
to measure sensitivity. It is bookkeeping for the analytical model, not a
claim that a production eviction algorithm has been built.

## Comparisons and metrics

Compare predictive hierarchy with reactive hierarchy at the same `R`. Keep an
all-resident point as a capacity/performance reference, not as an attainable
equal-capacity baseline.

Report:

- HBM bytes retained and model capacity displaced;
- complete cold-demand coverage and residual synchronous miss waves;
- useful, false, late, and missed bytes;
- predicted/useful byte amplification and total traffic overhead;
- required staging bytes and minimum interconnect bandwidth;
- mean/P95/P99 modeled service penalty and stall reduction;
- recovery of the physical oracle opportunity;
- sensitivity to concurrency, domain locality, and prefill demand unions.

Every result must label measured, trace-derived, assumed, and hypothetical
inputs. Without GPT-specific unhooked layer timing, report normalized slack and
inverse bandwidth requirements rather than a measured MI355X speedup.

## Primary outputs

1. empirical complete-route coverage versus candidate-amplification frontier;
2. profitability phase map over predictor point, resident fraction, and
   bandwidth/service headroom;
3. minimum bandwidth versus lookahead inverse-design curve;
4. one workload-sensitivity view only if concurrency or prefill changes the
   qualitative regime.

## Claim boundary

A positive region shows where the measured predictor would satisfy the stated
analytical contract. It does not show that current GPT-OSS runs faster, that a
particular cache policy is optimal, or that training for predictability
preserves language quality and load balance.

The paper may motivate multi-horizon predictive-routing objectives or
auxiliary losses as future co-design work. Training such a model and measuring
its quality/load-balance Pareto are explicitly outside Milestone G.
