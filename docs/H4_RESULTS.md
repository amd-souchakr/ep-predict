# H4 oracle hardware-feasibility results

**Run:** `artifacts/runs/h1-standard-small/analysis/h4`  
**Evidence grade:** calibrated single-GPU pilot  
**Formal decision:** compact-tier short-horizon gate not supported  
**Broader interpretation:** mixed; a feasible oracle region exists with more
capacity, lead time, or bandwidth  
**Human figure review:** completed 2026-08-01

## Plain-language conclusion

Perfect prediction is not enough for the preregistered \(K=16\), short-horizon
design. At the best primary point, \(\Delta=3\), only 32.8% of cold expert bytes
arrive on time and oracle prefetch removes 38.9% of reactive stall. Both miss
the frozen 50% thresholds.

The limiting fact is simple: this GPU advances through one effective MoE-layer
interval in 0.639 ms, while one exact 12 MiB expert takes 0.524 ms to copy.
There is time for only about 1.22 serialized expert copies per layer of
lookahead. At \(K=16,\Delta=3\), the trace still needs an average 3.46 cold
experts per eligible wave, and sustained demand creates a transfer queue even
when an isolated wave appears nearly hideable.

This is not evidence that whole-expert prefetch is impossible everywhere.
\(K=32,\Delta=3\) reaches 55.5% on-time cold bytes and 61.8% stall reduction at
the measured rate. \(K=16\) reaches a similar 61.8% reduction at \(\Delta=9\),
and doubling bandwidth lets \(K=16,\Delta=1\) cross the stall gate. The
architectural result is therefore a capacity–lead-time–bandwidth boundary, not
a binary statement about prediction.

## Calibration

| Quantity | Result |
|---|---:|
| Hook-free cached-token forward, median | 10.229 ms |
| Effective interval per 16 MoE layers | 0.639 ms |
| Exact 12 MiB H2D copy, median | 0.524 ms |
| Exact-copy wall completion, median | 0.531 ms |
| Fitted pinned H2D bandwidth | 24.14 GB/s |
| Transfer fit startup | 0.0028 ms |

The decode timing contains 80 measured steps: 20 for each of code,
conversation, general text, and math. Domain medians span only 10.178–10.290
ms. The 12 MiB copy has a 0.524–0.526 ms P10–P90 interval. Calibration noise is
small relative to the failed-gate margin.

The inter-layer value is an effective average derived from a completely
unhooked forward, not a per-layer CUDA profile. It avoids contaminated hook
timing but does not capture layer-to-layer heterogeneity.

## Frozen primary gate

Decode, measured bandwidth, \(K=16\):

| Lookahead | Resident-hit bytes | On-time cold bytes | Late cold bytes | Oracle stall reduction | Stalled waves |
|---:|---:|---:|---:|---:|---:|
| \(\Delta=1\) | 55.5% | 26.6% | 73.4% | 31.8% | 83.4% |
| \(\Delta=2\) | 56.4% | 29.8% | 70.2% | 35.8% | 79.4% |
| \(\Delta=3\) | 56.8% | 32.8% | 67.2% | 38.9% | 74.5% |

The resident percentage changes slightly with lookahead because later
lookaheads have fewer eligible target layers. At \(\Delta=3\), 360,247 of
833,144 demanded expert occurrences are cold. Only 832 are compulsory first
uses; 359,415 are capacity-eviction misses. Thus the dominant problem is
limited residency and repeated movement, not model cold start.

## Feasibility boundary

| K | Lookahead | Bandwidth | Resident hits | On-time cold bytes | Stall reduction |
|---:|---:|---:|---:|---:|---:|
| 8 | 3 | measured | 40.0% | 24.1% | 29.3% |
| 16 | 3 | measured | 56.8% | 32.8% | 38.9% |
| 32 | 3 | measured | 79.6% | 55.5% | 61.8% |
| 16 | 9 | measured | 53.4% | 58.7% | 61.8% |
| 16 | 1 | 2× measured | 55.5% | 50.0% | 58.1% |

Three resources substitute for one another:

1. **Capacity** converts future transfers into resident hits.
2. **Lead time** drains more of the serialized transfer queue before demand.
3. **Bandwidth** increases the number of whole experts that fit inside each
   layer interval.

This makes prediction and caching complements. Prediction is most physically
useful after enough residency has reduced the cold set; it is not a substitute
for capacity.

The \(\Delta=15\) points reach 100% in the plotted model, but cover only the
single layer-0→layer-15 pair. They must not be read as whole-model performance.
The curve deliberately labels this changing eligible-layer composition.

## Decision and recommendations

The formal gate remains `PILOT_DOES_NOT_SUPPORT`: \(K=16\) and the established
\(\Delta=1\)–3 short horizons do not hide a meaningful majority of cold-expert
cost. Per the preregistration, transition and linear policies were not
overlaid, retrained, or tuned.

The broader H4 hypothesis is mixed rather than universally rejected because a
nontrivial oracle region exists at \(K=32,\Delta=3\). The next prototype step
does not require higher timing fidelity. Use the existing trace and first-order
model to:

1. sweep assumed predictor quality, amplification, capacity, lookahead, and
   normalized bandwidth;
2. identify analytical viability/profitability windows;
3. solve for minimum predictor requirements;
4. place the existing transition and linear streams on that surface without
   retraining.

Concurrent copy/compute validation remains necessary before an end-to-end
latency claim, but it is deferred until the analytical study identifies a
policy region worth implementing. Do not optimize the predictor, add an MLP,
or collect a new routing workload first.

## Post-hoc co-design regime map

The co-design map combines physical and informational conditions without
pretending they are already an end-to-end policy result.

Its horizontal axis is cold-service headroom:

\[
H =
\frac{\Delta T_{\text{layer}}}
{\bar N_{\text{cold}} T_{\text{copy}}}.
\]

- \(H<1\): average serialized cold-transfer work exceeds nominal lead time.
- \(H>1\): the average wave has nominal service headroom.

The vertical axis is complete top-8 route coverage from the existing
transition or linear candidate stream. A 50% descriptive boundary separates
obviously incomplete prediction from a candidate region. Filled markers
independently pass the stricter trace-driven H4 oracle thresholds, so an open
marker above \(H=1\) exposes tail or queue failure hidden by the mean ratio.

For this screening slice, the same numeric \(K\) is used for per-layer fast-tier
capacity and predictor candidate budget. This coupling is not a cache/prefetch
policy simulation.

The map exposes four useful regimes:

1. **Both limited:** \(K=8\) at short lookahead has neither service headroom nor
   complete prediction.
2. **Transfer limited:** \(K=32,\Delta=1\)–2 has roughly 63% complete prediction,
   but the trace-driven oracle still fails.
3. **Prediction limited:** \(K=16,\Delta\ge9\) is oracle-feasible, but complete
   coverage remains below 30%.
4. **Candidate co-design region:** \(K=32,\Delta=3\)–6 passes the oracle and
   exceeds 50% complete coverage for both policies. The linear sidecar remains
   above the descriptive coverage boundary at longer horizons, while the
   transition table falls below it after \(\Delta=6\).

“Candidate” is intentionally weaker than “profitable.” Profit requires an
actual policy replay, learned/oracle recovery, speculative-byte accounting,
and measured concurrent copy/compute overlap.

## Limitations

- one OLMoE checkpoint with top-8 routing;
- batch-1 decode only;
- a single RTX 3090 Ti host/GPU platform;
- one serialized copy engine and FIFO oracle transfer order;
- effective-average layer timing, not individual source-target intervals;
- no measured concurrent-copy slowdown or live execution from copied weights;
- per-layer LRU residency, not a global byte allocator;
- target layers without a same-token source at the chosen lookahead are
  excluded;
- no end-to-end TPOT or latency-speedup claim.

## Artifacts

- Protocol: [H4_PROTOCOL.md](H4_PROTOCOL.md)
- Machine summary: `analysis/h4/summary.json`
- Gate: `analysis/h4/gate.json`
- Calibration: `analysis/h4/measurement.json`
- Oracle grid: `analysis/h4/oracle_metrics.csv`
- Figures/review: `analysis/h4/figures/FIGURES.md`
- Co-design points: `analysis/h4/codesign/codesign_points.csv`
- Co-design regime map: `analysis/h4/codesign/figures/`
