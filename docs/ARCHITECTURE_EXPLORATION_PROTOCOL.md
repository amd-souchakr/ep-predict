# Predictive expert-memory architecture exploration

**Track:** AX — assumption-driven workload/software/hardware co-design  
**Status:** ready; protocol and sweep frozen, implementation not yet run  
**Primary input:** retained OLMoE decode traces and H4 transfer calibration  
**New inference or model training:** none  
**Primary config:** `configs/experiment/ax-future-predictor-architecture.toml`

## Purpose

This track asks what memory hierarchy would become attractive if a future MoE
router exposed accurate multi-horizon expert-demand predictions, analogous to
an MTP head for routing. It deliberately assumes that an appropriate training
objective can improve routing-trajectory predictability without degrading
language-model loss or load balance.

The objective is to derive quantitative architectural requirements, feasible
regions, capacity benefits, and workload/software/hardware co-design
principles. It is **not** to demonstrate wall-clock speedup on the current
OLMoE checkpoint or RTX 3090 Ti.

## Evidence contract

Every output must label each input as one of:

1. **Measured:** exact 12 MiB OLMoE expert size, 0.639 ms effective MoE-layer
   interval, and the measured host-to-device transfer curve.
2. **Trace-derived:** demanded expert sets, reuse, cold demand, layer/domain
   structure, and queue burstiness from the retained OLMoE workload.
3. **Assumed predictor:** complete cold-set coverage and false-positive
   amplification of a hypothetical trained multi-prediction router.
4. **Hypothetical hardware:** tier bandwidth, startup latency, capacity,
   concurrency, transfer granularity, and SLO slack.

Projected points must never be described as measured speedups or as evidence
that current OLMoE produces the assumed predictor quality. Existing H2/H3
streams may be plotted as reference markers but do not constrain the future
predictor sweep.

## Predictor abstraction

For a target execution wave:

- \(D\): demanded nonresident experts or transfer objects;
- \(P\): predicted nonresident objects;
- \(R\): already resident objects;
- \(C=P(D\subseteq R\cup P)\): complete cold-set coverage;
- \(A=|P|/|P\cap D|\): predicted/useful byte amplification.

Complete-set coverage and amplification are independent sweep axes. Ordinary
per-expert recall is secondary because one false negative may stall the token
or synchronous batch.

Generate synthetic predictor streams from oracle demand at the **wave** level:

1. select the configured fraction of waves for complete coverage;
2. introduce correlated false negatives into the remaining waves;
3. add unused nonresident candidates to reach the configured amplification;
4. suppress objects already resident or already in flight;
5. use deterministic seeds and record the realized coverage and amplification.

Do not independently flip expert labels: that would create unrealistically
benign complete-set and tail behavior. Report the idealized oracle-corruption
model as an assumption, not a trained predictor.

## First-order model

For tier \(i\rightarrow j\):

\[
T_{\mathrm{move}}(S)=\alpha_{ij}+\frac{S}{\beta_{ij}}+Q_{ij}.
\]

The primary bandwidth condition is cold-service pressure:

\[
\rho =
\frac{A\,\bar N_{\mathrm{cold}}\,S}
{\beta_{\mathrm{tier}}\,\Delta T_{\mathrm{layer}}}.
\]

The isolated-object timing ratio is:

\[
\gamma =
\frac{\alpha_{\mathrm{tier}}+S/\beta_{\mathrm{tier}}}
{\Delta T_{\mathrm{layer}}}.
\]

\(\rho<1\) is necessary mean headroom, not sufficient tail feasibility.
Trace replay must retain bursts, queueing, deadlines, false positives, and
correlated false-negative waves. Report the inverse requirements:

\[
\beta_{\min},\quad
C_{\min},\quad
A_{\max},\quad
S_{\max},\quad
K_{\min}.
\]

## Profitability semantics

Keep three claims separate:

- **Capacity viable:** the hierarchy runs a model whose full expert weights do
  not fit in the fast tier while satisfying an explicit slowdown or TPOT SLO.
- **Performance profitable:** prediction improves latency or throughput over
  the same hierarchy with reactive loading.
- **SLO safe:** the modeled P95/P99 wave or token latency remains within the
  configured bound.

CPU-memory prefetch cannot be claimed faster than an otherwise identical
all-HBM execution. Its value is larger model capacity and lower slowdown than
reactive offload. HBM-to-SRAM staging may additionally reduce HBM bandwidth or
energy pressure, but energy remains a byte-traffic proxy until calibrated.

## AX1 — Capacity-first host or pooled-memory prefetch

### Question

Under optimistic future-router quality, how much expert capacity can move from
GPU HBM to host or pooled memory while retaining an acceptable TPOT envelope?

### Primary replay

- Decode only; preserve empirical wave order and cold-demand burstiness.
- \(K\in\{8,16,32\}\) resident experts per layer.
- \(\Delta\in\{1,2,3,6,9,12,15\}\).
- Complete cold-set coverage
  \(C\in\{0.50,0.75,0.90,0.95,0.99,0.999\}\).
- Amplification \(A\in\{1,1.25,1.5,2,4,8\}\).
- Cold-tier bandwidth from 16 through 512 GB/s, including the measured
  24.14 GB/s point.
- Startup latency from 0.1 through 20 microseconds, including the measured
  fit.

### Outputs

- HBM expert bytes retained and offloaded;
- required bandwidth and lookahead;
- useful, false, late, and missed bytes;
- mean/P95/P99 residual stall and TPOT relative to reactive hierarchy;
- deadline-feasible wave fraction and oracle recovery;
- a capacity-versus-P99 Pareto frontier.

The primary current-system anchor is measured PCIe, not a success gate.

## AX2 — Latency, bandwidth, reliability, and granularity regimes

### Question

Which failure mode dominates each design point: bulk bandwidth, startup
latency, prediction reliability, speculative traffic, or staging capacity?

### Sweep

Reuse AX1 demand while varying:

- transfer object \(S\in\{12,4,1,0.25\}\) MiB;
- bandwidth from host-class through HBM-class values;
- startup latency;
- transfer concurrency;
- complete coverage and amplification;
- unique cold objects per wave \(U\in\{1,2,4,8\}\) as a normalized
  top-1/top-2/top-8 sensitivity, clearly separated from measured OLMoE.

Classify each cell:

1. bandwidth limited;
2. latency limited;
3. reliability limited;
4. staging-capacity or pollution limited;
5. feasible but not SLO safe;
6. analytically profitable and SLO safe.

Report required per-object recall only as an intuition under an explicitly
independent approximation:

\[
r_{\min}=(C_{\mathrm{wave}})^{1/U}.
\]

The simulator continues to use complete wave coverage directly.

## AX3 — Three-tier predictive hierarchy

### Question

What division of responsibility is suitable across pooled/host memory, local
HBM, and a future software-managed SRAM expert staging tier?

### Architecture

- Long-horizon MTP heads plan expert migration or replication into local HBM.
- Short-horizon heads refine matrices, blocks, or tiles staged into SRAM.
- The ordinary router confirms final demand.
- A bounded scheduler uses object ID, target layer, deadline, confidence,
  residency, and cancellation generation.
- A low-latency reactive or remote-execution path handles false negatives.

Sweep global SRAM staging capacities of 32, 64, 128, 256, and 512 MiB. Do not
model these as \(K\) persistent experts for every layer: SRAM is a rolling,
double-buffered staging store shared by upcoming layers. Charge false
positives against both tier bandwidth and SRAM occupancy.

Compare:

- two-tier reactive hierarchy;
- two-tier ideal predictive prefetch;
- three-tier multi-horizon predictive staging;
- oracle staging;
- all-resident reference where it fits.

Report fast-tier bytes, pooled/local traffic, staging churn, deadline misses,
P99 stall, and the maximum model expert capacity admitted by each SLO.

## Required figures

Create at most three primary figures:

1. **Profitability phase map:** complete cold-set coverage versus
   cold-service headroom \(1/\rho\), with compact amplification panels and
   architecture markers.
2. **Memory–latency Pareto frontier:** fast-tier capacity versus modeled P99
   TPOT or slowdown for reactive, predictive, oracle, and all-resident
   references.
3. **Inverse interconnect curve:** minimum bandwidth versus lookahead for
   selected capacities, amplifications, and transfer granularities.

Every plot must distinguish measured anchors, trace-driven projections, and
hypothetical architecture points. Save canonical CSV/JSON, PDF, PNG, a figure
manifest, and a human review note under one analysis directory.

## Interpretation and stop rules

This is an envelope study, not a binary validation of future training. Its
decisive products are quantitative bounds and regime classifications.

- If no feasible region exists even at \(C=99.9\%\) and \(A=1\), reject that
  hierarchy or transfer granularity rather than blaming the predictor.
- If feasibility requires implausibly perfect coverage but modest bandwidth,
  classify the design as reliability limited.
- If feasibility requires \(A\) near one but tolerates misses, identify
  selective admission as the co-design requirement.
- If whole experts fail but smaller objects pass, derive the maximum viable
  transfer granularity rather than claiming a particular tiling implementation.
- Run live asynchronous validation only after a representative point is
  selected to calibrate a conclusion. It is not required for the analytical
  architecture result.
- A later top-1/top-2 model trace is confirmation, not a prerequisite for this
  assumption-driven sweep, and still requires explicit permission.

## Lean execution sequence

1. Implement AX1 by extending the existing H4/H5 replay and artifact schema.
2. Review the capacity–P99 frontier and the realized synthetic predictor
   integrity table.
3. Add AX2 granularity and latency axes only after AX1 reproduces the H4/H5
   anchors.
4. Add AX3 using the same event model and a rolling staging-capacity state.
5. Synthesize architectural requirements and only then select any physical
   validation point.

