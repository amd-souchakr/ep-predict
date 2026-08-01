# H5 first-order co-design protocol

**Frozen:** 2026-08-01  
**Evidence grade:** trace-driven analytical pilot  
**Model:** pinned OLMoE-1B-7B-0125-Instruct BF16 checkpoint  
**Data:** immutable H1 routing trace, H3 projected features and predictors, and
H4 calibration  
**Primary phase:** held-out batch-1 decode

## Decisive question

Do any combinations of predictor quality, speculative traffic, fast-tier
capacity, lookahead, and normalized cold bandwidth create a useful analytical
window, and do the existing transition or fixed linear candidate streams enter
that window without retraining?

H5 is a requirements and screening study. It is not an end-to-end latency
forecast. It performs no inference, model modification, predictor training, or
inference-library modification.

## Frozen first-order model

For each capacity and lookahead, replay the existing trace only to obtain the
mean number of nonresident demanded experts per eligible wave. Define
cold-service headroom:

\[
H =
\frac{B_{\mathrm{scale}}\Delta T_{\mathrm{layer}}}
{\bar N_{\mathrm{cold}}T_{\mathrm{copy}}}.
\]

The first-order oracle ceiling is:

\[
R_{\mathrm{oracle}}=\min(1,H).
\]

Let \(C\) be complete-cold-set prediction coverage and \(A\) be transferred
candidate bytes divided by useful predicted cold bytes. Under proportional
FIFO service, the modeled stall reduction is:

\[
R_{\mathrm{policy}}
=
C\min\left(1,\frac{H}{AC}\right)
=
\min\left(C,\frac{H}{A}\right).
\]

Oracle recovery is \(R_{\mathrm{policy}}/R_{\mathrm{oracle}}\). This is an
optimistic normalized benefit proxy: a completely and timely covered cold wave
is credited in proportion to reactive cold stall. It deliberately omits
kernel overlap, layer heterogeneity, eviction caused by prefetch, and
end-to-end execution.

The controlled sweep treats \(C\) and \(A\) as independent assumptions. In
H5-C, both are measured from actual candidate streams after filtering already
resident candidates.

## Sweep

- Complete-cold-set coverage: 0% to 100% in 1-point increments.
- Candidate transfer amplification: 1×, 2×, and 4×.
- Per-layer LRU capacity: 8, 16, and 32 experts.
- Same-token lookahead: every \(\Delta=1\ldots15\).
- Cold bandwidth: 0.25×, 0.5×, 1×, 2×, and 4× the measured H4 rate.

Expert size and bandwidth remain collapsed into the measured transfer-time
normalization.

## Frozen gate

A controlled assumption cell is analytically profitable only if all hold:

- at least 25% modeled reactive-stall reduction;
- at least 50% first-order oracle recovery;
- no more than 2× transferred candidate bytes per useful predicted cold byte.

These are prototype screening thresholds, not speedup claims.

## Existing-policy placement

Use the unchanged 96/32 H2/H3 request split. Reconstruct transition candidates
from the training requests and load the already-trained fixed linear heads.
Evaluate only test requests while replaying the full request order for LRU
state.

Representative measured-bandwidth cells:

- K=32, Δ=1: prediction-good/physics-limited control;
- K=32, Δ=3: short boundary;
- K=32, Δ=9: long-range linear regime;
- K=16, Δ=9: oracle-feasible/prediction-limited control.

For every candidate list, suppress experts already resident in the target
layer. Count the remaining demanded candidates as useful transfers and every
other nonresident candidate as false transfers. Report cold-expert coverage,
complete-cold-set coverage, useful/false/late bytes, transfer amplification,
candidate churn, modeled stall reduction, and oracle recovery.

## Outputs and stop rule

Create machine-readable design points, viable windows, inverse requirements,
and actual-policy placement before exactly two primary figures:

1. categorical profitability phase diagram;
2. minimum complete-cold-set coverage versus lookahead.

After the figures, stop for human review. Do not begin H7, C1, new inference,
new model setup, MLP tuning, or timing-fidelity work.
