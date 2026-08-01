# AX assumption-driven architecture exploration

**State:** complete; pending human figure review

## Plain-language result

A predictive hierarchy has a real design window, but reliability and speculative traffic are first-class hardware parameters. More lookahead trades prediction difficulty for lower required bandwidth; it does not make false-negative tail stalls disappear.

## AX1 — host or pooled memory to HBM

At the measured PCIe anchor and the assumed C=99%, A=1.5× future router, the best modeled points are:

| HBM residents/layer | HBM expert GiB | Best Δ | P99 TPOT | Reactive P99 | Improvement |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.5 | 9 | 48.03 ms | 72.60 ms | 33.8% |
| 16 | 3.0 | 9 | 43.27 ms | 66.83 ms | 35.3% |
| 32 | 6.0 | 6 | 30.71 ms | 50.58 ms | 39.3% |

These are capacity-enabling comparisons against reactive offload on the same hierarchy, not speedups over all-HBM execution.

The selected FCFS queue replay is intentionally more pessimistic than the wave-local envelope:

| K | Δ | C | A | BW | Wave-local P99 stall | Queue P99 stall |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3 | 0.999 | 1.00× | 24.1 GB/s | 31.67 ms | 47.01 ms |
| 16 | 9 | 0.990 | 1.50× | 24.1 GB/s | 33.04 ms | 65.23 ms |
| 32 | 3 | 0.990 | 1.50× | 24.1 GB/s | 33.59 ms | 48.06 ms |
| 16 | 3 | 0.990 | 1.50× | 64.0 GB/s | 7.34 ms | 20.91 ms |

## AX2 — inverse requirements

For trace-derived K=16 cold demand, 12 MiB objects, A=1×, and one transfer lane, the minimum first-order bandwidth falls with lookahead:

| Δ | Minimum bandwidth |
|---:|---:|
| 1 | 71.3 GB/s |
| 3 | 22.8 GB/s |
| 6 | 11.6 GB/s |
| 9 | 8.2 GB/s |

This inverse law is the central co-design lever: required bandwidth scales approximately as A/Δ. Coverage is orthogonal: it controls how often the synchronous tail still takes the cold path.

## AX3 — rolling SRAM staging

OLMoE's top-8 route carries 96 MiB of whole-expert weights per layer. A rolling double buffer therefore needs at least 192 MiB at A=1× and 384 MiB at A=2×. This makes capacity pollution, not raw SRAM bandwidth, the first constraint for the frozen 32–512 MiB range.

1429 of 7200 factorized staging cells pass both timing and double-buffer capacity. A passing cell only shows that warming can finish; no SRAM compute-time or energy benefit is claimed without an execution model.

A compact cross-hierarchy comparison conservatively adds separate upstream and SRAM-staging P99 values and credits no SRAM compute benefit:

| Architecture | HBM expert GiB | SRAM MiB | Predictor | P99 TPOT |
|---|---:|---:|---|---:|
| two_tier_reactive_host_hbm | 3.0 | 0 | none | 66.83 ms |
| two_tier_predictive_host_hbm | 3.0 | 0 | C99_A1.5 | 43.27 ms |
| three_tier_predictive_host_hbm_sram | 3.0 | 512 | C99_A1.5_both_tiers | 43.78 ms |
| two_tier_oracle_host_hbm | 3.0 | 0 | C1_A1 | 31.74 ms |
| all_resident_hbm_reference | 12.0 | 0 | not_applicable | 10.23 ms |

## Most important insights

1. Lookahead buys bandwidth almost linearly, while amplification spends it almost linearly.
2. P99 is reliability-limited before the mean link is saturated: a 99% complete-wave predictor still exposes the one-percent tail.
3. Whole-expert SRAM staging is plausible only with hundreds of MiB for top-8 routing; top-1/top-2 or selective sub-expert staging changes this capacity bound much more than another small bandwidth gain.
4. Queue replay can be materially worse than mean or wave-local headroom. Architecture claims should use the phase map for bounds and the queue points as a tail-risk warning.

## Interpretation boundary

Projected points combine measured current-testbed anchors, trace-derived demand, assumed future predictor quality, and hypothetical hardware. They are not measured speedups and do not show that current OLMoE attains the assumed quality.
