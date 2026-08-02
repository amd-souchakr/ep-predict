# MI355X H4 calibration results

**Calibration run:** `artifacts/runs/mi355x-h4-calibration/analysis/h4`  
**Demand run:** `artifacts/runs/mi355x-h4-decode`  
**Evidence grade:** calibrated single-MI355X oracle replay  
**Formal decision:** `PILOT_SUPPORTS`  
**Human figure review:** completed 2026-08-01 with requested revisions

## Plain-language conclusion

The unchanged H4 whole-expert gate passes on one MI355X. At the frozen
`K=16`, measured-bandwidth scope, every short horizon passes both 50%
thresholds. The best primary point, delta 3, completes 83.9% of cold expert
bytes by demand and removes 86.5% of modeled reactive stall. The historical
RTX 3090 Ti result at the same point was 32.8% and 38.9%.

This reversal is a calibrated regime-space result, not a predictor result or
an end-to-end speedup. The testbed moves the exact 12 MiB expert 1.84 times
faster, while its measured batch-1 cached-token OLMoE forward is 52.9% slower.
Given MI355X's much greater peak compute and HBM bandwidth, the forward-time
gap is most plausibly a software-stack or small-batch kernel artifact of this
Transformers/PyTorch/ROCm testbed; it must not be presented as an inherent
MI355X hardware characteristic. Faster copies are sufficient to pass at delta
2 and 3 in a fixed-demand factorial replay. The extra measured forward slack
makes all three horizons pass and is useful for exploring the architecture
regime, even though this project is not claiming wall-clock benefit.

## Execution and integrity

- The runtime exposed exactly one AMD Instinct MI355X with `gfx950`, ROCm 7.2,
  PyTorch 2.11.0, and Transformers 5.14.1.
- The full 128-request standard workload produced 8,012 decode tokens and
  128,192 decode layer-waves. All requests completed with zero router
  validation mismatches.
- The demand trace uses the same prompt file, order, 384-token cap, greedy
  decoding, and 64-token generation limit as the historical H1 run.
- Timing was collected separately with no hooks installed. The measurement
  records the pinned model/tokenizer revision and prompt SHA-256.
- The analyzer enforces the configured MI355X trace path, so the substantive
  result cannot silently substitute the unavailable NVIDIA raw trace.

## Direct calibration comparison

| Quantity | RTX 3090 Ti | MI355X | MI355X / RTX |
|---|---:|---:|---:|
| Cached-token forward median | 10.229 ms | 15.638 ms | 1.529x |
| Effective interval per 16 MoE layers | 0.639 ms | 0.977 ms | 1.529x |
| Exact 12 MiB H2D event median | 0.524 ms | 0.285 ms | 0.544x |
| Exact 12 MiB wall-completion median | 0.531 ms | 0.300 ms | 0.565x |
| Fitted pinned H2D bandwidth | 24.14 GB/s | 44.69 GB/s | 1.852x |
| Serialized copies per effective layer interval | 1.22 | 3.43 | 2.81x |

The decode P10--P90 interval is 15.251--15.858 ms. Per-domain medians span
15.571--15.725 ms. The exact-copy P10--P90 interval is 0.284--0.286 ms, so
measurement dispersion is small relative to the gate margin.

## Frozen primary gate

Decode, measured bandwidth, `K=16`:

| Lookahead | Resident-hit bytes | On-time cold bytes | Oracle stall reduction | Stalled waves |
|---:|---:|---:|---:|---:|
| delta 1 | 55.5% | 68.8% | 74.8% | 49.6% |
| delta 2 | 56.4% | 78.7% | 82.7% | 30.0% |
| delta 3 | 56.8% | 83.9% | 86.5% | 20.4% |

All three short horizons pass. At delta 3, 360,011 of 833,248 eligible
expert occurrences are cold. Only 832 are compulsory; 359,179 are capacity
misses. Residency pressure remains the source of cold demand even though the
link now services most of it before deadline.

## What changed

Holding the new MI355X demand trace fixed gives the following descriptive
factorial result at `K=16`:

| Calibration substitution | delta 1 | delta 2 | delta 3 |
|---|---:|---:|---:|
| RTX layer time + RTX copy | 26.6% / 31.7% | 29.8% / 35.8% | 32.8% / 38.9% |
| RTX layer time + MI copy | 50.0% / 54.6% | 56.3% / 61.4% | 61.4% / 65.7% |
| Measured testbed layer time + RTX copy | 26.6% / 46.8% | 36.0% / 52.7% | 43.0% / 56.7% |
| Measured testbed layer time + MI copy | 68.8% / 74.8% | 78.7% / 82.7% | 83.9% / 86.5% |

Cells report on-time cold bytes / stall reduction. The RTX calibration on the
MI demand reproduces the historical result to rounding, which is strong
evidence that the decision reversal comes from the measured platform timing
rather than routing-demand drift. The MI copy alone passes at delta 2 and 3.
Extra testbed layer slack alone improves stall but never satisfies both gate
metrics.

## Broader boundary and insights

- At measured bandwidth, `K=8`, delta 3 already reaches 72.0% on-time cold
  bytes and 76.0% stall reduction. `K=32`, delta 3 reaches 95.9% and 96.7%.
- Halving measured MI bandwidth removes the primary pass: at `K=16`, delta 3,
  stall reduction remains 53.0% but on-time cold bytes fall to 40.4%. The
  result therefore still has a real bandwidth threshold rather than being
  insensitive to link performance.
- A passing byte/stall gate does not eliminate tails. Even at the best primary
  point, 20.4% of eligible waves retain some modeled stall.
- The observed 44.69 GB/s fit is host/platform-specific pinned-memory evidence,
  not a generic MI355X fabric rating.
- The result strengthens the physical oracle ceiling only. It does not repair
  the previously measured complete-set reliability or speculative-byte
  amplification of the transition and linear predictors.

## Decision boundary

Milestone B supports whole-expert short-horizon feasibility on this calibrated
MI355X setup. It does not establish live copy/compute overlap, execution from
transferred expert buffers, end-to-end TPOT improvement, or superiority to
all-resident execution. The 15.638 ms all-resident timing is treated as a
testbed/software-stack artifact for the present regime-space exploration, not
as a comparative statement about inherent MI355X compute performance.

The researcher accepted the regime-space result with a narrowed interpretation:
the forward-time gap is likely a software/testbed artifact, not an inherent
MI355X characteristic. The requested plain-language figures retain the full
grid and both formal metrics. Milestone B is closed; Milestone C qualification
is next but has not started.

## Figure reading guide

The two final views use plain operational language while retaining the full
data:

1. The grid shows all 63 combinations of GPU-resident capacity, advance
   notice, and copy-speed sensitivity. Every cell reports waiting removed; the
   black outline marks the frozen decision cells.
2. The measured-link chart shows both formal metrics separately--needed data
   arriving on time and waiting removed--for every capacity and lookahead. It
   does not average capacities, horizons, or the two metrics together.

## Artifacts

- Machine summary: `analysis/h4/summary.json`
- Frozen gate: `analysis/h4/gate.json`
- Calibration and raw samples: `analysis/h4/measurement.json`,
  `decode_timing_samples.csv`, and `transfer_samples.csv`
- Direct comparison: `analysis/h4/calibration_comparison.csv`
- Calibration-factor attribution: `analysis/h4/calibration_attribution.csv`
- Oracle grid: `analysis/h4/oracle_metrics.csv`
- Figures and review checklist: `analysis/h4/figures/`
