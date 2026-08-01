# H4 oracle hardware-feasibility protocol

**Frozen:** 2026-08-01  
**Evidence grade:** calibrated single-GPU pilot  
**Model:** pinned OLMoE-1B-7B-0125-Instruct BF16 checkpoint  
**Demand trace:** immutable 128-request H1 trace  
**Primary phase:** batch-1 decode

## Decisive question

Can perfect future knowledge move exact 12 MiB experts from pinned host memory
to the GPU before demand and remove a meaningful fraction of cold-expert stall?

H4 tests the physical mechanism, not prediction quality. No model, router, or
Transformers source is modified. Hooked inference contributes demand only;
timing is measured in a separate run with no hooks installed.

## Frozen model

- One serial host-to-device copy engine.
- Exact inspected BF16 expert size: 12,582,912 bytes (12 MiB).
- Effective inter-MoE-layer budget: median steady-state cached-token model
  forward time divided by the 16 MoE layers. This is deliberately labeled an
  effective average, not a per-layer kernel profile.
- Transfer cost at 12 MiB: measured pinned-host asynchronous copy completion
  time. Bandwidth sensitivity scales only its size-dependent component.
- Per-layer LRU fast tier with capacities 8, 16, and 32 experts.
- Existing requests are replayed in their recorded order from an empty cache.
- A cold occurrence is an expert absent from that layer's LRU tier immediately
  before its demand wave. It is reported separately as a compulsory first use
  or a capacity-eviction miss. Reuse hits require no transfer.
- An oracle issued at source layer \(n\) knows the exact cold expert set at
  target layer \(n+\Delta\). Transfers share one FIFO engine; synchronous
  target demand waits for the latest required expert.
- Metrics cover target layers having a same-token source at the selected
  lookahead. Earlier ineligible layers are excluded rather than granting them
  an unmeasured previous-token predictor.

The simulator is an analytical whole-expert mechanism test. It does not claim
end-to-end model speedup, model execution from transferred buffers, concurrent
kernel/copy validation, or multi-GPU behavior.

## Measurements

The timing calibrator uses four revision-pinned prompts, one per domain:

1. run an ordinary cached-token decode chain with no hooks;
2. discard five warmup steps and retain 20 CUDA-event timings per prompt;
3. report the median full decode-forward time and divide it by 16 for the
   effective inter-layer budget;
4. measure 4, 8, 12, 16, and 24 MiB pinned-host asynchronous copies, with 10
   warmups and 50 measured copies per size;
5. retain raw samples, medians, and a nonnegative-startup linear fit.

## Grid

- Lookahead: \(\Delta \in \{1,2,3,6,9,12,15\}\).
- Per-layer capacity: \(K \in \{8,16,32\}\).
- Transfer sensitivity: \(0.5\times,1\times,2\times\) measured effective
  bandwidth, with measured startup retained.

## Required metrics

- total and resident-hit demanded bytes;
- cold demanded bytes, split into compulsory and capacity-eviction misses;
- cold bytes completed by their deadline;
- late cold bytes;
- deadline-feasible cold-byte fraction;
- reactive and oracle synchronous-wave stall;
- oracle stall reduction.

All byte counts refer to unique experts in a token-layer demand wave, not the
eight weighted selections counted with duplicates.

## Frozen gate

At measured bandwidth and \(K=16\), H4 passes only if at least one already
established short horizon \(\Delta \in \{1,2,3\}\) achieves both:

- at least 50% of cold bytes completed by deadline; and
- at least 50% reduction in reactive cold-expert stall.

The longer-horizon and capacity/bandwidth scans are descriptive and cannot
retroactively change this gate.

- **Pass:** overlay the existing H2 transition and H3 linear candidate streams
  in the viable region, without retraining or tuning.
- **Fail:** stop the same-token whole-expert PCIe latency-hiding direction for
  this checkpoint. Retain routing prediction for residency, replication,
  bandwidth reservation, activation movement, or smaller-granularity studies.

## Figures and review

Create exactly:

1. an oracle feasibility heatmap over lookahead and bandwidth, with capacity
   shown as compact panels;
2. a measured-bandwidth stall-reduction curve over lookahead and capacity.

The generated figure note is the human review checkpoint required before H5.
