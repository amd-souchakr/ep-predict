# MI355X OLMoE parity and AMD-baseline protocol

**Frozen:** 2026-08-01  
**State:** Milestones A--B reviewed; Milestone C qualified pending review
**Model:** `allenai/OLMoE-1B-7B-0125-Instruct`, revision
`caada7d7b70f4b852b14108479e0812223a8794f`  
**New platform:** one visible AMD MI355X (`gfx950`), ROCm 7.2, PyTorch 2.11  
**Historical platform:** one NVIDIA RTX 3090 Ti, CUDA 12.4, PyTorch 2.6  

> **Execution amendment (2026-08-01):** The researcher confirmed that the
> NVIDIA request-level traces were not retained and cannot be restored.
> Milestone A therefore cannot establish record-level cross-platform parity or
> trace interchangeability. Its revised evidence contract is: (1) MI355X
> input/router integrity on the frozen 16-request prefix; and (2) descriptive
> comparison of H1 skew/hot-expert and H2 transition/horizon trends against
> the preserved 128-request NVIDIA CSV/JSON artifacts. The workloads are
> nested but not sample-matched, so differences combine platform numerical
> effects with finite-sample/workload effects. No derived-trend result may be
> called proof that raw traces are interchangeable.

> **Matched-workload amendment (2026-08-01):** Review of the 16-request result
> found a noticeable absolute H1 skew offset despite closely matched H2
> horizon curves. The 16-request tracer bullet remains preserved, but it is
> not the final Milestone A comparison. Run all 128 prompts with the exact C0
> raw serialization, one-token generation, H1 settings, 96/32 H2 split, and
> Δ=1..15 analysis used by the NVIDIA reference. This removes sample count,
> prompt composition, and split size as explanations for derived differences.
> It still cannot establish record-level route parity without NVIDIA traces.

## Decision rule

Follow an **AMD baseline + model generalization** sequence. Execute and review
one milestone at a time. Do not begin the next milestone until its predecessor
has a recorded result and the researcher explicitly approves the transition.

The next milestone is deliberately small:

> Can the pinned Transformers implementation expose GPT-OSS router decisions
> that are proven identical to the expert IDs actually dispatched, including
> any MXFP4 or custom-kernel path?

This is a model-specific instrumentation gate, not a routing-distribution
experiment or performance benchmark. It must succeed before a GPT-OSS trace
can be treated as workload evidence.

## Project position entering the AMD phase

The project asks whether future MoE expert demand is structured and
predictable enough to change the design of a hierarchical expert-memory
system. It evaluates complete demanded sets, timing, bandwidth, capacity,
critical-path stalls, and alternatives such as activation movement rather than
treating classification accuracy as the objective.

The OLMoE pilot established the following scoped evidence:

- H1 found strong domain-conditioned skew but rejected one universal static
  hot tier.
- H2 found that held-out layer-transition tables beat marginal popularity at
  the tested horizons.
- H3 rejected a fixed linear hidden-state sidecar as a universal next-layer
  replacement, while an all-layer scan found early-source/long-horizon value.
- H4 rejected the preregistered RTX 3090 Ti whole-expert point at
  `K=16, delta=1..3`; capacity, lead time, and bandwidth created a broader
  descriptive region.
- H5 found an analytical profitability region, but the unchanged predictors
  required too much speculative traffic.
- H6 showed that depth-trajectory prediction did not improve temporal
  residency over strong static/domain/LRU baselines.
- C0 showed that Base and Instruct retain 89.7% of selected expert IDs under
  matched inputs, so the trajectory scaffold is largely preserved within this
  training lineage.
- AX1--AX4 converted those observations into explicit future architecture
  contracts. Their positive regions depend on assumed predictor quality,
  high-bandwidth hardware, or graceful expert degradation; they are not
  measured end-to-end speedups.

The durable thesis is therefore narrower than the original prefetch idea:
routing structure is real, but prediction matters architecturally only when
complete-set reliability, transfer pressure, residency, and deadlines are
jointly satisfied. The MI355X phase first replaces the hardware calibration,
then tests whether this conclusion survives a substantially different routing
geometry.

### Experimental and co-design philosophy

The project treats MoE inference as placement and movement of three different
state classes: request-owned KV state, shared sparse expert weights, and
transient activations. It does not presume that moving expert weights is the
right answer; expert movement must beat additional residency, replication, or
moving activations to already resident experts.

Its experimental discipline is correspondingly narrow:

- freeze one decisive question and gate at a time;
- validate actual dispatch semantics before scaling collection;
- keep prefill and decode, layers, domains, and expert namespaces separate;
- evaluate complete token/wave/step sets and P95/P99 behavior rather than
  hiding correlated misses behind mean per-expert accuracy;
- test an oracle physical ceiling before optimizing a learned policy;
- report inverse requirements and dimensionless ratios, not only forward
  point estimates;
- distinguish measured calibration, trace-derived demand, assumed predictor
  behavior, and hypothetical hardware in every co-design claim;
- generate machine-readable tables and scripted figures, then stop for human
  review before expanding the hypothesis tree.

This is a research instrument and rapid tracer-bullet workflow, not a
production serving stack. Hooks establish workload evidence but invalidate
unqualified timing claims. Simulators identify boundaries, not end-to-end
speedups.

### Testbed and publication readiness

At protocol freeze, all substantive empirical routing evidence came from
OLMoE on the RTX 3090 Ti, and C0 added a second checkpoint within the same
Base--Instruct lineage rather than a second model architecture. Milestones A
and B now add full-checkpoint MI355X routing and calibration while retaining
the same OLMoE architecture. The scale-up host offers eight MI355X GPUs with
288 GB HBM each; baseline and C1 experiments intentionally expose only one GPU
so platform and model effects are not confounded with expert/tensor parallelism.

The project already has most of the mechanics of a strong systems paper:
frozen protocols, revision-pinned workloads, integrity-checked traces,
machine-readable result tables, negative as well as positive gates, scripted
figures, and quantitative architecture envelopes. The central publication gap
is external validity: one top-8 model family and one historical hardware
calibration cannot support a general MoE or MI355X conclusion. C0 and AX4 also
still have recorded human-review checkpoints pending.

The hardware proposal is therefore at the trace-calibrated analytical stage.
It has useful quantitative targets for HBM residency, transfer bandwidth,
rolling SRAM capacity, deadline-aware DMA, and fallback semantics, but lacks
cross-model routing geometry, live copy/compute overlap, execution from
transferred expert state, and language-quality evidence under expert
degradation. Milestone B supplied the MI355X calibration; the sequence below
now targets cross-model validity before broader mechanism work.

## Long-term milestone sequence

Only the first incomplete milestone is active.

### Milestone A -- OLMoE cross-platform parity (complete)

Collect 16 raw-prefill requests under
`configs/experiment/mi355x-olmoe-parity.toml`. Establish exact input-token
provenance and hook/router integrity. Compare derived H1/H2 trends with the
historical NVIDIA artifacts without reusing either platform's raw trace as the
other platform's demand evidence.

**Review output:** machine-readable H1/H2 analyses, one compact derived-trend
comparison figure, and a concise result note.

### Milestone B -- MI355X H4 calibration (complete)

Run hook-free cached-token timing and pinned host-to-device transfer
calibration under `configs/experiment/mi355x-h4-calibration.toml`. Preserve
`artifacts/runs/h1-standard-small/analysis/h4/` as RTX 3090 Ti evidence. Write
all AMD measurements under the new `mi355x-*` directory.

Calibrate the same OLMoE geometry and unchanged H4 grid. Because Milestone A
cannot establish raw-trace interchangeability, any later replay of the old
NVIDIA demand trace is a counterfactual hardware-only sensitivity, not a
measured MI355X workload. A substantive AMD H4 result requires a new MI355X
standard decode trace.

**Review output:** decode-forward distribution, transfer samples and fit,
exact 12 MiB copy time, effective inter-MoE interval, H4 gate under the old
thresholds, and a direct old-versus-new calibration table. Hooked timing is
never used.

**Review decision:** accepted as platform-stack regime evidence. The slower
measured forward is treated as a likely software/testbed artifact, not an
inherent MI355X characteristic. Final figures use plain operational language
and retain every swept trend.

### Milestone C -- GPT-OSS Transformers qualification (qualified; review pending)

Add a model-specific qualification path; do not build a universal MoE
adapter. Before downloading the 120B weights:

1. pin the exact Transformers and checkpoint revisions;
2. identify the explicit router module and the code path that consumes its
   result;
3. prove that hooks observe the expert IDs actually dispatched, not a
   reconstructed or pre-filtered proxy;
4. test whether MXFP4 or custom kernels bypass ordinary module hooks;
5. record expert tensors, storage bytes, loaded bytes, compute dtype, shared
   experts, selected-weight normalization, and top-k ordering semantics;
6. keep vLLM out of the qualification and trace phases because module
   visibility is a primary experimental invariant.

Use `gpt-oss-20b` to exercise the implementation cheaply. A tiny synthetic or
configuration-only inspection may precede weight download, but no claim about
120B hook semantics follows until its implementation path is shown to be the
same or is independently validated.

**Execution result:** `QUALIFIED`. On the pinned 20B native MXFP4 path, the
dispatch-boundary hooks covered 24/24 routed layers and all 576 consumed
ID/weight pairs with zero mismatches. Ordinary router hooks fired zero times,
confirming the anticipated inline-router bypass. Stored/loaded expert bytes,
BF16 compute, top-k/dispatch ordering, selected-softmax semantics, absence of
shared experts, and complete source/kernel provenance are recorded in
`docs/GPT_OSS_MILESTONE_C_RESULTS.md`. Stop for review before Milestone D.

### Milestone D -- GPT-OSS 20B tracer bullet (blocked on C review)

Run the complete project SOP on a small workload: pinned checkpoint download,
inspection, tokenization, deterministic inference, routing and selected-score
trace capture, integrity analysis, compact routing tables, output capture, and
scripted visualization. This milestone qualifies the workflow, not the main
scientific claim.

Its exit condition is an end-to-end reproducible artifact chain with no hook
ambiguity and with every model-specific constant removed from, or explicitly
scoped within, the GPT-OSS path.

### Milestone E -- GPT-OSS 120B decisive C1 (blocked on D review)

Use `gpt-oss-120b` as the scientific comparison target. Its advertised
36-layer, 128-expert, top-4, 5.1B-active-parameter geometry and approximately
60.8 GiB MXFP4 checkpoint provide the intended contrast with OLMoE's 16
layers, 64 experts, and top-8 routing while fitting on one MI355X. Reconfirm
these properties from the pinned model config and loaded inspection report;
do not rely on prose metadata alone. See the
[OpenAI architecture overview](https://openai.com/index/introducing-gpt-oss/)
and [official model card](https://huggingface.co/openai/gpt-oss-120b).

Preregister only the following C1 comparison:

- H1 skew and window stability;
- H2 transition predictability through every available horizon;
- complete-route coverage at normalized capacities;
- candidate amplification `K/k`;
- resident fraction `K/E`.

Do not expand H3, remeasure model-specific H4, or replay H6 unless the routing
geometry produces a meaningful difference at this gate. In particular,
top-4 complete-route coverage must not be compared with top-8 OLMoE coverage
at the same absolute `K` without also reporting `K/k` and `K/E`.

### Milestone F -- publication and hardware proposal (blocked on E review)

Produce a cross-model synthesis that labels every conclusion as
single-testbed, replicated, architecture-dependent, or contradicted. The
publication target is a measured account of how routing geometry changes the
coverage/capacity contract, paired with a calibrated MI355X physical bound.

Advance the hardware proposal only where the combined evidence supports it.
Candidate long-term mechanisms remain:

- long-horizon placement in HBM and short-horizon rolling staging;
- deadline-aware, priority-isolated expert DMA;
- explicit commit/degradation telemetry and an always-resident fallback path;
- smaller or factorized transferable expert state;
- activation movement or replication where expert movement loses.

The proposal must separate measured MI355X facts, trace-derived demand,
assumed future-router behavior, and hypothetical hardware. It must compare
hierarchical execution with both reactive offload and all-resident execution;
it may not claim a production speedup from analytical replay alone.

## Milestone A frozen design

### Paired scope

- MI355X run ID: `mi355x-olmoe-parity`.
- NVIDIA reference run:
  `artifacts/runs/olmoe-instruct-c0-paired`.
- Workload: the first 16 records of the existing revision-pinned
  `h1-standard-small` prompt file.
- Serialization: raw checkpoint-independent text, matching C0.
- Maximum prompt length: 384 tokens.
- Generation: greedy, one new token, batch size one.
- Primary evidence: matched prefill token-layer records.
- Model and tokenizer revisions, BF16 dtype, SDPA, and Transformers 5.14.1
  remain pinned.
- One MI355X is exposed with `HIP_VISIBLE_DEVICES=0`; the model config remains
  `device = "cuda:0"` because PyTorch exposes ROCm through `torch.cuda`.

The one-token generation bounds cost and avoids interpreting divergent
free-running decode. It does not qualify decode-route parity or language
output equivalence.

### Reference-artifact prerequisite

The NVIDIA run definition, manifest, model report, and derived C0 evidence are
durable. Its request-level `trace/` is intentionally ignored by Git and is not
present in the current workspace. Exact cross-platform route agreement cannot
be reconstructed from aggregate C0 tables.

Before the parity comparison, restore the original NVIDIA trace shards for the
16 paired requests and verify that they match the preserved run definition and
request keys. Do not regenerate this reference on MI355X and call it an
NVIDIA baseline.

If the NVIDIA shards cannot be restored:

- the MI355X run may still establish local hook integrity;
- a second MI355X run may establish repeatability;
- cross-platform parity is `INCONCLUSIVE_NO_REFERENCE_TRACE`;
- old and new traces must remain platform-scoped and non-interchangeable.

### Required integrity checks

1. `verify-rocm` passes with exactly one visible `gfx950` device.
2. The loaded model reports 16 routers, 64 experts per router, top-8 routing,
   and 12,582,912 BF16 bytes per expert.
3. The prompt-file SHA-256, model revision, tokenizer revision, prompt format,
   truncation, generation arguments, and software versions are recorded.
4. All 16 paired requests have the same sample IDs, request order, token
   counts, input-token hashes, and per-position input token IDs.
5. Every forward observes exactly 16 router calls.
6. Every hook-captured selected-ID tuple agrees with top-k of that router's
   logits; missing routers and mismatches are both zero.
7. Cross-platform records have identical `(request, sample, phase,
   token_position, layer)` keys.
8. Report unordered selected-ID intersection, route Jaccard, exact route-set
   match, ordered top-k match, and selected-weight differences by layer and
   domain.

### Interchangeability gate

The routing trace is interchangeable for existing ID/set-based H1--H6 replay
only if all of the following hold across every paired prefill token-layer
record:

- zero input-token mismatches;
- zero router-integrity failures on each platform;
- identical routing geometry and record keys;
- selection agreement = 1.0;
- route Jaccard = 1.0;
- exact unordered top-8 route-set match = 1.0.

Also report ordered top-k equality and selected-weight maximum/mean absolute
difference. They are diagnostic for ID/set-based analysis. Reusing weighted
routes for AX4 additionally requires a separately recorded numerical tolerance
for aligned selected weights and normalized selected mass; do not infer that
permission from an ID-only pass.

Any nonzero route-set difference fails interchangeability. Preserve the clean
MI355X trace, quantify the disagreement by layer/domain and router-score
margin, and design the next comparison only after review. Do not weaken this
gate post hoc.

### Commands

Environment and model smoke:

```bash
HIP_VISIBLE_DEVICES=0 uv run ep-predict verify-rocm

HIP_VISIBLE_DEVICES=0 uv run ep-predict inspect \
  --config configs/model/olmoe-1b-7b-instruct.toml \
  --output artifacts/runs/mi355x-olmoe-parity/model-inspection.json
```

Collect exactly the frozen 16-request prefix:

```bash
HIP_VISIBLE_DEVICES=0 uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/mi355x-olmoe-parity.toml \
  --limit 16
```

The current repository does not yet expose a generic cross-platform parity
CLI. The comparison implementation must consume the two run directories,
write tables under
`artifacts/runs/mi355x-olmoe-parity/analysis/parity/`, and enforce the frozen
gate above. Do not substitute the C0 Base--Instruct gate, which has a different
scientific question and allows route changes.

### Stop and review

Stop after writing the parity summary and result note. Review:

- reference trace provenance;
- input-token equality;
- router integrity and observed dispatch semantics;
- layer/domain locations of any numerical route changes;
- whether the trace is ID/set-interchangeable, weighted-route-interchangeable,
  or platform-scoped.

Milestone B begins only after the researcher records that decision.

## Milestone B frozen calibration isolation

The calibration output boundary was fixed before Milestone A review. The
researcher's 2026-08-01 instruction that Milestone A is verified and to
execute Milestone B authorizes this step.

- New run ID: `mi355x-h4-calibration`.
- New demand-trace run ID: `mi355x-h4-decode`.
- New output:
  `artifacts/runs/mi355x-h4-calibration/analysis/h4/`.
- Historical RTX output remains:
  `artifacts/runs/h1-standard-small/analysis/h4/`.
- Timing remains hook-free and uses four existing domain prompts.
- Transfer sizes, warmups, measured repetitions, capacities, horizons,
  bandwidth sensitivities, and the old H4 gate remain unchanged for a direct
  hardware comparison.
- ROCm event timing and pinned host-to-device completion are measured
  platform-stack evidence combining the GPU, host, kernels, PyTorch,
  Transformers, and ROCm. They never overwrite or retroactively reinterpret
  the RTX calibration, and cross-platform forward differences are not treated
  as inherent GPU characteristics.
- The demand trace uses the same 128 prompts, ordering, 384-token prompt cap,
  greedy decoding, and 64-token generation limit as `h1-standard-small`.
  Hooks are used only to collect MI355X demand; all timing remains hook-free.
- `analyze-h4` must reject a `--run` path different from the config's frozen
  `trace_run`, preventing accidental counterfactual replay in the substantive
  AMD result.
- Write a machine-readable direct calibration table against the preserved RTX
  measurement alongside the unchanged H4 oracle grid.

The existing H4 calibrator is OLMoE-specific: it assumes 16 routed layers and
an exact 12 MiB BF16 expert. It is appropriate for this baseline milestone and
must not be reused unchanged for GPT-OSS.

## Evidence and publication hygiene

- Retain all new compact definitions, manifests, measurement samples, tables,
  reports, and figures under their `mi355x-*` run directories.
- Keep raw traces ignored and locally restorable until the parity and C1
  comparisons close.
- Record `HIP_VISIBLE_DEVICES`, PyTorch backend/HIP version, device name,
  architecture, model/tokenizer revisions, prompt hash, and Git commit.
- Never use a hooked run for a latency claim.
- Never merge prefill and decode in a headline metric.
- Apply the normal artifact audit only after the milestone's derived evidence
  and human review note exist.
