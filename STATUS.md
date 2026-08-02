# Project status

**Current focus:** Milestone E review — GPT-OSS 20B held-out route prediction
**Current stage:** The Milestone E prediction gate passes 3/3 short horizons,
but the overall result is `CONDITIONAL_PILOT_SUPPORT_WITH_TRACE_WEIGHT_EXCEPTION`.
All 2,323,200 executed expert IDs match and token/layer coverage is complete;
six independently reconstructed weights differ by at most 0.001953125, so the
frozen `1e-6` trace gate formally fails. The 120B comparison is cancelled
under the disk constraint. Milestone E's K is candidate count, not resident
capacity; no residency state was collected or replayed.
**Last updated:** 2026-08-01

| Gate | Question | State | Exit evidence |
|---|---|---|---|
| M0 | Can we inspect and validate the model without source changes? | Passed | Model report, complete hook traces, and zero integrity failures |
| AMD-A | Does pinned OLMoE run with valid hooks on one MI355X, and are matched derived NVIDIA trends retained? | Reviewed; aggregate reproduction accepted with raw parity unresolved | 128/128 requests, 222,688 records, zero router mismatches; H1 top-8 correlation 0.999962 and H2 horizon-gain correlation 0.999989; no raw-record parity claim |
| AMD-B | Does MI355X calibration make the unchanged H4 whole-expert oracle gate feasible on measured AMD demand? | Pilot supports; reviewed with narrowed testbed interpretation | K=16 passes at delta 1/2/3; delta 3 reaches 83.9% timely cold bytes and 86.5% stall reduction |
| AMD-C | Can Transformers expose GPT-OSS routing with IDs and weights proven identical to actual dispatch? | Qualified; reviewed and advanced | 24/24 dispatch hooks, 576/576 ID-weight pairs match, zero ordinary router hooks confirms the covered MXFP4 bypass; provenance and storage semantics recorded |
| AMD-D | Does the qualified GPT-OSS 20B path produce a complete deterministic tracer-bullet artifact chain? | Qualified; reviewed and advanced | 4,248/4,248 token-layer records, 16,992/16,992 ID-weight pairs match, exact same-process repeat, retained outputs/tables/figures/manifests |
| AMD-E | On held-out GPT-OSS 20B requests, do transition tables beat strong cheap route baselines? | Conditional pilot support; review pending | At decode K=8 candidates, Δ=1/2/3 selection gains are +18.2/+16.7/+15.5 pp and complete-route gains are +32.3/+30.0/+26.9 pp; 3/3 gates pass, but 6/2,323,200 independent weights fail the frozen trace tolerance; residency is unmodeled |
| H1 | Is hotness strong and stable enough for a fast tier? | Mixed; global gate failed | 2/16 mixed decode layers passed; code and math are locally strong |
| H2 | Does conditional locality beat marginal popularity? | Pilot supported | All Δ=1/2/3 transition baselines passed the held-out decode gate |
| H3 | Can a small predictor beat transition tables at equal candidate budget? | Formal pilot not supported; reviewed | Global replacement failed; post-hoc scan found strong early-layer/long-range value |
| H4 | Is oracle prefetch physically viable? | Formal K=16 short-horizon gate not supported; broader region mixed | At K=16, Δ=3: 32.8% timely cold bytes and 38.9% stall reduction; K=32, Δ=3 reaches 55.5%/61.8% |
| H5-A | What prediction × hardware assumptions create a profitable analytical window? | Supported analytically | 22,618/68,175 controlled cells pass the frozen screen |
| H5-B | What predictor quality is required to enter that window? | Complete | 25–50% minimum complete cold-set coverage in nonempty windows |
| H5-C | Where do existing transition/linear policies land? | Raw streams not supported | 3.4–6.7× transfer amplification; no representative row passes |
| H5-D | Do existing scores separate useful from useless cold candidates? | Mixed; strong ranking, insufficient threshold | Linear AUROC 0.883/0.861 at Δ=3/9; C≥50% needs A≈3.0–3.3× |
| H6 | Does prediction-guided residency beat static/domain/LRU placement? | Pilot not supported; reviewed | At decode K=16, Δ=3, transition/linear lose 3.9/2.5 pp expert-stall reduction and 0.7/0.6 pp complete hits versus the strongest matched baseline |
| H7 | Can routing be made more predictable without harming loss or balance? | Deferred after H6 failure | Requires a new mechanism and explicit permission |
| C0 | Does post-training materially change matched-token trajectory predictability? | Pilot not supported; review pending | Base/Instruct retain 89.7% of selections; layer-0→15 conditional-gain change is +1.6 pp versus a 5 pp gate |
| C1 | Does the result transfer to a top-1/top-2 checkpoint? | Deferred; explicit permission required | No model download or testbed change authorized |
| AX1 | Under assumed future MTP-style routing quality, what capacity/TPOT envelope does predictive offload enable? | Projected region exists; review pending | At measured PCIe and assumed C=99%, A=1.5×, wave-local P99 improves 34–39% versus reactive offload; FCFS queue tails are materially worse |
| AX2 | What bandwidth, latency, reliability, amplification, and granularity bounds define viable regions? | Complete; review pending | K=16, A=1× needs 71.3/22.8/11.6/8.2 GB/s at Δ=1/3/6/9; reliability remains orthogonal |
| AX3 | What HBM and rolling-SRAM organization suits a three-tier predictive hierarchy? | Physical staging envelope complete; review pending | Top-8 whole-expert double buffering needs 192 MiB at A=1× and 384 MiB at A=2×; no SRAM execution speedup is claimed |
| AX4 | Can deadline-controlled expert erasure bound low-batch TPOT while retaining a plausible routed-mass/quality contract? | Supported analytically under explicit assumptions; review pending | K=8, 256 GB/s, C=99%, A=1.5× passes with 1/8 experts resident, 11.25 ms bounded TPOT, zero full fallback, and <1% degraded waves; measured PCIe fails |

## Immediate run checklist

- [x] Amend the parity scope after confirming NVIDIA request traces cannot be
      restored; derived trends cannot establish trace interchangeability.
- [x] Expose one MI355X and pass ROCm, BF16, pinned-H2D, and tiny-router-hook
      qualification.
- [x] Inspect the pinned full OLMoE checkpoint and reproduce its 16-layer,
      64-expert, top-8, 12 MiB geometry.
- [x] Collect the frozen 16-request raw-prefill prefix with input hashes and
      zero router mismatches.
- [x] Run descriptive H1 and held-out H2 analyses through Δ=15.
- [x] Compare MI355X H1 layer trends and H2 horizon trends with the preserved
      128-request NVIDIA artifacts.
- [x] Generate and inspect the derived-trend PDF/PNG with hashed inputs.
- [x] After review identified a visible H1 offset, preserve the 16-request
      tracer bullet and rerun all 128 requests under the exact NVIDIA C0
      collection and H2 settings.
- [x] Verify matching request keys/order, prompt hash, collection settings,
      96/32 H2 split, and 126/126 available historical input-token hashes.
- [x] Regenerate H1/H2 and the platform figure with residual panels; confirm
      the H1 offset collapses to 0.0134 pp mean absolute difference.
- [x] Researcher verified Milestone A and authorized isolated H4 calibration.
- [x] Collect a separate 128-request, 64-token MI355X standard decode trace;
      require zero router mismatches and keep timing hook-free.
- [x] Measure 80 cached-token forwards and 250 pinned H2D copies on exactly one
      visible `gfx950` device.
- [x] Replay the unchanged H4 grid on the measured MI355X demand trace and
      compare calibration directly with the preserved RTX result.
- [x] Apply the frozen gate unchanged: all K=16 delta 1--3 points pass.
- [x] Generate and programmatically inspect the two H4 figures with hashed
      inputs.
- [x] After researcher feedback, replace the dense H4 views with a
      plain-language 63-cell grid and a two-metric measured-link chart; retain
      every capacity/lookahead trend and mark the frozen decision region.
- [x] Researcher reviewed Milestone B, accepted the regime-space result, and
      narrowed the forward-time interpretation to a likely software/testbed
      artifact rather than an inherent MI355X property.
- [x] Complete the requested plain-language figure revision without dropping
      any capacity, bandwidth, lookahead, or gate-metric trends.
- [x] Freeze exact Transformers and GPT-OSS checkpoint revisions for
      Milestone C before executing weights.
- [x] Trace the model-specific router-to-dispatch code path and define how
      hook observations will be proven identical to executed expert IDs.
- [x] Qualify whether MXFP4/custom kernels bypass module hooks; record stored
      and loaded bytes, compute dtype, shared experts, top-k order, and weight
      normalization semantics.
- [x] Exercise only the cheapest configuration/tiny or GPT-OSS 20B path needed
      to prove the implementation, then stop for review before Milestone D.
- [x] Researcher advanced the sequence by explicitly requesting Milestone D.
- [x] Freeze the two-request deterministic GPT-OSS 20B tracer-bullet protocol.
- [x] Retain tokenized inputs, eight outputs per request, and a terminal decode
      forward so every retained output token has routing coverage.
- [x] Capture the qualified MXFP4 dispatch boundary and require complete
      24-layer/token coverage with zero independent-router parity errors.
- [x] Repeat the workload and require identical input/output IDs, route IDs,
      and selected weights within `1e-6`.
- [x] Generate compact routing tables, two scripted figures, and a hashed
      artifact manifest.
- [x] Researcher verified Milestone D and replaced the 120B comparison with a
      20B-only prediction-quality experiment because disk capacity is
      insufficient.
- [x] Freeze the 128-request, four-domain GPT-OSS 20B Milestone E protocol with
      96/32 request-held-out analysis and K=4/8/16 through Δ=23.
- [x] Collect 22,152 prompt and 2,048 decode tokens with complete 24-layer
      coverage and exact dispatch-consumed IDs/weights.
- [x] Preserve the frozen trace-gate failure: 6/2,323,200 independent weights
      differ by at most 0.001953125 even though every executed ID matches.
- [x] Apply the prediction gate conditionally using exact dispatch records:
      all Δ=1/2/3 K=8 request-bootstrap points pass against the stronger cheap
      baseline.
- [x] Generate and programmatically inspect the horizon and source-layer
      figures and hash the durable artifact chain.
- [x] Correct K/E terminology: it is candidate-set fraction in Milestone E,
      not resident fraction; retain residency R/E as an independent future
      replay variable.
- [ ] Researcher reviews Milestone E and decides whether to replay a bounded
      20B resource contract or close the GPT-OSS track.

- [x] Confirm the actual machine exposes the intended 24 GB NVIDIA GPU.
- [x] Install the `data` and `inference` dependency groups.
- [x] Materialize and review the revision-pinned standard-small workload.
- [x] Run model inspection and retain `model_report.json`.
- [x] Run two standard examples with `--limit 2`; require zero integrity errors.
- [x] Run the 128-request standard pilot.
- [x] Review H1 gate output before expanding the workload.
- [x] Generate the three scripted H1 decision figures.
- [x] Human reviewed H1 and approved advancing with the mixed interpretation.
- [ ] If borderline, run the confirmation workload with more requests and
      request-level bootstrap intervals.
- [x] Decide H1 and record the outcome in `EXPERIMENT_LOG.md`.
- [x] Preregister H2 with a request-level held-out split and fixed gate.
- [x] Reuse the H1 trace to evaluate static, domain, lagged, and transition
      baselines at K=8/16/32 and Δ=1/2/3.
- [x] Generate H2 diagnostics, then replace the primary view with two simple
      lookahead plots.
- [x] Human reviewed the simplified H2 figures and approved advancing to H3.
- [x] Preregister the minimal H3 proof/disproof gate before collecting hidden
      features or training a predictor.
- [x] Validate deterministic projected router-input capture on two requests.
- [x] Collect all 128 H3 requests with aligned routing and feature shards.
- [x] Train the single fixed linear recipe on the preserved 96/32 split.
- [x] Apply the primary H3 gate and reproduce all H2 transition metrics.
- [x] Generate two simple H3 decision figures and hash their inputs.
- [x] Extend H2/H3 descriptively through Δ=15 without new inference.
- [x] Generate the global horizon curve and source-target gain heatmap.
- [x] Human reviewed H3 and the extended figures; formal H3 failure stands,
      with early-layer linear prediction retained as an exploratory regime.
- [x] Freeze one simple H4 question and oracle stop/go rule.
- [x] Measure unhooked inter-layer time and a small host-to-device transfer
      curve; hooked timings remain excluded.
- [x] Replay exact 12 MiB experts over a small bandwidth/capacity/issue-point
      grid and plot the oracle feasible region.
- [x] Apply the formal gate before interpreting the broader scan; the
      preregistered K=16, Δ=1–3 target failed.
- [x] Do not overlay or tune transition/linear policies after the formal H4
      failure.
- [x] Researcher accepted first-order analytical modeling as sufficient for
      the next viability/profitability-window study; timing fidelity is not the
      current focus.
- [x] Add the post-hoc cold-service-headroom versus complete-prediction regime
      map; keep “candidate region” distinct from demonstrated profitability.
- [x] Preregister H5-A’s simple analytical-profit screen: ≥25% modeled stall
      reduction, ≥50% oracle recovery, and ≤2× predicted/useful bytes.
- [x] Sweep complete coverage, candidate amplification, K=8/16/32, Δ=1–15,
      and 0.25×–4× normalized cold bandwidth using existing artifacts.
- [x] Derive H5-B minimum complete coverage and maximum amplification rather
      than tuning a predictor blindly.
- [x] Generate one profitability phase diagram and one inverse-requirement
      curve.
- [x] Researcher explicitly advanced from H5 to the minimal H6 residency
      study; the H5 formal result remains unchanged.
- [x] Reconstruct and place existing transition/linear streams at four
      representative H5-C points without retraining.
- [x] Sweep a shared standardized score threshold and plot useful-versus-false
      expert separation at K=32, Δ=3/9.
- [x] Defer the cost-sensitive admission head; test placement value first.
- [x] Preregister H6 with held-out decode K=16, Δ=3, one demanded-miss
      insertion per wave, and a breadth gate across layers and domains.
- [x] Replay static, domain, LRU, transition, linear, and equal-budget oracle
      residency at K=8/16/32 over both phases and all valid lookaheads.
- [x] Report residual cold demand, complete-set hits, useful/wasted movement
      bytes, evictions/churn, first-order stall reduction, and oracle recovery.
- [x] Generate the compact layer/lookahead/capacity gain heatmap and hash its
      inputs.
- [x] Apply the H6 gate unchanged: neither guided policy passes.
- [x] Complete human review of the H6 heatmap and record the final
      interpretation.
- [x] Defer overlap microbenchmarks, MLPs, fresh routing collection, H7, C1,
      and timing
      fidelity until a selective policy identifies a worthwhile mechanism.
- [x] Preregister C0 before Base collection with exact matched-token
      serialization and a fixed layer-0→15 conditional-gain gate.
- [x] Download and qualify OLMoE Base only after explicit researcher approval;
      verify identical 16-layer, 64-expert, top-8 geometry.
- [x] Collect Base and Instruct raw-prefill traces on the same 128 prompts with
      one forward per request and zero router mismatches.
- [x] Verify all 13,918 input tokens match exactly across checkpoints.
- [x] Fit H2 tables independently on the same 96 requests and evaluate the same
      32 held-out requests through Δ=15.
- [x] Apply the C0 gate unchanged: +1.6 pp is below the 5 pp stage-effect
      threshold.
- [x] Generate and programmatically inspect the predictability and matched-route
      figures with hashed inputs.
- [ ] Complete researcher visual review; do not add SFT/DPO unless that review
      overturns the frozen endpoint stop decision.
- [x] Freeze the AX evidence contract separating measured, trace-derived,
      assumed-predictor, and hypothetical-hardware inputs.
- [x] Freeze AX1 predictor-quality, capacity, lookahead, bandwidth, latency,
      concurrency, granularity, and SLO sweep axes.
- [x] Define capacity viability, reactive-hierarchy profitability, and SLO
      safety without claiming speedup over all-HBM execution.
- [x] Define AX2 inverse requirements and AX3 rolling three-tier SRAM
      semantics.
- [x] Implement AX1 by extending the existing H4/H5 replay.
- [x] Reproduce the measured H4/H5 anchors before interpreting synthetic
      future-router points.
- [x] Generate the capacity–P99 Pareto frontier and execute the factorized AX2
      inverse-bound and AX3 rolling-SRAM sweeps.
- [x] Generate the three principal PDF/PNG figures and inspect them
      programmatically.
- [ ] Researcher reviews the AX figures and selects one representative
      architecture point, or closes the track without live calibration.
- [x] Freeze AX4's hard commit deadline, normalized routed-mass definitions,
      null/renormalized/shared-residual policies, and perturbation bounds.
- [x] Freeze the analysis-only weighted-route replay and mass-priority oracle
      without authorizing model training or inference collection.
- [x] Define low-batch bounded TPOT/tokens-s projections and clearly label
      top-1/top-2/large-model geometry as sensitivity rather than evidence.
- [x] Freeze the deadline-elastic HW proposal: always-resident fallback,
      optional refinements, commit bitmap, deadline-aware DMA, traffic
      isolation, and degradation telemetry.
- [x] Implement AX4 and first verify the selected-weight execution semantics.
- [x] Replay the retained trace, generate the three principal figures, and
      apply the plausible-degradation-contract gate before any training work.
- [ ] Researcher reviews the AX4 figures and accepts, narrows, or rejects the
      erasure-robustness target before any training or new-model work.

Full plan: [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md).
Prior result: [docs/H4_RESULTS.md](docs/H4_RESULTS.md).
Architecture protocol:
[docs/ARCHITECTURE_EXPLORATION_PROTOCOL.md](docs/ARCHITECTURE_EXPLORATION_PROTOCOL.md).
Latest empirical result:
[docs/MI355X_H4_RESULTS.md](docs/MI355X_H4_RESULTS.md).

## Evidence policy

For empirical hypotheses, `Ready` means code and protocol exist. AX results
are explicitly projected: they combine measured calibration, trace-derived
demand, assumed predictor quality, and hypothetical hardware. The canonical
tables, report, and figures are under
`artifacts/runs/h1-standard-small/analysis/architecture/`; none is a measured
end-to-end speedup. AX4's canonical result is under
`artifacts/runs/h1-standard-small/analysis/ax4_deadline_degradation/`. Its gate
pass identifies a future training contract; it does not mean current OLMoE
tolerates missing experts.
