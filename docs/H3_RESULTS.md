# H3 pilot findings

## Outcome

The preregistered H3 gate is **not supported** for the pinned OLMoE
checkpoint. At the primary decode setting \(K=16,\Delta=1\), the fixed linear
sidecar did not materially and consistently beat the H2 transition table:

| Primary gate condition | Required | Observed | Pass |
|---|---:|---:|:---:|
| Mean selection-coverage gain | ≥ +3.0 pp | +0.4 pp | no |
| Mean complete-token gain | ≥ +2.0 pp | +4.7 pp | yes |
| Layer-domain scopes with positive selection gain | ≥ 75% | 56.7% (34/60) | no |
| Layer-domain scopes with positive complete-token gain | ≥ 75% | 75.0% (45/60) | yes |
| Domains positive on both metrics | ≥ 3/4 | 2/4 | no |

The correct prototype decision is to stop learned-predictor work for this
checkpoint and carry the simpler transition policy into H4. The result does
not say that router-input hidden states contain no additional information:
they improve strict complete-route coverage in some regimes. It says that this
fixed linear sidecar does not provide a broad enough improvement to justify
itself under the frozen gate.

## Integrity and held-out design

- Reran the same pinned 128-request workload with the same greedy generation.
- Captured the tensor directly entering every explicit router with ordinary
  forward hooks; no model or Transformers source was modified.
- Stored a fixed 128-dimensional float16 Rademacher projection, seed 31.
- Projection SHA-256:
  `bbf5b1a3a45256c86d0b59acc4ba6dddf854a7eb3db297b9a02659adbf0a9bb1`.
- Collected 377,488 projected layer-events in 128 atomic feature shards.
- All 128 routing shards completed with zero top-k validation mismatches.
- Feature metadata and ordering matched every routing record exactly; no
  non-finite features were present.
- Reproduced the exact H2 96/32 request split and all 1,008 H2 transition
  coverage scopes with maximum absolute difference 0.0.
- Fit all normalizers and predictor weights on the 96 training requests only.
- Evaluated the 32 held-out requests with prefill, decode, and domains
  separated.

The predictor artifact contains 84 numeric linear heads: 42 eligible
source-layer/lookahead pairs for each of two phases. It uses non-pickle NPZ
storage and records its SHA-256 and fixed training configuration.

## Primary result in plain language

At decode \(K=16,\Delta=1\):

| Policy | Selection coverage | Complete top-8 coverage | Candidate churn |
|---|---:|---:|---:|
| Transition table | 79.0% | 24.1% | 42.8% |
| Linear sidecar | 79.4% | 28.7% | 52.8% |

The linear model finds almost exactly the same fraction of future experts as
the transition table, while changing its candidate set more often. It does
form a more complementary set in some tokens, raising the chance of covering
all eight experts by 4.7 points. That strict-coverage improvement is useful
scientific evidence, but not enough to pass because it is concentrated in
code and mathematics rather than broad across workloads.

## Domain consistency

Primary decode \(K=16,\Delta=1\) linear gains over transition:

| Domain | Selection gain | Complete-token gain | Linear selection | Linear complete token |
|---|---:|---:|---:|---:|
| Code | +3.0 pp | +9.7 pp | 86.1% | 43.0% |
| Conversation | −3.3 pp | −3.3 pp | 72.5% | 14.8% |
| General prose | −1.0 pp | +0.0 pp | 73.5% | 15.4% |
| Mathematics | +3.0 pp | +12.2 pp | 85.5% | 41.7% |

Across the 60 eligible layer-domain scopes, selection gains range from
−14.3 to +15.6 points with a median of +0.8. Complete-token gains range from
−21.6 to +24.4 points with a median of +4.7. This dispersion is the central
reason not to summarize the experiment using only its favorable aggregate
complete-token gain.

## Lookahead and capacity

Domain-balanced decode results at \(K=16\):

| Lookahead | Transition selection | Linear selection | Transition complete | Linear complete |
|---:|---:|---:|---:|---:|
| \(n+1\) | 79.0% | 79.4% | 24.1% | 28.7% |
| \(n+2\) | 77.9% | 79.3% | 23.5% | 28.8% |
| \(n+3\) | 76.8% | 79.3% | 22.2% | 28.8% |

The linear readout is more stable with lookahead than the transition table,
but \(\Delta=2,3\) are secondary and cannot rescue the failed primary gate.
At \(K=32\), the policies are nearly tied: for \(n+1\), transition reaches
93.2% selection and 64.1% complete coverage versus 92.9% and 63.2% for the
linear model. This reinforces that hidden-state prediction is not uniformly
dominant as capacity grows.

Candidate amplification is fixed at 1x, 2x, and 4x for \(K=8,16,32\). At the
primary \(K=16\) setting, the linear sidecar replaces 52.8% of candidates per
decode token versus 42.8% for transition. Any later hardware replay would need
to charge this higher movement pressure.

Prefill is directionally more favorable—at \(K=16,\Delta=1\), linear reaches
80.6% selection and 36.5% complete coverage versus 79.0% and 26.7% for
transition—but prefill is secondary and has different parallel execution
semantics.

## Interpretation

Three conclusions are justified:

1. A 128-dimensional router-input projection retains useful future-route
   information. The complete-set gains, especially for code and math, are too
   large to call the hidden state uninformative.
2. The basic learned policy is not a broadly superior replacement for current
   route transitions on this checkpoint. It misses the selection-gain and
   cross-domain consistency requirements and incurs about 10 points more
   candidate churn at the primary setting.
3. The simplest next architectural test should therefore use transition
   tables. H4 can ask whether even perfect or transition-guided lookahead has a
   physically viable bandwidth/capacity/timing region without confounding the
   answer with predictor tuning.

Do not start an MLP, combine route and hidden features, tune projection size,
or specialize heads by domain now. Those are explicitly outside this gate and
would weaken the prototype kill-switch discipline. They can be reconsidered
only if a later cross-model result or H4 feasibility result creates a concrete
reason.

## Limitations and claim boundary

- One checkpoint, one deterministic held-out split, and one balanced
  128-request workload are a pilot, not a generalization claim.
- OLMoE routes top-8 of 64 experts; both the transition baseline and strict
  complete-set behavior may differ in top-1/top-2 models.
- The experiment did not define batch waves or hardware replay, so it does not
  claim wave-complete or decode-step-complete coverage.
- Hooked collection and sidecar accuracy provide no latency measurement.
- A frozen external readout does not show that the base model learned to manage
  hardware resources.

## Figures and human checkpoint

Figures are under
`artifacts/runs/h3-standard-small/analysis/h3/figures/`:

- `fig1_h3_lookahead_comparison`: transition versus linear coverage through
  \(n+3\);
- `fig2_h3_domain_consistency`: primary-gate gains by domain.

The figure inputs and outputs are hashed in `figure_manifest.json`.
Human review is pending. After review, the recommended single next action is:
run the minimum H4 hardware-feasibility study using the transition policy,
without predictor optimization.

## Post-hoc extended-horizon qualification

The subsequent all-layer analysis in
`docs/H23_EXTENDED_HORIZON_RESULTS.md` does not alter the formal H3 gate, but
it narrows the failure interpretation. Linear hidden-state prediction strongly
beats transitions for early source layers and long horizons, including
layer 0→15. It remains unjustified as a global replacement policy. H4 should
therefore remain oracle-first while retaining both existing candidate streams
for source-target-aware comparison; no predictor tuning is warranted.
