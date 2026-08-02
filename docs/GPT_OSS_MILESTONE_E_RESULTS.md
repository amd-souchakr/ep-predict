# GPT-OSS 20B Milestone E results

**Overall decision:** `CONDITIONAL_PILOT_SUPPORT_WITH_TRACE_WEIGHT_EXCEPTION`
**Prediction gate:** `PILOT_SUPPORTS_20B_ROUTE_PREDICTION` (3/3 primary
lookaheads pass)
**Frozen trace-integrity gate:** `FAILED` (6/2,323,200 independent selected
weights exceed `1e-6`; all executed IDs match)
**Run:** `artifacts/runs/gpt-oss-20b-milestone-e`
**Checkpoint:** `openai/gpt-oss-20b` at
`6cee5e81ee83917806bbde320786a8fb61efebee`

## Result in one paragraph

GPT-OSS 20B routing is strongly predictable across network depth on this
held-out workload. At decode K=8—twice the top-4 route width and a candidate
set spanning 25% of the 32-expert namespace—the transition table covers
86.3%, 85.5%, and 84.4% of selected
experts at Δ=1,2,3. It beats the stronger of domain-conditioned popularity and
current-route copy by +18.2, +16.7, and +15.5 percentage points. Exact
complete-top-4 coverage is 60.7%, 59.4%, and 56.8%, gains of +32.3, +30.0,
and +26.9 points. All request-level 95% bootstrap intervals exclude zero and
all four domains improve, so the prediction gate passes decisively. The
overall milestone is conditional, not cleanly qualified: the preregistered
bit-level trace gate failed on six rare BF16-scale weight deviations even
though every consumed expert ID matched and the analysis uses the exact
dispatch-consumed weights.

## Frozen design and collected evidence

- 128 revision-pinned requests: 32 each from code, conversation, general text,
  and math.
- Fixed request split: 96 train / 32 test, with 24/8 per domain. No token from
  a test request trains a popularity or transition table.
- Greedy batch-one generation with 16 traced decode tokens per request.
- 22,152 prompt tokens and 2,048 decode tokens; 580,800/580,800 token-layer
  records across all 24 routed layers.
- 2,323,200 exact dispatch-consumed `(expert_id, weight)` pairs retained under
  the standard trace schema.
- K=4/8/16, all Δ=1--23, and prefill/decode analyzed separately.
- Metrics are aggregated within request before domains are balanced. Primary
  uncertainty uses 2,000 stratified request-level bootstrap resamples.

## Preregistered decode gate at K=8

The comparator below is selected conservatively as the stronger of
domain-conditioned target-layer popularity and current-route copy separately
for each metric. Domain popularity is the stronger comparator at all three
primary points.

| Δ | Transition selection | Comparator | Gain (95% CI) | Transition routed mass | Mass gain | Transition complete route | Complete gain (95% CI) | Positive domains | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 86.3% | 68.1% | +18.2 pp [+17.4, +19.0] | 88.2% | +17.7 pp | 60.7% | +32.3 pp [+31.3, +33.1] | 4/4 | yes |
| 2 | 85.5% | 68.8% | +16.7 pp [+16.0, +17.6] | 87.5% | +16.5 pp | 59.4% | +30.0 pp [+28.9, +31.0] | 4/4 | yes |
| 3 | 84.4% | 68.8% | +15.5 pp [+14.8, +16.4] | 86.4% | +15.5 pp | 56.8% | +26.9 pp [+26.1, +27.6] | 4/4 | yes |

The smallest domain selection gain is still +16.2/+15.1/+13.9 points at
Δ=1/2/3. This is not an effect driven by one domain.

## Candidate count is the prediction tradeoff

| K candidates | K/top-4 | Candidate-set fraction K/32 | Δ=1 selection | Δ=1 routed mass | Δ=1 complete route | Δ=3 selection | Δ=3 complete route |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1× | 12.5% | 66.7% | 70.6% | 19.9% | 63.4% | 18.3% |
| 8 | 2× | 25.0% | 86.3% | 88.2% | 60.7% | 84.4% | 56.8% |
| 16 | 4× | 50.0% | 96.4% | 97.0% | 87.4% | 95.7% | 85.3% |

This is the central negative qualification. Selection coverage is not a
surrogate for complete-route coverage. At K=4, roughly two thirds of expert
selections are correct but four fifths of tokens still miss at least one
expert. K=16 looks excellent only by nominating half the expert population and
issuing four candidates per demanded expert. The prediction is scientifically
strong; its resource contract is not automatically attractive. The faceted
[candidate-count–horizon figure](../artifacts/runs/gpt-oss-20b-milestone-e/figures/fig3_coverage_by_horizon_and_candidate_count.png)
plots K=4/8/16 on identical axes so this gap is visible across every lookahead,
not only at the tabulated primary points.

K is not residency. K controls how many experts the predictor nominates;
residency R controls how many experts a fast tier already retains. They are
independent: an R=4 system can issue K=16 candidates, and an R=16 system can
issue K=4. K/E becomes a memory fraction only under an additional staging
policy that loads every candidate simultaneously, assumes equal expert sizes,
and ignores already-resident overlap. No such policy is modeled here. A future
resource replay must sweep K and R independently and report cold predicted
candidates, transfers, staging occupancy, and eviction pressure.

Across the all-valid-source horizon scan, the transition predictor spans
58.8--66.7% selection and 15.3--19.9% complete-route coverage at K=4;
78.6--86.3% and 49.2--60.7% at K=8; and 92.9--96.4% and 79.9--87.4% at K=16.
Those bands summarize different eligible source-layer mixtures at different
Δ, so the faceted plot is a candidate-count comparison, not a fixed-cohort decay
curve.

## All-horizon and prefill evidence

At decode K=8, transition selection coverage remains 78.6--86.3% over the
reported all-valid-source Δ=1--23 scan, while complete-route coverage remains
49.2--60.7%. The transition advantage stays positive in every valid
source/horizon cell in the heatmap. This apparent stability must not be read as
a fixed-cohort horizon curve: the eligible source-layer set shrinks with Δ,
and Δ=23 is only layer 0→23. The source/horizon heatmap is the honest view of
that composition.

Prefill is also predictable but weaker at K=8: transition selection coverage
is 80.5%, 79.6%, and 78.7% at Δ=1/2/3, with gains of +17.0, +15.1, and +13.7
points over the stronger cheap baseline. Complete-route coverage is
42.9--41.0% across those three points.

## Trace-weight exception

The frozen collection gate required zero independent-router/dispatch weight
mismatches at absolute tolerance `1e-6`. It failed:

- executed expert-ID mismatches: 0 / 2,323,200;
- selected-weight mismatches: 6 / 2,323,200 (0.000258%);
- maximum absolute error: 0.001953125;
- affected scopes: six different request/layer pairs; each has one mismatch;
- observed errors: 0.0009765625 or 0.001953125, BF16-sized power-of-two
  increments.

The native route uses the pinned Triton kernel's fused top-k and FP32 softmax
followed by a cast to the router dtype. The independent observer recomputes
top-k and softmax with PyTorch. The exact IDs plus sparse BF16-sized deviations
are consistent with backend numerical differences, not route ambiguity. That
is a diagnosis, not permission to rewrite a preregistered threshold after
seeing data. Therefore:

1. the frozen trace-integrity gate remains failed;
2. set-coverage results are unaffected because every executed ID matches;
3. routed-mass results use the dispatch-consumed weights themselves, not the
   independent reconstruction; and
4. the prediction result is reported as conditional post-hoc evidence.

A future protocol should use exact ID parity plus an explicitly BF16-aware
weight tolerance or ULP criterion. It should not pretend `1e-6` is meaningful
for two independently implemented BF16 softmax paths.

## Interpretation and stop decision

The 20B model has strong structured route trajectories, including long-range
structure. This generalizes the qualitative OLMoE H2 result to a different
top-4/32-expert architecture, but it is not a controlled cross-model effect
size because tokenization, prompts-as-rendered, route width, depth, and expert
count differ.

Do not infer prefetch profitability. No GPT-OSS expert-copy timing, overlap,
residency replay, or end-to-end latency was measured. Do not infer language
quality: this experiment predicts internal expert demand and does not score
answers. The 120B comparison is cancelled under the disk constraint, not
completed negatively.

The researcher selected the following publication sequence after review:
Milestone F fits a compact learned total-demand predictor on the existing 20B
traces, and Milestone G places the resulting empirical
coverage/amplification frontier in an analytical workload/memory-system
sweep. A cache manager remains conceptual, and predictability-aware model
training is future work. See
[GPT_OSS_MILESTONE_F_PROTOCOL.md](GPT_OSS_MILESTONE_F_PROTOCOL.md) and
[GPT_OSS_MILESTONE_G_PLAN.md](GPT_OSS_MILESTONE_G_PLAN.md).
