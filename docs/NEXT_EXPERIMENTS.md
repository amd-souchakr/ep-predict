# Next experiments: first-order co-design and predictable routing

**Updated:** 2026-08-01  
**Current next action:** human review of the completed C0 Base–Instruct
figures; do not add SFT/DPO after the failed endpoint gate
**Operating rule:** use the cheapest existing artifacts first; model broad
viability and expected benefit before improving timing fidelity or predictors.

This plan separates three questions that should not be conflated:

1. Under what hardware and prediction assumptions is hierarchical expert
   management analytically worthwhile?
2. Where do the existing transition and linear candidate streams land within
   that design space?
3. Can routing later be trained to move the prediction–quality Pareto frontier
   without harming language-model quality or load balance?

## Experiment list

| ID | Question | New inference/training? | Status |
|---|---|---:|---|
| H5-A | What prediction × hardware combinations create a first-order profitability window? | No | Complete; region exists |
| H5-B | What minimum predictor quality is required at each capacity, bandwidth, and lookahead? | No; derived from H5-A | Complete |
| H5-C | How much analytical oracle benefit do the existing transition and linear streams recover? | No retraining; reconstruct existing predictions | Complete; raw streams fail traffic gate |
| H5-D | Do existing scores separate useful from useless cold candidates? | No | Complete; signal present, shared threshold insufficient |
| H6 | Does prediction-guided on-demand residency beat static/domain/LRU at equal capacity and movement budget? | No | Complete; frozen gate failed |
| H7 | Can a routing-predictability objective improve modeled benefit without harming loss or load balance? | Yes; small controlled intervention | Deferred after H6 failure |
| C0 | Does Base→Instruct post-training materially change matched-token trajectory predictability? | Yes; two endpoint traces | Complete; frozen stage-effect gate failed |
| C1 | Does the trajectory/co-design result transfer to one newer top-1/top-2 checkpoint? | Yes; one trace collection | Deferred; explicit permission required |

Detailed timing validation, concurrent-copy microbenchmarks, multi-GPU
extrapolation, MLPs, predictor tuning, new inference, and model downloads
remain deferred. H6 did not establish placement value for the current depth
predictors.

C0 adds a within-family control, not a new placement mechanism. Base and
Instruct preserve 89.7% of expert selections and differ by only +1.6 points on
the frozen long-range conditional-predictability metric. Do not spend two more
checkpoint downloads on SFT/DPO stage localization. The next generalization
experiment, if explicitly approved later, should change routing architecture
rather than add another OLMoE post-training stage.

## H5-A — Controlled prediction × hardware sweep

### Question

For the observed OLMoE cold-demand and reuse structure, which combinations of
prediction quality, speculative traffic, fast-tier capacity, lookahead, and
cold-path bandwidth produce useful expected benefit?

### First-order model

Use the dimensionless cold-service pressure:

\[
\rho =
\frac{
\bar N_{\text{candidate}} S_{\text{expert}}
}{
B_{\text{cold}}\Delta T_{\text{layer}}
}.
\]

Reuse existing trace-derived demand, LRU hit/cold rates, exact 12 MiB expert
size, and the effective layer interval. Do not add a more detailed performance
model for this experiment.

Sweep:

- complete-route coverage from 0% to 100%;
- candidate amplification \(A\in\{1,2,4\}\);
- per-layer capacity \(K\in\{8,16,32\}\);
- lookahead \(\Delta=1\ldots15\);
- normalized cold bandwidth \(0.25\times,0.5\times,1\times,2\times,4\times\).

Treat expert size and bandwidth as the same first-order \(S/B\) control. Add a
separate expert-granularity axis only if a result cannot be explained by that
normalized ratio.

### Proposed screening gate

Call a point analytically profitable only if it achieves all three:

- at least 25% modeled reactive-stall reduction;
- at least 50% oracle recovery;
- no more than 2× predicted/useful transfer bytes.

These are prototype screening thresholds, not end-to-end speedup claims.
Freeze them in the H5 protocol before running the sweep.

### Outputs

- `h5_design_points.csv`: every assumption cell and category;
- `h5_windows.csv`: qualifying horizon range/count by capacity, bandwidth, and
  predictor assumption;
- expected stall reduction, oracle recovery, useful/false bytes, and
  amplification;
- one categorical phase diagram over physical headroom and complete coverage,
  with amplification as compact panels.

## H5-B — Inverse predictor requirements

### Question

What minimum complete-route coverage and maximum candidate amplification are
required to cross the H5-A screening gate?

Compute:

\[
C_{\min}(K,\Delta,B,A)
\]

and, where useful:

\[
A_{\max}(K,\Delta,B,C).
\]

This converts the hardware sweep into an ML and training target. The headline
output is one curve of minimum required complete coverage versus lookahead for
K=8/16/32, with bandwidth shown as a small number of line styles or panels.

Report empty windows explicitly. Do not hide a requirement above 100% or a
candidate amplification below the demanded top-k.

## H5-C — Existing-policy placement and replay

### Question

At representative points on the H5 surface, how much oracle benefit do the
existing untuned transition and linear candidate streams recover after false
candidate bytes are charged?

Reconstruct candidates from existing H2/H3 artifacts. Do not retrain, tune, or
add an MLP.

Primary representative cells:

- K=32, Δ=1: prediction-good/physics-limited control;
- K=32, Δ=3: current boundary candidate;
- K=32, Δ=9: long-range linear advantage;
- K=16, Δ=9: oracle-feasible/prediction-limited control.

Report cold-set rather than only total-route coverage:

- useful, false, and late candidate bytes;
- cold-expert and complete-cold-set coverage;
- expected stall reduction;
- oracle recovery;
- candidate amplification and churn.

The visualization should place actual policies on the H5-A phase diagram and
show a small actual-versus-required table. Do not create a separate dashboard.

## H6 — Mechanism competition

H6 is complete. It compared static popularity, domain popularity, reactive
LRU, transition-guided residency, linear-guided residency, and an equal-budget
next-use oracle at K=8/16/32. Prediction could admit only an actually demanded
miss; no broad candidate prefetch was allowed.

At the frozen decode K=16, Δ=3 gate, transition and linear lose 3.9 and 2.5 pp
of expert-stall reduction and 0.7 and 0.6 pp of complete-set hits relative to
the strongest matched simple baseline. The oracle remains strong, but existing
depth-trajectory scores do not predict temporal reuse well enough to select
residency.

Decision: stop this placement mechanism after human figure review. Do not fit
the previously proposed cost-sensitive head, tune an MLP, collect fresh
confirmation, begin H7, or download a second model. Any later work must first
pose a genuinely different mechanism or a direct temporal-reuse hypothesis and
receive explicit permission.

## H7 — Controlled routing-predictability intervention

H1–H6 observe a normally trained model; they do not show that predictability
can be increased without quality loss. H6 also removes the immediate placement
justification for this intervention.

If H5-B produces a plausible predictor target, run one small matched pilot:

1. standard load-balancing continuation objective;
2. the same objective plus one trajectory-predictability term.

Start with one seed and a short token budget. Measure the Pareto tuple:

\[
(\text{validation loss},\ \text{load balance},\
\text{complete trajectory coverage},\ \text{modeled HW benefit}).
\]

Proceed only if the intervention moves the modeled-benefit frontier without a
material validation-loss or load-balance regression. Replication, regularizer
ablations, and broader training wait for that result.

This intervention is a distinct project phase because it modifies training.
It must not be retroactively inferred from the current hook-only evidence.

## C1 — Sparse-model transfer check

Before generalizing beyond OLMoE, repeat only the decisive trace and analytical
steps on one checkpoint with top-1/top-2 routing and more sparsity:

- routing integrity and exact expert bytes;
- H1/H2 trajectory structure;
- the H5-A normalized co-design map;
- existing simple transition baseline;
- a linear sidecar only if transitions leave a meaningful information gap.

The purpose is to test whether OLMoE top-8 demand makes complete-set coverage
and cold-service pressure unusually harsh.

## Insight mining after each major experiment

After applying the frozen gate, use cheap post-hoc analysis to extract:

- boundary locations and empty opportunity windows;
- capacity–bandwidth–lookahead substitution rates;
- whether mean headroom disagrees with trace-driven tail behavior;
- the source layers and domains that create or destroy a viable region;
- whether complete coverage, speculative traffic, or physics is limiting;
- dimensionless quantities that transfer across hardware assumptions;
- negative results that redirect prediction toward residency, replication, or
  activation movement.

Label each conclusion as directly supported, analytical inference, or
speculation. Update `docs/FOUNDATIONAL_INSIGHTS.md` only when a result changes
the durable thesis.

## Visualization standard for the next phase

Use at most two primary figures per major experiment:

1. a categorical co-design phase diagram;
2. an inverse-design curve showing the predictor quality required for a chosen
   benefit threshold.

Use a compact actual-versus-required table for H5-C. Every plot must name:

- prediction metric and set semantics;
- physical normalization;
- capacity and candidate-budget semantics;
- profitability assumptions;
- whether the result is measured, trace-driven, or analytical.

Complete the human review checkpoint before beginning H7 or C1.
