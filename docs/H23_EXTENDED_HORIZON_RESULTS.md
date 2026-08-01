# Extended H2/H3 horizon findings

## Plain-language conclusion

Future expert routing remains predictable all the way to OLMoE's final MoE
layer, but the answer depends more on **which source layer issues the
prediction** than on distance alone.

At decode \(K=16\), the routing-transition table degrades substantially as its
lookahead grows from \(\Delta=1\) to \(\Delta=15\):

- selection coverage falls from 79.0% to 53.8%;
- complete top-8 coverage falls from 24.1% to 4.6%.

The fixed linear hidden-state sidecar degrades much less:

- selection coverage falls from 79.4% to 69.2%;
- complete top-8 coverage falls from 28.7% to 19.7%.

At the maximum layer-0→15 lookahead, linear prediction is substantially better
than every simple baseline:

| Decode, layer 0→15, K=16 | Selection coverage | Complete top-8 coverage |
|---|---:|---:|
| Static target-layer popularity | 37.4% | 0.1% |
| Domain-oracle popularity | 48.2% | 1.9% |
| Route transition | 53.8% | 4.6% |
| Linear projected hidden state | 69.2% | 19.7% |

This does not overturn the formal H3 failure: the linear sidecar is not a
reliable **global replacement** for transitions at \(\Delta=1\). It supports a
narrower and architecturally relevant result: early-layer hidden states are
much better than early routes for anticipating far-future expert demand.

## Scope and integrity

- Post-hoc descriptive analysis; formal H2/H3 gates remain unchanged.
- No new inference or feature collection.
- Reused all 377,488 H3 projected layer-events and routing records.
- Exact H2/H3 96/32 request split.
- Evaluated all 120 valid source-target pairs per phase.
- Trained 240 fixed linear heads with the original H3 recipe.
- Evaluated \(K=8,16,32\), \(\Delta=1,\ldots,15\), prefill/decode, and all
  four domains.
- Reproduced all 1,008 original H2 transition scopes with zero difference.

Artifacts are under
`artifacts/runs/h3-standard-small/analysis/h23_extended_horizon/`.

## Global horizon behavior

Domain-balanced decode results at \(K=16\):

| Δ | Eligible source layers | Transition selection | Linear selection | Transition complete | Linear complete |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 79.0% | 79.4% | 24.1% | 28.7% |
| 3 | 13 | 76.8% | 79.3% | 22.2% | 28.8% |
| 5 | 11 | 74.1% | 78.4% | 19.9% | 28.4% |
| 8 | 8 | 70.7% | 76.5% | 15.9% | 27.4% |
| 10 | 6 | 66.7% | 74.4% | 12.0% | 25.1% |
| 12 | 4 | 62.8% | 72.0% | 9.9% | 21.9% |
| 15 | 1 | 53.8% | 69.2% | 4.6% | 19.7% |

The linear advantage grows with the reported horizon. At \(\Delta=15\), it is
+15.5 points for selection and +15.1 points for complete-token coverage.

However, this aggregate curve changes its source-layer composition. The final
point is only layer 0→15, whereas \(\Delta=1\) averages fifteen different
source layers. It should not be read as a controlled decay curve.

## Fixed-source result: distance alone is not the main effect

Holding source layer 0 fixed gives a cleaner comparison:

| Layer 0 predicts | Transition selection | Linear selection | Transition complete | Linear complete |
|---:|---:|---:|---:|---:|
| Layer 1 | 59.1% | 69.9% | 3.7% | 15.5% |
| Layer 3 | 59.2% | 74.5% | 3.3% | 20.7% |
| Layer 6 | 62.3% | 75.5% | 8.0% | 25.0% |
| Layer 9 | 64.0% | 80.0% | 11.3% | 36.7% |
| Layer 12 | 58.8% | 74.9% | 8.7% | 27.6% |
| Layer 15 | 53.8% | 69.2% | 4.6% | 19.7% |

The linear layer-0 readout does not monotonically lose accuracy with target
distance. It is strongest for middle/deep targets and ends close to its
layer-1 selection coverage. This indicates that source-layer identity and the
target layer's routing structure matter at least as much as raw \(\Delta\).

## Source-target structure

Across the 120 domain-balanced decode source-target pairs at \(K=16\):

- linear selection coverage exceeds transition in 100/120 pairs;
- linear complete-token coverage exceeds transition in 112/120 pairs;
- median gains are +3.1 selection points and +8.5 complete-token points.

The gains form a clear depth regime:

| Source layer | Mean selection gain across all future targets | Mean complete-token gain |
|---:|---:|---:|
| 0 | +14.7 pp | +17.2 pp |
| 1 | +11.0 pp | +15.0 pp |
| 2 | +3.9 pp | +9.2 pp |
| 5 | +2.3 pp | +7.4 pp |
| 8 | +1.2 pp | +4.4 pp |
| 10 | −0.8 pp | +2.1 pp |
| 12 | −5.3 pp | −4.0 pp |
| 14 | −6.2 pp | −3.2 pp |

Early hidden states preserve semantic/token information that an eight-expert
route compresses away. Near the end of the model, the current route becomes
the better simple predictor. A single global choice between "linear" and
"transition" is therefore the wrong policy abstraction.

## Domain, capacity, phase, and churn

At maximum lookahead, linear beats transition in every domain:

| Domain, layer 0→15, K=16 | Transition selection | Linear selection | Transition complete | Linear complete |
|---|---:|---:|---:|---:|
| Code | 62.5% | 77.9% | 11.5% | 32.5% |
| Conversation | 50.0% | 63.9% | 3.0% | 10.9% |
| General prose | 50.8% | 63.2% | 0.8% | 13.3% |
| Mathematics | 51.8% | 72.0% | 3.2% | 22.0% |

The maximum-horizon linear advantage persists across capacity:

| K | Transition selection | Linear selection | Transition complete | Linear complete |
|---:|---:|---:|---:|---:|
| 8 | 34.8% | 48.9% | 1.1% | 2.5% |
| 16 | 53.8% | 69.2% | 4.6% | 19.7% |
| 32 | 77.3% | 87.3% | 21.0% | 51.1% |

Prefill shows the same qualitative pattern. At layer 0→15 and \(K=16\),
linear reaches 72.6% selection and 29.7% complete coverage versus 52.9% and
9.4% for transition.

The cost warning remains: decode linear candidate churn rises from 52.8% at
\(\Delta=1\) to 58.7% at \(\Delta=15\), while transition churn is 34.0% at
\(\Delta=15\). Prediction accuracy must therefore be translated into
wave-level transfers, reuse, and residency rather than treating every changing
candidate as an immediate load.

## Architectural interpretation

The extended view strengthens the information side of the architecture:

1. Long lookahead is available. A layer-0 hidden-state sidecar identifies about
   69% of final-layer selections at 2x candidate amplification.
2. Early issue points are precisely where linear prediction adds the most
   information and where hardware has the most time to act.
3. Route transitions remain attractive near the target because they are
   simpler, lower-churn, and often more accurate for late source layers.
4. A later policy should select the predictor by source-target regime:
   hidden-state sidecar for early/long-range planning, transition tables for
   late/short-range refinement.

This is evidence for hierarchical planning, not yet for transfer feasibility.
The minimum H4 study should remain oracle-first, scan issue point as an
explicit dimension, and carry both existing transition and linear candidate
streams as non-optimized comparisons only after the oracle exposes a viable
region. Do not begin MLP, projection tuning, or a new inference workload.

## Figures

- `fig1_extended_horizon_coverage`: global horizon curves with individual
  source-layer means and eligible-layer counts.
- `fig2_source_target_gain_heatmap`: one cell per valid source-target pair,
  showing linear-minus-transition gains.

Human visual review completed on 2026-08-01. The formal H3 decision remains
unchanged; the early-versus-late source-layer regime is accepted as exploratory
guidance for H4.
