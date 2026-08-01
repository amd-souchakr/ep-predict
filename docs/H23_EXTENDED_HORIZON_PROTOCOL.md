# H2/H3 extended-horizon characterization

## Status and purpose

This is a post-hoc descriptive extension requested after the formal H3 gate.
It cannot change the preregistered H2 or H3 decisions.

Question:

> How do routing-transition and fixed linear-sidecar coverage change over every
> valid future-layer distance in the 16-layer OLMoE checkpoint?

No new inference is collected. The analysis reuses the complete H3 routing
trace and 128-dimensional projected router-input features.

## Frozen analysis

- Preserve the exact H2/H3 96/32 held-out request split.
- Preserve separate prefill, decode, and domain metrics.
- Use the same transition estimator and the same fixed H3 linear training
  recipe.
- Train one linear head for every valid
  `(phase, source layer, target layer)` pair.
- Evaluate \(\Delta=1,\ldots,15\) and \(K=8,16,32\).
- Report static popularity, domain popularity, transition, and linear policies.
- Use decode \(K=16\) for the two headline figures.
- Write under
  `artifacts/runs/h3-standard-small/analysis/h23_extended_horizon/` so formal
  H3 artifacts remain unchanged.

There are \(15+14+\cdots+1=120\) valid source-target pairs per phase and 240
linear heads overall. Every head retains the H3 feature size, loss, optimizer,
epochs, and seed. This is additional training only, not a tuning sweep.

## Figures

1. **Horizon curve:** selection and complete-token coverage for
   \(\Delta=1,\ldots,15\). Faint points show source-layer means across domains;
   thick lines show domain-balanced means. The number of eligible source
   layers, \(16-\Delta\), is printed along the x-axis.
2. **Triangular gain heatmap:** one cell for every valid source-target pair,
   showing linear-minus-transition gain for selection and complete-token
   coverage.

## Interpretation constraint

The horizon mean changes composition: \(\Delta=1\) averages 15 source layers,
while \(\Delta=15\) contains only layer 0→15. Therefore:

- use the curve to summarize operational coverage across all available issue
  points;
- use the heatmap to expose layer-specific regimes;
- separately inspect the fixed layer-0 trajectory when attributing change to
  distance rather than source-layer composition.

Long-distance results have the same number of held-out tokens per pair but
fewer independent source-layer scopes. Do not interpret a smooth aggregate
line as a source-layer-controlled causal decay curve.
