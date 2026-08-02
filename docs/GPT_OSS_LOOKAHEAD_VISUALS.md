# Lookahead expert-demand visual aids

These three aids explain the mechanics of the frozen GPT-OSS 20B route
predictor using one real token from the 64-request fresh-confirmation run.
They are qualitative illustrations, not substitutes for the aggregate
confirmation statistics.

## Shared visual language

- **Blue:** the predictor nominated the expert early, but this token did not
  use it at the target layer. It represents candidate amplification and
  potential residency or transfer cost.
- **Green:** the target router selected an expert already present in the
  earlier prediction. This is a demand-coverage “cache hit.”
- **Red:** the target router selected an expert absent from the earlier
  prediction. This is a demand-coverage “cache miss.”
- **Gray:** actual demand at an initial layer that had no earlier source layer
  from which to make the specified lookahead prediction.

“Cache hit” and “cache miss” are deliberately narrow terms here. The figures
show membership in the predicted candidate set. They do not simulate finite
capacity, eviction, transfer completion, contention, or latency; Milestone G
owns those system effects.

## Figure 1: one token at three lookaheads

[PNG](../artifacts/visuals/gpt-oss-lookahead/fig1_single_token_horizons.png) ·
[PDF](../artifacts/visuals/gpt-oss-lookahead/fig1_single_token_horizons.pdf)

Each panel aligns the prediction emitted at layer `N` with actual demand at
layer `N+Δ`. The target router selects four experts while the predictor
nominates eight. This makes hits, misses, and unused speculative candidates
visible simultaneously.

## Figure 2: layer-by-layer lookahead animation

[Animated GIF](../artifacts/visuals/gpt-oss-lookahead/fig2_lookahead_cache_dynamics.gif)

The animation fixes `Δ=2` and `K=8`. A gold outline marks the current layer;
the dashed blue column is the layer being prefetched. When that target layer is
eventually reached, each pending prediction resolves into green hits, red
misses, and blue candidates that were not used.

## Figure 3: candidate amplification

[PNG](../artifacts/visuals/gpt-oss-lookahead/fig3_candidate_amplification.png) ·
[PDF](../artifacts/visuals/gpt-oss-lookahead/fig3_candidate_amplification.pdf)

The same token and fixed `Δ=2` are shown with `K=4, 8, 12, 16`. Increasing K
generally converts red misses into green hits, but the growing blue area makes
the additional residency or transfer pressure explicit.

## Figure 4: forecast cone and cache ledger

[Animated GIF](../artifacts/visuals/gpt-oss-lookahead/fig4_forecast_cone_cache_ledger.gif)

The left panel shows the three candidate sets emitted after observing the
current layer's weighted top-4 route. The right panel independently resolves
the `Δ=2` batch emitted two layers earlier: candidates become covered demand or
unused prefetches, while actual experts absent from that batch enter as late
misses. Fixing the ledger to one horizon preserves a causal interpretation;
it does not pretend that the three overlapping forecast sets constitute a
fully specified cache policy.

## Reproducibility

The example token is selected deterministically as a medoid of the fresh
confirmation token population, rather than chosen manually for an especially
good route. The exact request, token position, per-figure metrics, source
hashes, color semantics, and output hashes are recorded in
[the visualization manifest](../artifacts/visuals/gpt-oss-lookahead/visualization_manifest.json).

Regenerate all three aids on CPU with:

```bash
uv run python scripts/plot_gpt_oss_lookahead_mechanics.py
```
