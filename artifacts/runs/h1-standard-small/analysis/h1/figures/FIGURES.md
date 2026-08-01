# H1 figures: `h1-standard-small`

These figures are generated from the immutable H1 metric tables. PNG files are
450 DPI for review; PDFs retain vector text and lines for publication.

## Figure 1 — Fast-tier capacity versus expert-demand coverage

Layer-local expert ranks are computed before averaging across the 16 layers.
The curves therefore never combine expert namespaces. The top axis converts
resident experts per layer to aggregate BF16 capacity using the inspected
12 MiB expert size. Coverage counts expert selections, not fully resident
token routes.

## Figure 2 — Skew–stability operating map

The dashed lines are the preregistered top-8 skew and Jaccard thresholds.
Filled points also pass the 0.80 lagged/oracle threshold. Only L6 and L9 pass
all three conditions for mixed decode. Per-domain points are pilot evidence
because some scopes contain only three or four complete 512-token windows.

## Figure 3 — Domain shift versus sampling drift

Between-domain routing divergence is compared with within-domain split-half
divergence on a log scale. The mean ratios are descriptive pilot statistics,
not confidence intervals.

## Human visual-review checkpoint

- [x] Axes, units, phase separation, and selection-versus-token semantics are
      clear.
- [x] Curves and points agree with the machine-readable report.
- [x] No important outlier, layer regime, saturation point, or domain confound
      is hidden by aggregation.
- [x] The reviewer records whether the figures support, weaken, or revise the
      experiment interpretation before the next hypothesis starts.

Review recorded 2026-08-01: proceed with the mixed H1 interpretation. Global
static placement failed, while domain-conditioned differences justify H2.
