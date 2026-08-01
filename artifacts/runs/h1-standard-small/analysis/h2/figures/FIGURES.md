# H2 figures: `h1-standard-small`

These figures use held-out requests only. PNG files are 450 DPI; PDFs retain
vector text and lines.

## Simple conclusion

The current route predicts future routing well. With 16 candidates, it finds
79.0% of the eight experts used in the next MoE layer and 76.8% three layers
ahead. The drop from n+1 to n+3 is only 2.2 percentage points. Static
popularity finds about 41%.

## Figure 1 — Future experts found

`fig1_predictability_by_lookahead` reports the fraction of the actual top-8
future experts contained in the 16 candidates. This is the clearest evidence
that current routing predicts routing one to three layers ahead.

## Figure 2 — Complete future route found

`fig2_complete_route_by_lookahead` uses the stricter metric: all eight future
experts must be among the 16 candidates. Coverage is 24.1% at n+1 and 22.2% at
n+3, compared with less than 1% for static popularity. Predictability is real,
but guaranteeing the entire top-8 route remains difficult.

The older capacity, layer/domain heatmap, and coverage/churn figures in this
directory are supplementary diagnostics rather than primary communication
figures.

## Human visual-review checkpoint

- [x] A reader can state the H2 conclusion after viewing Figure 1 for a few
      seconds.
- [x] The difference between "future experts found" and "all eight found" is
      clear.
- [x] Headline values agree with `REPORT.md`, `summary.csv`, and `gate.json`.
- [x] The plots are interpreted as predictability evidence, not yet as latency
      or hardware benefit.
- [x] The reviewer records whether to advance to an external H3 predictor.

Review recorded 2026-08-01: the simplified figures clearly support the H2
pilot decision. Advance to a minimal H3 proof/disproof experiment; require the
learned sidecar to beat the transition table before optimization or ablations.
