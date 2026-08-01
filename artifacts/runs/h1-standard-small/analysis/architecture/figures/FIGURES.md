# AX architecture figures: human review

These figures combine measured anchors, trace-derived demand, assumed future-router quality, and hypothetical hardware. They are analytical projections, not measured speedups.

## Review checklist

- [ ] The phase-map axes are complete cold-set coverage and raw service headroom; amplification is applied within each panel.
- [ ] Green is read as an SLO candidate, not a demonstrated system.
- [ ] The Pareto comparison is predictive versus reactive offload on the same measured PCIe hierarchy.
- [ ] The all-resident point is a capacity/performance reference, not the baseline that predictive CPU offload must beat.
- [ ] The inverse curve is a necessary first-order bandwidth bound; queue and reliability tails remain separate constraints.
- [ ] AX3's 192/384 MiB whole-expert double-buffer bounds are checked against `ax3_staging.csv`.
- [ ] One architectural point is selected before any optional live asynchronous calibration.

## One next action

Pending researcher review.
