# Extended-horizon figure review

This is post-hoc descriptive evidence and does not alter the formal H2/H3 gates.

## Automated headline

At decode K=16, transition selection coverage changes from 79.0% at Δ=1 to 53.8% at Δ=15, while linear changes from 79.4% to 69.2%. Complete-token coverage changes from 24.1% to 4.6% for transition and 28.7% to 19.7% for linear.

The Δ=15 point contains only layer 0→15; use the heatmap before attributing the aggregate trend to distance alone.

## Human review checklist

- [x] Horizon means are interpreted with the changing number of eligible source layers.
- [x] Heatmap cells agree with source/target layer semantics.
- [x] Selection and complete-token coverage are not conflated.
- [x] Long-horizon outliers and layer clusters are recorded.
- [x] One next action is recorded before H4.

Review recorded 2026-08-01: the early-versus-late source-layer regime is
accepted as post-hoc guidance, not a revision of the formal H3 gate.

Next action: preregister the minimum H4 oracle issue-point, bandwidth, and
capacity scan.
