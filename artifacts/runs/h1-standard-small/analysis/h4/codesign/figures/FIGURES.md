# H4/H2/H3 co-design map review

This is post-hoc descriptive synthesis. It does not change the H3 or H4 formal gates.

## Reading the map

- Headroom below 1 means mean serialized cold-transfer work exceeds nominal lead time.
- K is deliberately coupled across per-layer fast-tier capacity and prediction candidate budget for this screening slice; it is not a policy replay.
- Complete-route coverage below 50% leaves prediction as the dominant limitation.
- Filled markers independently pass the trace-driven oracle on-time-byte and stall-reduction thresholds.
- The upper-right region is only eligible for a policy replay; profitability requires measured overlap and learned/oracle recovery.

## Human review checklist

- [x] The headroom ratio and complete-route metric are interpreted independently.
- [x] Open points above headroom 1 are recognized as tail or queue failures, not contradictions.
- [x] Candidate-region points are not called profitable.
- [x] Changing eligible target-layer count at long Δ is retained as a limitation.
- [x] Next action: quantify analytical profitability windows, inverse predictor requirements, and existing-policy placement.

**Review completed:** 2026-08-01. The researcher accepted the trajectory view:
long-range prediction expands the overlap with physical headroom, while
profitability remains an analytical H5 question.
