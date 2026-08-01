# AX architecture figures: human review

These figures combine measured anchors, trace-derived demand, assumed future-router quality, and hypothetical hardware. They are analytical projections, not measured speedups.

## Review checklist

- [ ] Figure 1 includes all four selected queue-sensitivity checks and holds each line's scenario fixed.
- [ ] Figure 2 compares predictive and reactive offload on the same hierarchy and does not claim equivalence to all-resident execution.
- [ ] Figure 2 includes all K=8/16/32 capacity points; its 99% coverage and 1.5× traffic are assumed future-predictor properties.
- [ ] Figure 3 is a necessary first-order average-bandwidth bound; queue and reliability tails remain separate.
- [ ] AX3's 192/384 MiB whole-expert double-buffer bounds remain recorded in `ax3_staging.csv` and `REPORT.md`.

## One next action

Review the queue-sensitive capacity point before selecting any live asynchronous calibration.
