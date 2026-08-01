# H5 admission-separation result

**Result:** useful ranking signal, insufficient scalar admission

Useful cold experts score materially higher than useless nonresident experts.
The fixed linear sidecar reaches AUROC 0.883 at Δ=3 and 0.861 at Δ=9,
outperforming transition scores at both horizons.

The plot is related to distributional divergence, but it does not compare
globally hot and cold experts. It compares predictor scores for two classes
among nonresident IDs: experts that are actually demanded (and therefore cold)
versus IDs that are not demanded. Linear JS divergence is 0.381 bits at Δ=3
and 0.332 bits at Δ=9, with 38.7% and 42.8% distribution overlap. Only 7–8% of
the scored IDs are useful, so the corresponding score/label mutual information
is 0.126 and 0.117 bits.

However, one shared standardized-score threshold cannot meet both H5
references. At transfer amplification A≤2×, linear complete cold-set coverage
is 28.4% at Δ=3 and 22.8% at Δ=9. Preserving at least 50% complete coverage
requires A=3.0× and 3.3×, respectively. Transition requires A=4.0× and 5.0×.

The current ranking roughly halves the unfiltered 6.3–6.7× traffic cost, but
does not reach 2× without destroying set coverage. This motivates a targeted,
cost-sensitive admission model or per-head calibrator—not generic predictor
capacity tuning.
