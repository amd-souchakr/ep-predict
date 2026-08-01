# H5 admission-separation protocol

**Frozen:** 2026-08-01  
**Status:** post-hoc H5 mechanism diagnosis  
**Evidence grade:** held-out trace-driven analytical pilot

## Question

Does the existing predictor ranking separate useful from useless nonresident
expert candidates strongly enough that a cost-sensitive admission rule can
reduce transfer amplification from approximately 6.5× to at most 2× without
destroying complete cold-set coverage?

The exact threshold is descriptive, not a gate. The objective is to identify
whether a qualitative separation boundary exists before training a more
complex predictor.

## Frozen scope

- Reuse the unchanged H2/H3 96/32 request split.
- Score only held-out decode requests.
- Replay the full request order to obtain K=32 per-layer LRU residency.
- Evaluate Δ=3, the short physical boundary, and Δ=9, the long-range linear
  regime.
- Use the existing transition tables and already-trained fixed linear heads.
- Score all 64 target-layer expert IDs.
- Suppress resident expert IDs before transfer admission.
- Exclude waves with no residual cold experts from complete-cold-set metrics.

## Cost-sensitive admission proxy

Raw score scales differ across source-target heads. Within each wave, normalize
the 64 expert scores to zero mean and unit variance. Sweep one shared
standardized-score threshold. Admit only nonresident experts above it.

This is a deliberately simple confidence rule. It tests ranking separation,
not final calibration. No threshold is tuned on the held-out labels.

Report:

- useful and false admitted experts;
- admitted/useful transfer amplification;
- cold-expert coverage;
- complete cold-set coverage;
- mean admitted experts per cold wave;
- useful-versus-useless score AUROC;
- histogram Jensen–Shannon divergence, overlap, and label mutual information
  as descriptive separation statistics.

The plotted classes are conditional outcomes among **nonresident** expert IDs:
`actually demanded and cold` versus `not demanded`. They are not globally hot
versus globally cold expert populations. Jensen–Shannon divergence is primary
because it is symmetric, finite, and bounded by 1 bit. Directed KL estimates
use a 0.5-count correction and are reported only as bin-dependent diagnostics.

## Figures

Create exactly:

1. transfer amplification versus complete cold-set coverage over the threshold
   sweep;
2. standardized score distributions over nonresident expert IDs, separated
   into demanded-cold and useless candidates.

Mark A=2× and 50% complete cold-set coverage as reference lines. They define
the previously frozen H5 screen, not a newly optimized threshold.

## Interpretation

- A frontier crossing near A≤2 and substantial complete coverage means the
  current ranking contains an admission signal; test a simple policy before an
  MLP.
- Strong distribution overlap and collapsed coverage at A≤2 mean ranking
  quality is the limiting factor and a more expressive predictor becomes
  better motivated.

This analysis performs no inference, model modification, inference-library
modification, predictor retraining, H7 intervention, or C1 model setup.
