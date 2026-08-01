# H5 first-order co-design results

**Run:** `artifacts/runs/h1-standard-small/analysis/h5`  
**Evidence grade:** trace-driven analytical pilot  
**Decision:** controlled design region exists; unchanged candidate streams do
not enter it  
**Human figure review:** pending

## Plain-language conclusion

The hardware idea has a first-order profitable region, but the current
transition and linear predictors should not be used as blind prefetch lists.
Their coverage is already adequate in the K=32 cells; the failure is that each
useful cold expert is accompanied by roughly five false nonresident candidates.

At K=32, the predictors cover 67–81% of complete cold sets, but transfer
amplification is 6.3–6.7× useful bytes. None of the eight representative
policy points passes the frozen 2× traffic screen. A selective admission or
residency policy—not a larger predictor—is now the highest-value use of the
existing signal.

## Frozen controlled sweep

The analysis swept 68,175 assumption cells over complete cold-set coverage,
candidate transfer amplification, capacity, every lookahead through Δ=15, and
0.25×–4× measured bandwidth. A cell passed only with:

- at least 25% modeled reactive-stall reduction;
- at least 50% recovery of the first-order oracle ceiling;
- at most 2× candidate bytes per useful predicted cold byte.

The sweep found 22,618 qualifying cells. This count is not itself a result;
the boundary is. At measured bandwidth and 1× amplification, minimum required
complete cold-set coverage rises from 25–35% in low-headroom cells to 50% once
the physical oracle ceiling saturates. With 2× amplification, the K=8 and K=16
Δ=1 windows are empty; both first become viable at Δ=2. K=32 remains viable
from Δ=1.

## Existing-policy placement

All values use the unchanged 96/32 held-out split. Fast-tier state is replayed
in full request order, but only held-out requests are scored. Already-resident
candidate experts are suppressed before transfer accounting.

| Cell | Policy | Complete cold sets | Cold experts | Transfer amp. | Modeled stall reduction | Oracle recovery |
|---|---|---:|---:|---:|---:|---:|
| K=32, Δ=1 | transition | 77.4% | 88.6% | 6.5× | 9.5% | 13.5% |
| K=32, Δ=1 | linear | 80.9% | 90.7% | 6.6× | 9.5% | 13.6% |
| K=32, Δ=3 | transition | 76.5% | 87.8% | 6.5× | 30.0% | 30.0% |
| K=32, Δ=3 | linear | 81.5% | 90.7% | 6.7× | 30.1% | 30.1% |
| K=32, Δ=9 | transition | 67.1% | 82.4% | 6.3× | 67.1% | 67.1% |
| K=32, Δ=9 | linear | 77.0% | 87.9% | 6.3× | 77.0% | 77.0% |
| K=16, Δ=9 | transition | 28.9% | 63.4% | 3.8× | 28.9% | 28.9% |
| K=16, Δ=9 | linear | 42.4% | 72.7% | 3.4× | 42.4% | 42.4% |

No row passes. K=32, Δ=1 is both traffic- and physics-limited. K=32, Δ=3
crosses the 25% benefit floor but recovers only 30% of the oracle and exceeds
the traffic cap. K=32, Δ=9 has enough coverage and lead time, but fails only
the traffic gate. K=16, Δ=9 is additionally prediction-limited.

## Main insight

Complete top-8 prediction was too pessimistic for placement decisions because
many missed route experts are already resident. The operational prediction
target is the residual cold set. At K=32, complete cold-set coverage is
67–81%, even though earlier complete-route coverage was lower.

The converse is equally important: a K-wide candidate list is not a transfer
list. At K=32, sending every nonresident candidate causes 6.3–6.7×
amplification. Reaching the 2× gate would require rejecting roughly 81–82% of
false transfers while retaining useful candidates. At K=16, Δ=9 the analogous
requirement is 58–64%.

This changes the next mechanism question from “can we predict more experts?”
to:

> Can a cheap admission/residency policy convert broad trajectory predictions
> into a small set of high-value movements?

## Post-hoc score-separation result

The follow-up admission analysis answers that question more sharply. It scores
all 64 expert IDs, removes K=32 residents, standardizes scores within each
wave, and sweeps one shared confidence threshold without retraining.

Useful cold experts are ranked substantially above useless nonresident experts:

| Lookahead | Policy | AUROC | JS divergence | Distribution overlap | Useful base rate | Score/label MI |
|---:|---|---:|---:|---:|---:|---:|
| Δ=3 | transition | 0.850 | 0.308 bits | 44.6% | 7.3% | 0.095 bits |
| Δ=3 | linear | 0.883 | 0.381 bits | 38.7% | 7.3% | 0.126 bits |
| Δ=9 | transition | 0.803 | 0.223 bits | 53.1% | 7.9% | 0.070 bits |
| Δ=9 | linear | 0.861 | 0.332 bits | 42.8% | 7.9% | 0.117 bits |

This is related to KL divergence, but it is not a hot-versus-cold expert
comparison. Both curves contain only nonresident expert IDs. They estimate
the standardized score distribution conditional on the expert later being
`actually demanded and cold` or `not demanded`. The symmetric, finite
Jensen–Shannon (JS) divergence is therefore easier to interpret than directed
KL. The visible shift is real and the linear head improves it, but 39–53% of
the two probability masses still overlap.

The base rate explains the apparent tension between strong AUROC and weak
admission economics. Only 7–8% of scored nonresident IDs are useful. The score
therefore carries 0.07–0.13 bits about the useful/useless label—18–33% of the
label's available entropy—but not enough evidence to turn most high-scoring
candidates into useful transfers. Distributional separation is a ranking
result, not proof of a low-amplification operating point.

The separation is real but insufficient for a scalar threshold to meet both
H5 references:

| Lookahead | Policy | Complete cold sets at A≤2× | Minimum A for C≥50% |
|---:|---|---:|---:|
| Δ=3 | transition | 7.5% | 4.0× |
| Δ=3 | linear | 28.4% | 3.0× |
| Δ=9 | transition | no 2× crossing; 4.3% at minimum 2.7× | 5.0× |
| Δ=9 | linear | 22.8% | 3.3× |

Thus the linear ranking roughly halves the original 6.3–6.7× amplification
while preserving 50% complete coverage, but cannot reach 2×. The qualitative
boundary is approximately 3× for the current linear scores.

This does not justify a generic MLP accuracy sweep. It does justify one
targeted cost-sensitive admission experiment: first fit per-head calibration
or a very small admission model using score, rank, margin, residency, recent
demand, and layer/horizon context. Its objective must be complete cold-set
coverage at a byte budget, not ordinary route recall. The current held-out set
is now development data for that policy.

## Limitations

- one OLMoE top-8 checkpoint and the current 32 held-out requests;
- first-order proportional FIFO service, not live prefetch execution;
- no eviction or cache pollution caused by speculative insertion;
- complete cold waves are credited proportionally rather than with a
  trace-level transfer schedule;
- no end-to-end latency, energy, cost, or multi-GPU claim;
- the repeatedly inspected split is development data for any newly designed
  admission policy.

## Decision and next action

H5 is mixed: the co-design opportunity is analytically plausible, but the
existing raw candidate streams are not profitable prefetch policies. Do not
train a larger predictor or collect a new model yet.

After human figure review, run one lean H6 analysis on the current testbed:
compare reactive JIT, static/domain residency, and a simple
prediction-guided admission/residency score. The decisive metric is whether
admission can approach ≤2× transfer amplification while preserving enough
complete cold-set coverage to recover at least half of the analytical oracle.

## Artifacts

- Protocol: [H5_PROTOCOL.md](H5_PROTOCOL.md)
- Design points: `analysis/h5/h5_design_points.csv`
- Viable windows: `analysis/h5/h5_windows.csv`
- Inverse requirements: `analysis/h5/h5_inverse_requirements.csv`
- Existing policies: `analysis/h5/h5_policy_placement.csv`
- Gate: `analysis/h5/gate.json`
- Figures/review: `analysis/h5/figures/`
- Admission protocol: [H5_ADMISSION_PROTOCOL.md](H5_ADMISSION_PROTOCOL.md)
- Admission frontier and score separation:
  `analysis/h5/admission/figures/`
