# GPT-OSS route-head horizon trade-off

**Evidence role:** post-hoc exploratory extension of the frozen confirmation

**Training support:** all 276 forward source-target pairs, `Δ=1…23`

**Confirmatory scope:** only `Δ=1…3`

**Evaluation:** unchanged checkpoint, 64 fresh requests, no refit

The predictor was not trained only through `N+3`. Its 276 independent heads
cover every forward pair in the 24-layer model:

\[
\sum_{N=0}^{22}(23-N)=23+22+\dots+1=276.
\]

The original decision protocol restricted primary reporting to `Δ=1…3`
because those are the likely useful transfer horizons. The retained fresh
traces and frozen checkpoint nevertheless permit an exploratory evaluation of
all horizons through `Δ=23`.

## Result

At fixed `K=8`, earlier prediction causes a real but surprisingly modest loss:

| Δ | Valid layer pairs | Selection coverage | Routed-mass coverage | Complete top-4 |
|---:|---:|---:|---:|---:|
| 1 | 23 | 91.7% | 93.1% | 74.1% |
| 3 | 21 | 90.0% | 91.5% | 70.8% |
| 6 | 18 | 89.1% | 90.5% | 69.0% |
| 12 | 12 | 87.2% | 88.8% | 65.5% |
| 18 | 6 | 87.4% | 88.9% | 67.2% |
| 23 | 1 | 84.7% | 86.5% | 62.8% |

From `Δ=1` to `Δ=23`, K=8 selection coverage falls 7.0 points and complete
top-4 coverage falls 11.3 points. The curve is not monotone: it plateaus and
occasionally rebounds. Long-range layer routing retains substantial structure;
this is not a rapidly decorrelating next-step process.

Candidate amplification partially buys back the lost horizon. At `Δ=23`:

| Candidate count | Amplification | Selection coverage | Complete top-4 |
|---:|---:|---:|---:|
| 4 | 1× | 67.0% | 35.9% |
| 8 | 2× | 84.7% | 62.8% |
| 12 | 3× | 91.5% | 77.6% |
| 16 | 4× | 94.7% | 84.6% |

Thus K=12 at `Δ=23` nearly recovers the K=8 selection coverage at `Δ=1`, but
it does so by nominating half again as many experts. Whether that is profitable
depends on the Milestone G transfer and residency budget.

The learned heads also remain stronger than the transition table at long
range. At `Δ=23`, transition K=8 reaches 78.1% selection and 46.8% complete
coverage, versus 84.7% and 62.8% for the frozen heads.

## Interpretation boundary

The x-axis is operational lookahead, not a clean causal estimate of distance.
At `Δ=1`, the mean contains 23 source-target pairs; at `Δ=23`, it contains only
layer 0→23. Source layer, target layer, and horizon therefore change together.
Uncertainty also widens because forecasts per request fall from 368 to 16.

The request bootstrap handles within-request token and layer correlation, but
it cannot remove this layer-composition confound. A future causal
horizon-ablation should compare fixed target-layer cohorts and should be
preregistered. Consequently, `Δ=4…23` is useful design evidence but does not
retroactively enlarge the confirmed Milestone F claim.

## Artifacts

- [Horizon trade-off plot](../artifacts/analysis/gpt-oss-20b-long-horizon-exploratory/fig1_long_horizon_tradeoff.png)
- [PDF figure](../artifacts/analysis/gpt-oss-20b-long-horizon-exploratory/fig1_long_horizon_tradeoff.pdf)
- [Request-level metrics](../artifacts/analysis/gpt-oss-20b-long-horizon-exploratory/request_metrics.csv)
- [Bootstrap summaries](../artifacts/analysis/gpt-oss-20b-long-horizon-exploratory/horizon_summary.csv)
- [Machine-readable result](../artifacts/analysis/gpt-oss-20b-long-horizon-exploratory/result.json)
- [Reproduction script](../scripts/analyze_gpt_oss_long_horizons.py)
