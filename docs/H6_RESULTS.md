# H6 prediction-guided residency results

**Decision:** `PILOT_DOES_NOT_SUPPORT`  
**Evidence grade:** trace-driven pilot  
**New inference or predictor training:** none

## Plain-language conclusion

The existing routing predictors are good at predicting where the **current
token** will route later in the network, but they do not make better cache
retention decisions than simple domain popularity or reactive LRU.

At equal fast-tier capacity and a one-expert-per-wave movement cap,
prediction-guided residency fails the preregistered H6 gate. The oracle remains
much better than LRU, so useful residency decisions are possible in principle;
the missing information is future **reuse across tokens**, not future routing
along the current token's depth trajectory.

## Frozen primary result

Held-out decode, \(K=16,\Delta=3\):

| Policy | Residual cold demand | Complete resident-set hits | Useful / wasted MiB per wave |
|---|---:|---:|---:|
| Static popularity | 58.3% | 0.7% | 0.00 / 0.00 |
| Domain popularity | 49.5% | 3.4% | 0.00 / 0.00 |
| Reactive LRU | 48.1% | 2.8% | 8.14 / 3.53 |
| Transition-guided | 50.2% | 3.4% | 5.73 / 3.80 |
| Linear-guided | 48.8% | 3.5% | 7.33 / 3.71 |
| Oracle next-use | 31.2% | 11.8% | 8.94 / 0.49 |

The guided policies slightly improve aggregate complete-set hits over LRU, but
do so while serving fewer expert occurrences from residency. More importantly,
they do not beat the strongest matched static/domain/LRU comparator in each
layer/domain scope.

## Preregistered gate

| Guided policy | Expert-stall gain | Complete-set gain | Positive scopes on both | Positive domains | Positive layers | Pass |
|---|---:|---:|---:|---:|---:|:---:|
| Transition | -3.9 pp | -0.7 pp | 5/52 | 0/4 | 0/13 | No |
| Linear | -2.5 pp | -0.6 pp | 3/52 | 0/4 | 0/13 | No |

The gate required at least +2 pp on both headline metrics, positive gains on
both in at least half of the scopes, and breadth across at least three domains
and four layers.

## Why it failed

### The predictor optimizes the wrong axis for this policy

H2/H3 predict a target layer for the same token. H6 needs to know whether an
expert admitted now will be reused by later tokens before eviction. These are
different conditional distributions:

\[
P(E_{\ell+\Delta,t}\mid \text{state}_{\ell,t})
\quad\ne\quad
P(E_{\ell,t+\tau}\mid \text{history}).
\]

Depth-trajectory predictability is real, but it is not automatically a
temporal reuse forecast.

### Movement selectivity is worse than LRU

At the primary point:

- LRU: 69.7% of insertions earn a later hit;
- linear: 66.3%;
- transition: 60.0%;
- oracle: 94.7%.

Transition reduces movement by bypassing more misses, but its saved bytes come
with higher residual cold demand. Linear nearly spends the LRU budget while
still retaining less useful state.

### The oracle gap is not the bottleneck

The equal-budget oracle cuts residual cold demand from 48.1% to 31.2% and
raises complete resident-set hits from 2.8% to 11.8%. This is a meaningful
policy ceiling. H6 therefore rejects the existing depth predictors for this
residency rule; it does not prove that expert residency itself is useless.

## Broad descriptive scan

The heatmap covers all valid source-target pairs, both phases, all four
domains, and \(K=8,16,32\).

- Decode: no domain-balanced source/lookahead cell improves both expert and
  complete-set metrics at \(K=16\) or \(K=32\).
- Decode \(K=8\): only 2/120 cells have positive mean gains on both metrics,
  and neither is positive in three domains.
- Prefill has a narrow weak band around several middle target layers, but no
  cell reaches +2 pp on both metrics.
- Larger \(K=32\) makes the guided policy distinctly worse: LRU complete hits
  are 26.3%, linear 22.4%, transition 22.6%, and oracle 51.1% at \(\Delta=3\).

Positive cells are isolated and metric-dependent, not a coherent placement
regime. The failure is not a small aggregate artifact.

## Decision and next action

For this checkpoint and mechanism:

> Routing trajectories are scientifically predictable, but the current
> transition and linear depth predictors are not useful residency controllers.

Do not fit a cost-sensitive predictor, tune an MLP, modify the router, start
H7, collect fresh confirmation, or download a second model to rescue this
negative gate. Pause for human review of the H6 heatmap. Any later work must
pose a genuinely different mechanism or predict temporal reuse directly.

## Artifacts

- Protocol: [H6_PROTOCOL.md](H6_PROTOCOL.md)
- Config: `configs/experiment/h6-residency.toml`
- Machine report:
  `artifacts/runs/h1-standard-small/analysis/h6/REPORT.md`
- Scope table:
  `artifacts/runs/h1-standard-small/analysis/h6/scope_metrics.csv`
- Summary:
  `artifacts/runs/h1-standard-small/analysis/h6/summary.csv`
- Gate:
  `artifacts/runs/h1-standard-small/analysis/h6/gate.json`
- Figure:
  `artifacts/runs/h1-standard-small/analysis/h6/figures/`
