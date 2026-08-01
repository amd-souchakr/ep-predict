# H3 figure review

## Automated conclusion

Decision: **PILOT_DOES_NOT_SUPPORT**.

At the primary decode K=16, n+1 gate, the linear sidecar changes selection coverage by +0.4 percentage points and complete-token coverage by +4.7 points versus the transition table. It does not satisfy the preregistered consistency and selection-gain conditions.

## Human review checklist

- [x] Axes, units, aggregation, and baselines are correct.
- [x] Headline values agree with `gate.json` and `summary.csv`.
- [x] Domain heterogeneity and candidate churn are considered.
- [x] The reviewer accepts or challenges the automated H3 decision.
- [x] One next action is recorded before H4 starts.

Review recorded 2026-08-01: the formal H3 failure stands. The post-hoc
all-layer scan narrows the interpretation to an early-layer linear and
late-layer transition regime.

Next action: run the minimum oracle-first H4 hardware-feasibility study; carry
both existing streams only after oracle feasibility and do not tune predictors.
