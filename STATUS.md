# Project status

**Current hypothesis:** H3 — lightweight future-route prediction
**Current stage:** H3 primary gate failed; post-hoc all-layer analysis complete;
human figure review is pending
**Last updated:** 2026-08-01

| Gate | Question | State | Exit evidence |
|---|---|---|---|
| M0 | Can we inspect and validate the model without source changes? | Passed | Model report, complete hook traces, and zero integrity failures |
| H1 | Is hotness strong and stable enough for a fast tier? | Mixed; global gate failed | 2/16 mixed decode layers passed; code and math are locally strong |
| H2 | Does conditional locality beat marginal popularity? | Pilot supported | All Δ=1/2/3 transition baselines passed the held-out decode gate |
| H3 | Can a small predictor beat transition tables at equal candidate budget? | Pilot not supported; review pending | +0.4 pp selection, +4.7 pp complete at primary gate; only 2/4 domains positive on both |
| H4 | Is oracle prefetch physically viable? | Recommended after H3 review | Hardware feasibility phase diagram using the transition policy |
| H5 | Does prediction recover oracle benefit? | Not started | Learned/oracle recovery |
| H6 | Is prediction better for residency than JIT loading? | Not started | Policy comparison |

## Immediate run checklist

- [x] Confirm the actual machine exposes the intended 24 GB NVIDIA GPU.
- [x] Install the `data` and `inference` dependency groups.
- [x] Materialize and review the revision-pinned standard-small workload.
- [x] Run model inspection and retain `model_report.json`.
- [x] Run two standard examples with `--limit 2`; require zero integrity errors.
- [x] Run the 128-request standard pilot.
- [x] Review H1 gate output before expanding the workload.
- [x] Generate the three scripted H1 decision figures.
- [x] Human reviewed H1 and approved advancing with the mixed interpretation.
- [ ] If borderline, run the confirmation workload with more requests and
      request-level bootstrap intervals.
- [x] Decide H1 and record the outcome in `EXPERIMENT_LOG.md`.
- [x] Preregister H2 with a request-level held-out split and fixed gate.
- [x] Reuse the H1 trace to evaluate static, domain, lagged, and transition
      baselines at K=8/16/32 and Δ=1/2/3.
- [x] Generate H2 diagnostics, then replace the primary view with two simple
      lookahead plots.
- [x] Human reviewed the simplified H2 figures and approved advancing to H3.
- [x] Preregister the minimal H3 proof/disproof gate before collecting hidden
      features or training a predictor.
- [x] Validate deterministic projected router-input capture on two requests.
- [x] Collect all 128 H3 requests with aligned routing and feature shards.
- [x] Train the single fixed linear recipe on the preserved 96/32 split.
- [x] Apply the primary H3 gate and reproduce all H2 transition metrics.
- [x] Generate two simple H3 decision figures and hash their inputs.
- [x] Extend H2/H3 descriptively through Δ=15 without new inference.
- [x] Generate the global horizon curve and source-target gain heatmap.
- [ ] Human review the H3 figures and record the decision before starting H4.

Full interpretation: [docs/H3_RESULTS.md](docs/H3_RESULTS.md) and
[docs/H23_EXTENDED_HORIZON_RESULTS.md](docs/H23_EXTENDED_HORIZON_RESULTS.md).

## Evidence policy

`Ready` means code and protocol exist, not that the hypothesis is supported.
Only immutable output under `artifacts/runs/<run-id>/analysis/` can change a
hypothesis state to supported, mixed, or rejected.
