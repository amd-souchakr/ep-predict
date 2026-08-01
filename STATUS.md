# Project status

**Current hypothesis:** H2 — routing-conditioned future expert demand
**Current stage:** H2 pilot supported; figures awaiting human visual review
**Last updated:** 2026-08-01

| Gate | Question | State | Exit evidence |
|---|---|---|---|
| M0 | Can we inspect and validate the model without source changes? | Passed | Model report, complete hook traces, and zero integrity failures |
| H1 | Is hotness strong and stable enough for a fast tier? | Mixed; global gate failed | 2/16 mixed decode layers passed; code and math are locally strong |
| H2 | Does conditional locality beat marginal popularity? | Pilot supported | All Δ=1/2/3 transition baselines passed the held-out decode gate |
| H3 | Can a small predictor cover compact future expert sets? | Not started | Candidate-budget coverage curves |
| H4 | Is oracle prefetch physically viable? | Not started | Hardware feasibility phase diagram |
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
- [x] Generate the three scripted H2 decision figures.
- [ ] Human reviews H2 figures and records whether to advance to H3.

Full interpretation: [docs/H2_RESULTS.md](docs/H2_RESULTS.md).

## Evidence policy

`Ready` means code and protocol exist, not that the hypothesis is supported.
Only immutable output under `artifacts/runs/<run-id>/analysis/` can change a
hypothesis state to supported, mixed, or rejected.
