# Project status

**Current hypothesis:** H1 — expert popularity is skewed  
**Current stage:** standard workload materialized; real-model run pending
**Last updated:** 2026-07-31

| Gate | Question | State | Exit evidence |
|---|---|---|---|
| M0 | Can we inspect and validate the model without source changes? | Ready to run | Model report plus zero router-validation mismatches |
| H1 | Is hotness strong and stable enough for a fast tier? | Active | Per-layer/domain/phase skew and window-stability report |
| H2 | Does conditional locality beat marginal popularity? | Blocked by H1 | Transition and conditional-entropy gain |
| H3 | Can a small predictor cover compact future expert sets? | Not started | Candidate-budget coverage curves |
| H4 | Is oracle prefetch physically viable? | Not started | Hardware feasibility phase diagram |
| H5 | Does prediction recover oracle benefit? | Not started | Learned/oracle recovery |
| H6 | Is prediction better for residency than JIT loading? | Not started | Policy comparison |

## Immediate run checklist

- [ ] Confirm the actual machine exposes the intended 24 GB NVIDIA GPU.
- [x] Install the `data` and `inference` dependency groups.
- [x] Materialize and review the revision-pinned standard-small workload.
- [ ] Run model inspection and retain `model_report.json`.
- [ ] Run two standard examples with `--limit 2`; require zero integrity errors.
- [ ] Run the 128-request standard pilot.
- [ ] Review H1 gate output before expanding the workload.
- [ ] If borderline, run the confirmation workload with more requests and
      request-level bootstrap intervals.
- [ ] Decide H1 and record the outcome in `EXPERIMENT_LOG.md`.

## Evidence policy

`Ready` means code and protocol exist, not that the hypothesis is supported.
Only immutable output under `artifacts/runs/<run-id>/analysis/` can change a
hypothesis state to supported, mixed, or rejected.
