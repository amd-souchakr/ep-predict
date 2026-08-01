# Research artifact retention

This directory is the canonical home of both runtime data and durable
experiment evidence. Do not copy completed results into a second `results/`
tree: publication and status documents should link to these paths directly.

## Tracked in Git

Keep all compact or publication-relevant products in their native run
directory:

- prepared dataset manifests and, when redistributable, exact prompt files;
- run definitions, run manifests, model reports, and environment metadata;
- analysis tables (`.csv`), decisions and summaries (`.json`), and generated
  reports or review notes (`.md`);
- fitted analysis outputs such as compact predictor weights (`.npz`);
- figure manifests and both publication PDF and review PNG figures;
- measured timing samples and analytical design tables.

These files are the durable evidence behind the claims in `docs/`,
`EXPERIMENT_LOG.md`, and `STATUS.md`. Plotting scripts consume them in place.

## Ignored and disposable

Only large inference-replay inputs are excluded from ordinary Git:

- `artifacts/runs/*/trace/`;
- `artifacts/runs/*/features/`;
- `artifacts/runs/*/hidden_states/`;
- `artifacts/runs/*/activations/`.

They may be deleted when local space is needed. A deleted raw input means the
corresponding analysis cannot necessarily be recomputed from inference, but
the checked-in tables, fitted outputs, reports, and figures preserve the
evidence used by the project.

## Closeout

After analysis, plotting, and human review, run:

```bash
uv run ep-predict audit-artifacts
git add artifacts
uv run ep-predict audit-artifacts --require-tracked
```

The audit verifies figure input/output hashes, checks result-document
references, enforces the raw-data boundary, and reports unexpectedly large
tracked artifacts. Commit the staged artifacts with the code, configuration,
and documentation that produced or interpreted them.
