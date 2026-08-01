# Lean experiment standard operating procedure

Use this loop for every major hypothesis experiment. It keeps a human in the
decision path without adding a tracking system or notebook ceremony.

## 1. Freeze

- State the hypothesis, scope, metrics, thresholds, and stop/go rule.
- Pin model, dataset, generation, and analysis configurations.
- Separate primary, secondary, and exploratory analyses.

## 2. Validate

- Run the smallest instrumentation smoke test.
- Require trace/schema/router integrity before scaling.
- Confirm that resumed artifacts match the frozen configuration fingerprint.

## 3. Collect

- Write restartable request-level artifacts.
- Preserve raw traces; never repair them in place after analysis.
- Record environment, hardware, revisions, and request completion.

## 4. Analyze

- Generate machine-readable tables before figures.
- Keep layer, phase, domain, and top-k semantics explicit.
- Apply the frozen decision gate before exploratory interpretation.

## 5. Visualize

- Generate only one to three decision-relevant figures from the immutable
  tables or traces.
- Use checked-in scripts, not manual plotting or a notebook as the source of
  truth.
- Save vector PDF and high-resolution PNG plus a manifest that hashes the
  figure inputs.
- Label selection-, token-, wave-, and step-level quantities precisely.
- Avoid uncertainty marks until the corresponding resampling unit is defined.

## 6. Human visual-review checkpoint

Pause before starting the next hypothesis. The researcher reviews the figures
and records:

- whether axes, units, aggregation, baselines, and thresholds are correct;
- whether machine-readable headline values agree with the plots;
- visible regimes, outliers, layer clusters, saturation points, and confounds;
- whether the visual evidence supports, weakens, or changes the automated
  decision;
- the single next action.

The review is not permission to change a preregistered gate after seeing the
result. Post-hoc findings must remain explicitly exploratory.

## 7. Decide and hand off

- Update `EXPERIMENT_LOG.md` with the formal result and human interpretation.
- Update `STATUS.md` with supported, mixed, rejected, or inconclusive.
- Link the metric report, figures, and review note.
- Advance only after the human has reviewed the result.

For a fast prototype, a Markdown checklist in the generated figure directory
is sufficient evidence of review. Do not introduce an experiment-tracking
service unless artifact discovery becomes a real bottleneck.
