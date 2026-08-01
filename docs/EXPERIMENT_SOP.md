# Lean experiment standard operating procedure

Use this loop for every major hypothesis experiment. It keeps a human in the
decision path without adding a tracking system or notebook ceremony.

## 1. Freeze

- State one simple decisive question, one primary scope, and one stop/go rule.
- Pin model, dataset, generation, and analysis configurations.
- Keep cheap descriptive and post-hoc analysis non-gating. Do not preregister
  every possible layer, domain, or hardware interaction.

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
- Then run cheap broad scans over structure already present in the artifacts,
  such as all source-target layer pairs. These scans may narrow the conclusion
  but never rewrite the original gate.
- Mine the result for boundary locations, empty windows, crossover points,
  dimensionless ratios, and which constraint is active. Keep this cheap and
  post-hoc; do not turn every observation into a new preregistered branch.
- When studying co-design, solve the inverse requirement as well as the forward
  result: report what predictor quality or hardware headroom would be needed
  to change the decision.
- Add request-level uncertainty only for a confirmation run or a genuinely
  borderline decision; do not bootstrap every pilot table.

## 5. Visualize

- Prefer one simple headline curve and, when heterogeneity matters, one
  layer/domain/regime heatmap. Add a third figure only if it changes a decision.
- For assumption-driven co-design work, prefer one categorical phase diagram
  and one inverse-design curve. Clearly distinguish candidate, analytically
  profitable, and experimentally demonstrated regions.
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
- Once a held-out set drives post-hoc policy discovery, treat it as development
  data. Use fresh requests only if the physical gate justifies confirmation.

For a fast prototype, a Markdown checklist in the generated figure directory
is sufficient evidence of review. Do not introduce an experiment-tracking
service unless artifact discovery becomes a real bottleneck.
