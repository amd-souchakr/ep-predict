# Lean experiment strategy

## Architectural evidence ladder

The desired hardware conclusion is not supported by predictor accuracy alone.
The shortest defensible evidence chain is:

1. **Demand structure (H1):** expert demand is skewed or locally stable.
2. **Information (H2):** current routing state predicts future demand better
   than marginal popularity.
3. **Physics (H4):** an oracle can exploit the available lookahead under a
   plausible bandwidth/capacity/latency regime.
4. **Policy (H3/H5):** a lightweight sidecar predictor recovers a useful share
   of oracle benefit without excessive candidate traffic.
5. **Competition (H6):** prediction-guided residency or replication beats
   static caching, reactive load, and moving activations in a defined regime.

Each step is a kill switch. A negative result narrows the claim instead of
triggering more model complexity.

## Fast execution order

The hypothesis numbers describe logical claims, not the work queue. Use the
cheapest existing artifact first, but do not optimize a predictor before the
physical mechanism survives:

1. Answer H1/H2 with routing traces.
2. Try one fixed linear H3 sidecar when conditional information is clearly
   present.
3. Run cheap post-hoc structural scans, such as the full source-target matrix,
   rather than adding early conditional gates or ablation studies.
4. Run the minimum H4 oracle calculation from exact expert bytes, separately
   measured unhooked layer timing, and a host-to-device transfer curve.
5. Use a first-order normalized H5 sweep to map prediction-quality × hardware
   profitability windows and solve for minimum predictor requirements.
6. Place the existing transition and linear streams on that surface without
   retraining.
7. Compare residency/replication/JIT roles for H6 only after the analytical
   policy screen is favorable.
8. Consider routing-training intervention, sparse-model confirmation, or
   timing-fidelity work only after those requirements are explicit.

For the current project, H4–H6 are complete. H5 found a controlled analytical
window but rejected the unchanged K-wide candidate streams as transfer
policies. H6 then rejected using the same transition/linear depth predictions
for on-demand residency: they do not beat static/domain/LRU at equal capacity
and movement budget, even though the next-use oracle is substantially better.
The current mechanism stops after human figure review. No admission-head fit,
new inference, generic MLP sweep, projection sweep, H7/C1 setup, model
download, or overlap microbenchmark follows automatically.

## Simple gate, broad post-hoc view

Preregister one narrow engineering decision, not a complicated hypothesis tree.
After applying it unchanged, use inexpensive analysis of already collected data
to find layer, domain, phase, horizon, and capacity regimes. Record those
regimes as exploratory and use them to design the next physical experiment.

This keeps early experiments fast while avoiding the mistake of filtering out
useful structure merely because it was outside one primary aggregate.

Once hardware parameters exist, future predictor gates should be evaluated only
at issue points where bytes can plausibly arrive before demand. Coverage alone
is an information result; deadline-feasible bytes and residual stalls are the
architectural result.

For early co-design exploration, collapse expert size, bandwidth, and
lookahead into dimensionless cold-service pressure. The purpose is to expose
boundaries and required predictor quality, not to forecast exact latency.

## H5 first-order package

Treat controlled assumption sweep, inverse design, and existing-policy
placement as one hypothesis package:

1. categorize assumption cells as physics-, prediction-, speculative-traffic-,
   or jointly viable;
2. solve for minimum complete coverage and maximum amplification;
3. replay only the existing transition and linear streams at representative
   boundary/control points.

The default figures become one phase diagram and one inverse-requirement curve.
Detailed plan: [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md).

Current result: a first-order region exists, but actual-policy placement shows
that a broad candidate set is not an affordable movement set. H6 further shows
that smoothing the existing depth-prediction scores into an on-demand
residency belief does not recover that region. Predicting a token's route down
network depth and predicting reuse by later tokens are distinct tasks.

## Implementation boundary

The base model and inference library are treated as read-only:

```text
Hugging Face model inference
          │
          ├── root forward hooks: request/token/phase context
          └── router forward hooks: logits/weights/actual expert IDs
                              │
                              ▼
                 immutable routing trace
                    ├── H1/H2 analysis
                    ├── sidecar predictor training
                    └── trace-driven HW policy replay
```

No custom model fork, monkey patch, router rewrite, or serving-library edit is
needed. Hooked runs characterize workload only. Timing is measured in separate
unhooked runs so tracing overhead cannot contaminate latency claims.

## Minimum artifacts per run

Keep only:

- `run_definition.json`: frozen configs and fingerprint, written before work;
- `run_manifest.json`: model revision, environment, hardware, and completion;
- `model_report.json`: exact router mapping and expert bytes;
- compressed request traces;
- machine-readable CSV/JSON metrics;
- one short Markdown report;
- one to three decision-relevant PDF/PNG figures and their input-hash manifest.

Use `STATUS.md` for the current gate and `EXPERIMENT_LOG.md` for one-paragraph
decisions. Do not add a tracking service, database, notebook-only pipeline, or
large configuration framework during the prototype.

## Human-in-the-loop visualization gate

After every major experiment, generate the smallest set of plots needed to
understand the decision, then pause for human review before starting the next
hypothesis. The reviewer checks semantics, aggregation, thresholds, outliers,
regimes, and confounds against the machine-readable tables. Record the review
and one next action in Markdown; do not use visual inspection to retroactively
change the formal gate.

See `docs/EXPERIMENT_SOP.md` for the lean operating loop.

The default visual pair is:

1. a plain-language headline curve for the primary metric;
2. a compact heatmap exposing layer/domain/regime heterogeneity.

Do not add dashboards or experiment tracking services for this prototype.

## Evidence grades

- **Smoke:** synthetic/random tiny model; tests instrumentation.
- **Pilot:** checked-in 20-prompt workload; decides whether to spend more.
- **Confirmation:** larger corpus-backed workload, multiple order seeds,
  request-level uncertainty; supports a research statement.
- **Architecture:** calibrated timing plus competitive baselines; supports a
  HW/SW claim.

Reports must state their grade.

## Claim discipline

With a frozen model and a trained sidecar, the strongest direct claim is:

> Existing MoE routing state contains sufficient predictive signal for a
> lightweight policy to anticipate expert demand and manage limited fast
> residency or replication capacity.

The stronger wording “models learn to manage their own expert skew” requires
the later H7 intervention phase that co-trains or fine-tunes routing/control
behavior. H7 is conditional on a plausible H5 requirement and is outside the
current hook-only evidence; it must not be inferred from sidecar results.

The memory-hierarchy claim additionally requires a competitive win over moving
activations to resident experts or simply provisioning more local capacity.
