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

The hypothesis numbers describe logical claims, not the most efficient work
queue. After H1:

1. Run H2 with transition tables and conditional entropy; this is nearly free
   once the trace exists.
2. Build the minimum oracle feasibility calculation from exact expert bytes,
   separately measured unhooked layer time, and a PCIe transfer curve. This
   advances H4 early as a physical kill switch.
3. Train a linear H3 predictor only if either conditional information is real
   or oracle feasibility exists. Add an MLP only if the linear model leaves a
   meaningful oracle gap.
4. Replay learned policies for H5.
5. Compare residency/replication/JIT roles for H6.

This resolves a tension in the research agenda: learned predictor work is
listed before the oracle simulator, while the oracle-first gate correctly says
not to optimize prediction for a physically impossible mechanism.

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
- one short Markdown report.

Use `STATUS.md` for the current gate and `EXPERIMENT_LOG.md` for one-paragraph
decisions. Do not add a tracking service, database, notebook-only pipeline, or
large configuration framework during the prototype.

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

The stronger wording “models learn to manage their own expert skew” requires a
later intervention phase that co-trains or fine-tunes routing/control behavior.
That is outside this prototype and should not be inferred from sidecar results.

The memory-hierarchy claim additionally requires a competitive win over moving
activations to resident experts or simply provisioning more local capacity.
