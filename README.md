# ep-predict

Hook-only experiments for testing whether MoE expert demand is skewed,
predictable, and useful for hierarchical memory placement. The model and
Transformers source remain unmodified.

Follow a "tracer bullet" development approach that develops a narrow vertical slice to a result instead of breadth and ceremony. This is a scientist's tool for rapid prototyping and experimentation, not a production software.

The project advances one research gate at a time. The current gate is **H1:
expert popularity is skewed and stable at an operationally useful scope**.
See [STATUS.md](STATUS.md) for the live state and
[docs/H1_PROTOCOL.md](docs/H1_PROTOCOL.md) for the preregistered pilot.

## Testbed

The primary model is `allenai/OLMoE-1B-7B-0125-Instruct`:

- 7B total and about 1.3B active parameters;
- 16 MoE layers;
- 64 routed experts per layer;
- top-8 routing;
- an explicit router module returning logits, routing weights, and selected
  expert IDs.

The BF16 checkpoint is about 13.8 GB and fits the target 24 GB GPU. The
implementation discovers routers by module behavior and attributes rather than
hard-coding OLMoE into the trace format.

## Quick start

Install the inference dependencies:

```bash
uv sync --extra inference
```

Inspect the model before collecting data:

```bash
uv run ep-predict inspect \
  --config configs/model/olmoe-1b-7b-instruct.toml
```

Run the small H1 pilot:

```bash
uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/h1-pilot.toml

uv run ep-predict analyze-h1 \
  --run artifacts/runs/h1-pilot \
  --config configs/experiment/h1-pilot.toml
```

The collector writes one compressed JSONL trace per request. This is
deliberately simple, crash-safe, and resumable. It is sufficient for the pilot;
Arrow/Parquet should be added only when hidden-state features or million-event
traces make JSON decoding a bottleneck.

## Invariants

- Inference only: no model training or router modification.
- Hooks capture the model's actual router output.
- Every hook run validates selected IDs against top-k router logits.
- Prefill and decode are never combined in headline metrics.
- Expert IDs are always keyed by layer.
- Raw traces and reports are artifacts, not source files.
- No latency claim is made from a hooked run.

The long-form research agenda remains in [RESEARCH.md](RESEARCH.md).
