# ep-predict

Hook-only experiments for testing whether MoE expert demand is skewed,
predictable, and useful for hierarchical memory placement. The model and
Transformers source remain unmodified.

Follow a "tracer bullet" development approach that develops a narrow vertical slice to a result instead of breadth and ceremony. This is a scientist's tool for rapid prototyping and experimentation, not a production software. Keep the human in the loop. Explain your findings, analysis, and interpretation from each experiment before moving to the next one.

The project advances one research gate at a time. H1 was mixed: one
workload-agnostic static tier failed, while domain-conditioned demand was
strong. H2 was pilot-supported: held-out layer-transition tables strongly beat
per-layer marginal popularity. H3 did not support replacing that simple policy
with a fixed linear hidden-state sidecar: selection coverage was essentially
tied, complete-route gains were domain-dependent, and candidate churn rose.
See [STATUS.md](STATUS.md), [docs/H3_PROTOCOL.md](docs/H3_PROTOCOL.md), and
[docs/H3_RESULTS.md](docs/H3_RESULTS.md).

The current evidence supports routing-transition-guided placement research for
the pinned OLMoE checkpoint. It does not justify a learned sidecar for this
checkpoint and does not establish transfer feasibility, latency improvement,
or universal MoE behavior. After human review, the next gate is the minimum H4
hardware-feasibility study using the simpler transition policy.

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

Install plotting dependencies:

```bash
uv sync --all-extras
```

Inspect the model before collecting data:

```bash
uv run ep-predict inspect \
  --config configs/model/olmoe-1b-7b-instruct.toml
```

Materialize the small standard workload:

```bash
uv sync --extra data --extra inference

uv run ep-predict prepare-dataset \
  --config configs/dataset/h1-standard-small.toml
```

Run H1:

```bash
uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/h1-standard-small.toml

uv run ep-predict analyze-h1 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h1-standard-small.toml

uv run ep-predict plot-h1 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h1-standard-small.toml
```

Reuse the trace for H2; no new inference is required:

```bash
uv run ep-predict analyze-h2 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h2-standard-small.toml

uv run ep-predict plot-h2 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h2-standard-small.toml
```

Run the hook-only H3 feature collection, fixed linear analysis, and figures:

```bash
uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/h3-standard-small.toml

uv run ep-predict analyze-h3 \
  --run artifacts/runs/h3-standard-small \
  --config configs/experiment/h3-standard-small.toml

uv run ep-predict plot-h3 \
  --run artifacts/runs/h3-standard-small \
  --config configs/experiment/h3-standard-small.toml
```

Reuse the same H3 artifacts for the post-hoc all-layer horizon analysis:

```bash
uv run ep-predict analyze-h3 \
  --run artifacts/runs/h3-standard-small \
  --config configs/experiment/h23-extended-horizon.toml

uv run ep-predict plot-extended-horizon \
  --run artifacts/runs/h3-standard-small \
  --config configs/experiment/h23-extended-horizon.toml
```

This trains the same fixed linear recipe for all 120 valid source-target pairs
per phase; it performs no additional inference. See
[docs/H23_EXTENDED_HORIZON_RESULTS.md](docs/H23_EXTENDED_HORIZON_RESULTS.md).

The collector writes one compressed JSONL routing trace per request. H3 also
writes one aligned numeric NPZ shard containing compact projected router
inputs. Both are crash-safe and resumable at request granularity; no full
hidden states or Python pickle artifacts are stored.

`data/prompts/h1-pilot.jsonl` remains only an instrumentation smoke fixture.
Research evidence uses the revision-pinned standard mixture described in
[docs/DATASET_PROTOCOL.md](docs/DATASET_PROTOCOL.md).

Every major experiment ends with scripted visualization and a human review
before the next hypothesis begins. See
[docs/EXPERIMENT_SOP.md](docs/EXPERIMENT_SOP.md).

## Invariants

- Inference only: no model training or router modification.
- Hooks capture the model's actual router output.
- Every hook run validates selected IDs against top-k router logits.
- Prefill and decode are never combined in headline metrics.
- Expert IDs are always keyed by layer.
- Raw traces and reports are artifacts, not source files.
- No latency claim is made from a hooked run.

The long-form research agenda remains in [RESEARCH.md](RESEARCH.md).
