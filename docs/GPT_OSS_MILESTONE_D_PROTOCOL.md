# GPT-OSS 20B end-to-end tracer-bullet protocol

**Frozen:** 2026-08-01, before Milestone D inference
**State:** authorized by the researcher's explicit Milestone D instruction
**Checkpoint:** `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`
**Scope:** tracing-workflow qualification only; no routing-distribution
comparison and no performance claim

## Decisive question

Can the Milestone C-qualified native MXFP4 dispatch observer support a complete,
reproducible GPT-OSS 20B artifact chain from frozen text through tokenization,
deterministic inference, routing capture, integrity analysis, retained output,
compact tables, and scripted figures?

## Frozen workload and execution

- Run the two prompts in
  `configs/experiment/gpt-oss-20b-milestone-d.toml`, in recorded order and at
  batch size one.
- Use the checkpoint-bundled chat template with
  `current_date="2026-08-01"`, `add_generation_prompt=true`, and no external
  text or dataset.
- Load only the local revision-pinned snapshot, preserve native MXFP4, expose
  one MI355X with `HIP_VISIBLE_DEVICES=0`, and use neither distributed
  execution nor vLLM.
- Use evaluation/inference mode, seed `20260801`, greedy argmax decoding, and
  exactly eight retained output tokens per request.
- After sampling each output token, feed it through one cached-token forward.
  The last such call is a terminal trace-only forward: its logits are not used
  to retain a ninth token. This makes every retained output token eligible for
  and represented in the routing trace.
- Repeat the complete workload once without reloading the model. Retain the
  first trace and outputs, and compare the second run against it. Hooked
  execution time is not evidence and is not reported.

## Observer and stored semantics

At each routed layer, independently compute top-4 IDs and selected-softmax
weights from the hidden states and router parameters at the MLP boundary. At
the qualified expert boundary, decode the exact `RoutingData`, gather
permutation, and dispatch weights consumed by the MXFP4 grouped GEMMs. Store
the consumed `(expert_id, weight)` pairs in canonical ascending expert-ID order
using the project's standard per-request trace schema.

Record prompt text, rendered input IDs and hashes, generated IDs and text,
checkpoint/config/tokenizer hashes, software/device provenance, model
inspection, run definition, integrity tables, routing summaries, and hashes of
all retained artifacts.

## Frozen exit gate

Milestone D is `QUALIFIED` only if all conditions hold:

1. the loaded model is the native `Mxfp4GptOssExperts` path with the frozen
   24-layer, 32-expert, top-4 geometry;
2. both requests retain exactly eight generated tokens;
3. every prompt and retained generated token has exactly one record at every
   routed layer, with four ID/weight pairs per record;
4. all eligible token positions form a complete contiguous range and there
   are no duplicate token-layer keys;
5. every dispatch-consumed pair matches the independent router calculation,
   with zero ID mismatches and selected-weight error at most `1e-6`;
6. repeat two has identical rendered input IDs, generated output IDs, phases,
   token positions, layer keys, and expert IDs; its maximum selected-weight
   difference is at most `1e-6`;
7. the trace shards, outputs, inspection, integrity report, compact routing
   table, figures, and artifact manifest are present and parseable.

The descriptive routing tables and figures qualify the artifact workflow.
They must not be interpreted as a routing distribution result, domain effect,
model comparison, or performance measurement. Stop for researcher review
after this gate; do not begin the 120B Milestone E experiment.
