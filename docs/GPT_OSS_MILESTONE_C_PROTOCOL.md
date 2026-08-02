# GPT-OSS Transformers dispatch qualification protocol

**Frozen:** 2026-08-01, before executing the qualification harness
**State:** ready to execute
**Checkpoint:** `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`
**Tokenizer:** checkpoint-bundled tokenizer at the same revision
**Transformers:** `5.14.1` (source hashes recorded by the run)
**Scope:** Milestone C only; no routing-distribution analysis and no Milestone D

## Decisive question

Can model-specific hooks capture exactly the expert IDs and weights consumed by
GPT-OSS dispatch, including the checkpoint's native MXFP4/custom-kernel path?

## Frozen evidence contract

The qualification has two stages.

1. **Tiny eager control.** Instantiate a small `GptOssMLP`, attach an ordinary
   router output hook and an expert input hook, execute deterministic synthetic
   hidden states, and require exact ID and selected-weight parity. This checks
   the documented eager semantics without loading checkpoint weights.
2. **Native checkpoint path.** Load the local 20B checkpoint in its native
   MXFP4 representation on one visible MI355X. Attach a forward pre-hook to
   every expert module. The hook records the `RoutingData`, gather index, and
   scatter index passed to the fused grouped GEMMs. Decode expert IDs from the
   dispatch histogram plus the actual gather permutation, and compare each
   consumed `(expert_id, weight)` pair with an independent top-k computation
   from the same layer's router logits. Run one deterministic short prefill.

The native path deliberately does not use an ordinary router-module hook as
its primary observer. In Transformers 5.14.1, the MXFP4 MLP computes the router
linear operation inside its replacement forward and calls Triton routing
directly, bypassing `GptOssTopKRouter.forward`. The experiment records this
ordinary-hook bypass and requires complete dispatch-boundary hook coverage.

## Frozen inputs and execution

- Use only the already-downloaded 20B snapshot; do not download 120B.
- Run with `local_files_only=True` and the exact snapshot revision/path.
- Expose one GPU with `HIP_VISIBLE_DEVICES=0`; do not initialize distributed,
  tensor-parallel, expert-parallel, or vLLM execution.
- Use `eval()`, inference mode, batch size one, and the fixed text
  `"Milestone C dispatch parity."` without a chat template.
- Execute one ordinary prefill forward. Timing is not evidence.
- Preserve native MXFP4 (`use_kernels=False`); a BF16-dequantized run does not
  qualify the custom path.

## Recorded semantics

The run must record:

- model, tokenizer, Transformers, PyTorch, ROCm, and kernel provenance;
- routed-layer count, local experts per layer, top-k, and hidden size;
- checkpoint tensor shapes/dtypes, total stored bytes, per-expert stored bytes,
  and loaded tensor bytes;
- compute/input dtype and the actual expert class;
- ordinary router-hook calls and dispatch-boundary hook calls per layer;
- router ordering, dispatch ordering, selected-weight normalization, and
  whether shared experts exist;
- per-layer token counts, ID mismatches, maximum absolute weight error, and
  malformed dispatch permutations.

Checkpoint bytes mean serialized tensor payload from the safetensors index;
file bytes are reported separately. Loaded bytes are calculated from the
actual resident expert tensors and scale storage where the custom tensor API
exposes it, with any unmeasurable component reported rather than inferred.

## Pass/fail gate

`QUALIFIED` requires all of the following:

- zero eager hook-to-expert ID mismatches and zero selected-weight mismatches;
- zero native hook-to-dispatch ID mismatches for every tested token and layer;
- native selected weights agree within absolute tolerance `1e-6`;
- selected top-k weights sum to one within four BF16 epsilons (the stored BF16
  values can round independently even when computed by softmax);
- one native dispatch-boundary hook call for every routed layer and no missing
  or extra layer calls;
- the native expert implementation is MXFP4/custom-kernel, not dequantized
  eager experts;
- the ordinary router-hook bypass is observed and explicitly documented, with
  no additional unobserved bypass;
- provenance and all required routing/storage semantics are complete.

Any failed condition yields `NOT_QUALIFIED`. Instrumentation work stops there;
no routing trace is admissible and Milestone D remains blocked. A pass also
stops for researcher review before Milestone D.

## Outputs

All outputs are isolated under
`artifacts/runs/gpt-oss-20b-milestone-c/`:

- `qualification.json`: immutable provenance, semantics, checks, and gate;
- `dispatch_parity.csv`: one row per routed layer;
- `REPORT.md`: concise interpretation and claim boundary;
- `figures/`: scripted PDF/PNG summary and hash manifest.
