# Testbed decision

## Primary: OLMoE-1B-7B-0125-Instruct

Use [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct)
for the first complete experiment.

Why it is a good prototype:

- 7B total and roughly 1.3B active parameters is small enough for the target
  24 GB GPU in BF16 while remaining a real, trained causal MoE.
- Sixteen MoE layers provide useful cross-layer lookahead for later H2/H3.
- Sixty-four experts and top-8 routing make skew, candidate coverage, and
  bandwidth amplification nontrivial.
- The official Transformers implementation exposes an `OlmoeTopKRouter` at
  each layer. Its forward returns `(router_logits, router_scores,
  router_indices)`, so a hook observes the exact indices passed to expert
  dispatch.
- The checkpoint, code, training data, and research artifacts are unusually
  open, which makes later routing comparisons easier to defend.

The [official model card](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
documents the active/total parameter counts and Transformers support. The
[Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py)
is the source of truth for hook semantics.

## Paired checkpoint control

C0 additionally qualified the direct pretrained ancestor
`allenai/OLMoE-1B-7B-0125` at revision
`9b0c1aa87e34a20052389dce1f0cf01da783f654`. It has exactly the same routing
geometry and BF16 expert size as the primary Instruct checkpoint.

The Base checkpoint is a matched scientific control, not a second
architecture. C0 used checkpoint-independent raw serialization and verified
identical input IDs before comparing trajectories. The endpoint result did not
justify downloading SFT or DPO; see [C0_RESULTS.md](C0_RESULTS.md).

## Why not the obvious alternatives

| Model | Prototype issue |
|---|---|
| Mixtral-8×7B | Too large for a clean BF16 single-24-GB run |
| Qwen1.5-MoE-A2.7B | About 14B total parameters; requires quantization or offload on the target GPU |
| DeepSeekMoE-16B | Larger and more architecture-specific than needed for H1 |
| GPT-OSS-20B | Compact weight formats help capacity, but fused/custom paths complicate transparent hook validation |
| Switch Transformer | Smaller variants are useful unit fixtures, but encoder-decoder/top-1 behavior is less representative of modern causal decode |

## Look-ahead caveat

OLMoE's top-8 routing is intentionally demanding. A negative JIT-prefetch result
could be partly architecture-specific because each token requests eight small
experts. If H1–H4 reveal a sharp top-k sensitivity, repeat only the decisive
trace/simulator experiments on a top-2 or top-4 model. Do not add a second
model before the first decision boundary is understood.

## Future testbeds

Adding another checkpoint is a deliberate, model-specific qualification task;
the project does not need a universal adapter first. Naming, evidence
isolation, comparable capacity semantics, and cross-model reporting rules are
in [ADDING_MODEL_TESTBED.md](ADDING_MODEL_TESTBED.md).
