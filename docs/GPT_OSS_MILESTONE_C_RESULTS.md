# GPT-OSS 20B Milestone C results

**Decision:** `QUALIFIED`
**Review state:** complete pending researcher review; Milestone D remains blocked
**Checkpoint:** `openai/gpt-oss-20b` at
`6cee5e81ee83917806bbde320786a8fb61efebee`

## Result

The model-specific dispatch-boundary observer exactly matches the expert IDs
and selected weights consumed by the native MXFP4 grouped-GEMM path.

| Check | Result |
|---|---:|
| Routed layers covered | 24 / 24 |
| Prompt tokens | 6 |
| Top-k | 4 |
| Dispatch-consumed `(ID, weight)` pairs | 576 |
| Expert-ID mismatches | 0 |
| Selected-weight mismatches at `1e-6` | 0 |
| Maximum absolute selected-weight error | 0.0 |
| Dispatch-boundary hook calls | 24 |
| Ordinary router-module hook calls | 0 |

The zero ordinary-router-hook count is expected and decisive: Transformers
5.14.1 replaces the GPT-OSS MLP forward for MXFP4, computes the router linear
operation inline, and invokes Triton routing directly. A generic router hook
would silently collect nothing. The qualified observer instead hooks the
expert call and reads the exact `RoutingData`, gather index, and scatter index
passed to both fused expert GEMMs. It recovers each dispatched expert from the
consumed expert histogram and gather permutation, then pairs it with the
consumed dispatch weight.

An independent top-k calculation from the same layer's router logits agrees
with all consumed pairs. This comparison canonicalizes `(ID, weight)` pairs,
because router order is descending logit while the custom dispatch sorts
selected IDs and then forms an expert-major stable gather.

## Routing and storage semantics

- Geometry: 24 routed layers, 32 local experts/layer, top-4, hidden and expert
  intermediate size 2,880, and no shared expert.
- Weight normalization: softmax over the selected top-4 only. The consumed BF16
  values sum from 0.998046875 to 1.0029296875 because each stored value rounds
  independently; they match the independently computed BF16 values exactly.
- Storage: each serialized expert occupies 13,236,480 bytes
  (12.623 MiB). Each loaded expert occupies 13,253,760 measured bytes
  (12.640 MiB). The 17,280-byte increase is exactly the two expert bias vectors
  loading from checkpoint BF16 into FP32.
- Layer expert storage is 423,567,360 serialized bytes and 424,120,320 loaded
  bytes. The full checkpoint tensor payload is 13,761,264,768 bytes.
- Native implementation: `Mxfp4GptOssExperts`, BF16 activations/router,
  packed U8 MXFP4 blocks plus U8 scales, and fused Triton grouped GEMMs.

## Provenance

- Python 3.12.13
- PyTorch 2.11.0+rocm7.2 / HIP 7.2.26015
- Transformers 5.14.1
- tokenizers 0.22.2
- kernels 0.15.2 / kernels-data 0.16.0
- `kernels-community/gpt-oss-triton-kernels` snapshot
  `9655fcf7d0f638bec4a82f6f1a70014f0aa8cfb0`
- Device: one AMD Instinct MI355X (`gfx950` exposed as capability 9.5)

The immutable JSON also records hashes for the checkpoint config, tokenizer,
safetensors index, Transformers model/MXFP4 source, and loaded kernel sources.

## Interpretation and stop condition

Milestone C passes: there is no unobserved MXFP4 dispatch bypass, and the
model-specific observer is admissible for a later GPT-OSS tracer. This is not a
routing-distribution experiment, output-quality evaluation, timing result, or
20B end-to-end tracer bullet. Milestone D must not begin until the researcher
reviews this qualification and explicitly advances the sequence.

Canonical artifacts are under
`artifacts/runs/gpt-oss-20b-milestone-c/`.
