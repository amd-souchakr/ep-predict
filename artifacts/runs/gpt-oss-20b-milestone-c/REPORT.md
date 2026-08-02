# GPT-OSS 20B Milestone C qualification

**Decision:** `QUALIFIED` — pending researcher review.

The native MI355X MXFP4 run covered all 24 routed layers and all 576 consumed
expert-ID/weight pairs (6 tokens × 24 layers × top-4). It recorded zero ID
mismatches, zero selected-weight mismatches at absolute tolerance `1e-6`, and
maximum absolute weight error 0.0.

The ordinary `GptOssTopKRouter` hook fired 0 times because the MXFP4 replacement
MLP computes routing inline. The model-specific hook at the expert-dispatch
boundary fired exactly once in every layer and captured the `RoutingData` and
gather/scatter tensors consumed by the fused grouped GEMMs. This observed bypass
is covered, not ignored.

The checkpoint has 24 routed layers, 32 experts/layer, top-4 routing, no shared
expert, and BF16 compute inputs. Each expert is 13,236,480 stored bytes and
13,253,760 loaded bytes; the increase is the two BF16 checkpoint biases loaded
as FP32. Selected weights are a softmax over top-4. Router order is descending
logit; dispatch sorts selected IDs and then uses expert-major order.

This qualifies instrumentation only. It contains no routing-distribution or
performance result. Stop for review before Milestone D.

