# GPT-OSS 20B Milestone D result

**Decision:** `QUALIFIED`

The two-request deterministic tracer retained 161 prompt tokens and 16 generated tokens, producing 4248 token-layer records and 16992 consumed ID/weight pairs. Every eligible token is covered at all 24 layers. Dispatch parity has zero ID mismatches, zero selected-weight mismatches, and maximum absolute error 0.0. The immediate repeat reproduced all rendered inputs, generated IDs, routing IDs, and selected weights within the frozen tolerance.

The outputs, standard trace shards, inspection, integrity tables, compact routing summary, two scripted figures, and hash manifest complete the artifact chain. This qualifies the tracing workflow only. The small convenience workload cannot support a routing-distribution, domain, quality, or performance conclusion, and Milestone E remains blocked on review.
