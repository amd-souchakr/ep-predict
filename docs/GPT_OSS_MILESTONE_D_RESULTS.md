# GPT-OSS 20B Milestone D results

**Decision:** `QUALIFIED`
**Review state:** verified by the researcher; advanced to the 20B-only Milestone E
**Run:** `artifacts/runs/gpt-oss-20b-milestone-d`
**Checkpoint:** `openai/gpt-oss-20b` at
`6cee5e81ee83917806bbde320786a8fb61efebee`

## Result

The native MXFP4 GPT-OSS 20B path completed the frozen two-request tracer
bullet twice on one MI355X. The first pass retained the standard trace shards,
tokenized inputs, eight generated tokens per request, model inspection,
integrity tables, routing summaries, and figures. The second pass was an
immediate determinism check.

| Check | Result |
|---|---:|
| Requests | 2 |
| Prompt tokens | 161 |
| Retained generated tokens | 16 |
| Routed layers | 24 / 24 |
| Token-layer records | 4,248 / 4,248 |
| Dispatch-consumed `(ID, weight)` pairs | 16,992 |
| Missing / unexpected / duplicate token-layer keys | 0 / 0 / 0 |
| Bad phase or top-k-width records | 0 / 0 |
| Dispatch ID mismatches | 0 |
| Dispatch selected-weight mismatches at `1e-6` | 0 |
| Dispatch maximum absolute weight error | 0.0 |
| Repeat input/output token mismatches | 0 / 0 |
| Repeat routing-ID / selected-weight mismatches | 0 / 0 |
| Repeat maximum absolute weight error | 0.0 |

The factual request has 81 prompt plus eight retained output tokens, producing
2,136 records. The arithmetic request has 80 plus eight, producing 2,112.
Every token position is present once at each of 24 routed layers. A terminal
cached-token forward covers the eighth retained output token; its next-token
logits are intentionally discarded.

## Outputs and interpretation

The eight-token cap catches the beginning of GPT-OSS's analysis channel rather
than a completed final answer. Those token sequences are retained and repeat
exactly, which is sufficient for this tracing-workflow gate. They are not an
answer-quality result. Increasing generation length would add cost without
strengthening the frozen completeness or dispatch-parity question.

The compact routing table materializes both prefill and decode records across
all layers. Prefill touches 24--31 of 32 experts per layer and the tiny decode
slice touches 7--26, but these descriptive counts are only artifact sanity
checks. Two prompts and 16 decode tokens cannot support routing-distribution,
domain, skew, or predictability claims.

## Artifact chain

- `run_definition.json`: frozen config, revisions, hashes, environment, and
  Milestone C evidence link;
- `model_inspection.json`: loaded geometry, class, dtype, device, and chat
  template hash;
- `trace/*.jsonl.gz`: two standard request trace shards;
- `outputs.jsonl`: prompt/input/output token IDs, hashes, and decoded text;
- `integrity.json` and `layer_integrity.csv`: coverage, dispatch parity, and
  repeatability evidence;
- `routing_summary.csv`: compact phase/layer descriptive table;
- `figures/`: PDF and 450-DPI PNG routing-raster and
  occupancy/completeness figures;
- `artifact_manifest.json`: SHA-256 hashes for every retained evidence file,
  including the raw trace shards and report.

Milestone D therefore qualifies the end-to-end GPT-OSS tracing workflow. It
does not qualify a routing-distribution comparison, timing result, output
quality result, or 120B implementation. The researcher subsequently verified
this milestone and replaced the 120B plan with the disk-bounded 20B prediction
study reported in
[GPT_OSS_MILESTONE_E_RESULTS.md](GPT_OSS_MILESTONE_E_RESULTS.md).
