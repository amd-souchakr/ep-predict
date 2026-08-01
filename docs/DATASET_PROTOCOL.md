# Dataset protocol

## Current workloads

There are two deliberately separate workloads:

1. `data/prompts/h1-pilot.jsonl`: 20 authored prompts used only for offline
   tests and a two-request instrumentation smoke run.
2. `h1-standard-small`: the research pilot, deterministically materialized from
   four revision-pinned public datasets.

The authored prompts must not be used as evidence for H1.

## Standard-small mixture

Use 32 examples from each source, for 128 requests total:

| Domain | Dataset and split | Input field | Why |
|---|---|---|---|
| General prose | WikiText-2 raw, validation | `text` | Small standard language-modeling corpus with natural prose |
| Mathematics | GSM8K main, test | `question` | Standard multi-step reasoning workload |
| Code | HumanEval, test | `prompt` | Compact standard code-generation workload |
| Conversation/instruction | MT-Bench prompts, train | first `prompt` turn | Standard compact chat-evaluation workload |

Only the problem/prompt or first user turn is used. Reference answers,
canonical solutions, tests, second turns, and assistant responses are excluded.
This avoids target leakage and keeps the prefill trace attributable to the
presented input.

Dataset repositories, configurations, splits, revisions, field extraction,
minimum-length filters, and prompt prefixes are frozen in
`configs/dataset/h1-standard-small.toml`. The materializer also hashes the final
JSONL. Collection hashes it again in the run definition, preventing accidental
resume against changed data.

## Sampling

- Stream the pinned split.
- Shuffle with a fixed per-source seed and a 10,000-row buffer.
- Filter empty/very short entries.
- Remove exact duplicate source text.
- Take 32 accepted examples per domain.
- Deterministically interleave the four domains before tracing.
- Preserve whitespace for source code.

Interleaving prevents routing windows from lining up with a single domain.
Per-domain statistics use the explicit domain label.

## What this can establish

Routing is not an intrinsic property of data alone. It is a joint property of:

- the model checkpoint and router;
- token and context distribution;
- prompt formatting;
- prefill versus generated decode;
- generation policy.

The experimental decomposition is:

- **Prefill, within domain:** strongest initial evidence for data-conditioned
  routing because tokens come from a fixed external dataset.
- **Decode, within domain:** operational evidence for the model's generated
  serving behavior conditioned on that dataset.
- **Across domains, same checkpoint:** evidence that routing changes with data
  distribution.
- **Within domain, repeated subsets:** evidence that observed hotness is stable
  rather than a sample artifact.
- **Across checkpoints/models later:** required to separate model-specific from
  broadly shared routing structure.

A single aggregate histogram cannot support an “intrinsic model–data property”
claim.

The H1 analyzer therefore reports pairwise domain Jensen-Shannon divergence and
hot-set overlap alongside within-domain split-half versions of the same
statistics. Cross-domain differences are evidence only when they exceed the
same-domain sampling drift.

## Evidence progression

1. **Smoke:** two authored prompts; validate hooks only.
2. **Standard pilot:** 32 examples × four domains, seed 7.
3. **Confirmation:** repeat with two more sampling seeds and report
   request-bootstrap confidence intervals.
4. **Model effect:** if H1 survives, repeat H1 only on a second checkpoint or
   top-k architecture. Do not duplicate later experiments yet.

The primary H1 statement should be based on per-domain prefill results.
Token-weighted mixed-domain results are secondary because prompt lengths differ.
