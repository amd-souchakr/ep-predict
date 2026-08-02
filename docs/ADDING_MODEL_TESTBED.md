# Adding another MoE testbed

**Status:** deferred guidance; use only after explicit approval of a new model
or checkpoint.

The project does not need a universal model adapter. Router discovery, hook
placement, loading, and any fused-kernel workaround may be wired up once for
each chosen checkpoint. The important requirement is that evidence from
different models remains distinct until an explicit cross-model analysis
combines it.

## Governing rule

Every result is scoped first to:

1. model ID and exact revision;
2. routing geometry and loaded weight representation;
3. workload, tokenizer/chat template, and generation policy;
4. hardware and software environment;
5. hypothesis and analysis configuration.

An H2 result on one checkpoint does not update the H2 state of another
checkpoint, and neither should silently become a universal project result.

## One-time qualification for a new model

Before a full run:

- add a model-specific config and loading path;
- identify the explicit router hook and validate its actual selected IDs;
- record routed-layer count, experts per layer, top-k, hidden size, expert
  bytes, quantization/storage representation, and any shared experts;
- run a tiny architecture-level hook test when practical;
- run `inspect`, then a one- or two-request smoke collection with zero missing
  routers, mismatches, or incomplete token routes;
- confirm the model fits the intended device topology and that hooks are not
  bypassed by a fused execution path.

This is a qualification task, not a reason to build generalized discovery,
offload, or distributed-runtime infrastructure prematurely.

## Names and artifact isolation

Until the artifact ignore rules are deliberately made recursive, retain flat,
model-prefixed run IDs:

```text
configs/model/<model-slug>.toml
configs/experiment/<model-slug>-h1-standard-small.toml
artifacts/runs/<model-slug>-h1-standard-small/
docs/<MODEL_SLUG>_H1_RESULTS.md
```

Do not overwrite the OLMoE configs, analysis directories, figures, or result
documents. Keep analysis products beside their own run at the canonical
`artifacts/runs/<run-id>/analysis/` path, and apply the normal artifact audit.

## Experimental comparability

Reuse the same prompt records and request split when possible, but record that
tokenization, chat formatting, truncation, and generated decode text are
model-dependent.

Do not blindly reuse absolute capacities or gates. Report at least:

\[
\text{candidate amplification}=K/k
\]

and:

\[
\text{candidate-set fraction}=K/E,
\]

where \(K\) is prediction candidate count, \(k\) is routing top-k, and \(E\)
is the expert count. Residency is an independent variable \(R\), normalized as
\(R/E\); do not substitute \(K/E\) for \(R/E\) unless a separately stated
staging policy makes them equal. Also derive expert bytes and available
routed-layer horizons from the new testbed. Complete-set coverage naturally
becomes easier or harder as top-k changes; that is an architectural result,
not a nuisance to normalize away.

Preregister the new model's gate before seeing its result. Reuse the scientific
question and metric definitions, but adjust model-dependent capacity,
class-imbalance, horizon, and physical-calibration parameters explicitly.

### Paired checkpoints from one training lineage

When comparing Base, SFT, preference-trained, or RL descendants of the same
architecture:

- use one checkpoint-independent raw serialization for the causal comparison;
- verify exact input token IDs for every request before comparing routes;
- keep matched-token prefill/teacher-forced evidence separate from native
  chat-template and free-running decode evidence;
- compare conditional gain over marginal popularity so changed skew does not
  masquerade as changed predictability;
- use endpoint Base-versus-final first, and add intermediate checkpoints only
  if the frozen endpoint gate reveals an effect worth localizing.

C0 is the reference implementation of this pattern. Its endpoint gate failed,
so SFT and DPO were not added.

## Result and insight tracking

When a second model is active:

- make `STATUS.md` a model-by-hypothesis matrix;
- include model ID and revision in every `EXPERIMENT_LOG.md` entry;
- keep one result document per model and hypothesis;
- reserve a separate cross-model synthesis for matched comparisons;
- label durable insights as `single-testbed`, `replicated`,
  `architecture-dependent`, or `contradicted`;
- update `FOUNDATIONAL_INSIGHTS.md` only when the comparison changes a durable
  claim, not merely because another run completed.

The main README should summarize the primary testbed and link to the comparison
document. It should not accumulate interleaved per-model result prose.

## Efficient expansion sequence

1. Qualify the model and hooks.
2. Run H1 and H2 on the standard-small workload.
3. Review whether the new routing geometry changes the existing conclusion.
4. Run H3 only if H2 leaves a meaningful learned-predictor question; prefer
   selected source-target horizons before an all-pairs scan.
5. Run H4–H6 only after their expert-size, layer-count, expert-count, top-k,
   and plotting assumptions have been adapted for that checkpoint.
6. Produce the per-model result first, then one concise cross-model comparison.

This ordering extracts the main transfer signal before paying for projected
features, many predictor heads, or checkpoint-specific hardware calibration.

## Completion checklist

A new testbed is ready for substantive comparison only when:

- model revision and environment are pinned;
- router selection integrity passes;
- run and document names cannot collide with another model;
- candidate counts expose both \(K/k\) and \(K/E\), while any residency study
  independently exposes \(R/E\);
- model-specific constants are removed or consciously recalibrated in every
  experiment being run;
- status and conclusions remain explicitly scoped to the checkpoint;
- derived tables, reports, and figures pass `audit-artifacts`.
