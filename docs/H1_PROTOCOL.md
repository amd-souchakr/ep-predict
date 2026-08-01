# H1 protocol: expert skew and operational hotness

## Purpose

H1 is a workload-characterization gate:

> For at least some layers and workloads, a minority of experts receives a
> disproportionate and sufficiently stable share of demand to justify a
> limited fast-residency tier.

This experiment does **not** test prediction, prefetch speedup, or model
quality. Its job is to determine whether static or slowly adaptive hot-expert
placement deserves further study.

## Why H1 comes first

A single global histogram can manufacture a misleading positive result:
expert namespaces differ by layer, prefill and decode have different routing
regimes, and domain mixtures can hide changes. H1 is evaluated per
`(layer, domain, phase)` and only then summarized.

## Testbed and tracing mechanism

Use `allenai/OLMoE-1B-7B-0125-Instruct` in BF16 on one GPU. It has 16 MoE
layers, 64 experts per layer, and top-8 routing. At BF16, an OLMoE expert is
approximately 12 MiB; the inspection report computes the exact value from the
loaded tensors.

Tracing uses ordinary PyTorch forward pre/post hooks:

1. A root pre-hook captures the current input IDs, attention mask, request
   metadata, and whether the call is prefill or decode.
2. Router hooks capture the router's output tuple.
3. Selected expert IDs come from the router output actually consumed by the
   MoE block.
4. The collector independently recomputes top-k from router logits and treats
   any mismatch as an integrity failure.
5. A root post-hook verifies that every discovered router fired exactly once
   per model forward.

There is no monkey patching, model rewrite, custom model implementation, or
change to Transformers.

## Workloads

The checked-in 20-prompt authored workload is only a hook smoke fixture. The H1
research pilot uses 128 examples: 32 each from revision-pinned WikiText-2,
GSM8K, HumanEval, and MT-Bench splits. See `docs/DATASET_PROTOCOL.md`.

Each request is run separately with greedy generation for 64 decode tokens and
a maximum prompt length of 384 tokens. The standard pilot is large enough for
a spend/no-spend decision, but not a publication claim.

Run order is fixed for the pilot. Before a strong claim:

- repeat with at least three shuffled request orders;
- repeat with two additional deterministic subset seeds;
- add request-level bootstrap intervals;
- confirm that the domain conclusions are not driven by token-count imbalance.

## Metrics

For every layer, domain, and phase:

- expert probability distribution;
- entropy and entropy normalized by `log(num_experts)`;
- Gini coefficient;
- top-1/2/4/8/16 demand coverage;
- maximum-to-median popularity ratio;
- number of routed tokens and selections.

For windows of 128 and 256 routed tokens:

- consecutive hot-set Jaccard similarity;
- static global-hot-set coverage;
- lagged coverage using the previous window's hot set;
- current-window oracle hot-set coverage;
- lagged/oracle coverage ratio.

The last ratio is an operational stability measure: it asks how much of the
best possible coverage a one-window-old placement retains.

To distinguish a data-conditioned effect from sampling noise, also report:

- pairwise Jensen-Shannon divergence between domain expert distributions;
- top-8 hot-set Jaccard between domains;
- the same metrics between split halves of each domain.

Between-domain divergence should be interpreted relative to within-domain
split-half drift, not against zero.

## Preregistered pilot gate

The default fast-tier budget is 8 of 64 experts per layer. A layer is:

- **strongly skewed** when top-8 coverage is at least 2× its 12.5% uniform
  baseline (at least 25%);
- **stable** when mean consecutive top-8 Jaccard is at least 0.50 and the
  previous-window hot set retains at least 80% of current-window oracle
  coverage at the 256-token window.

The pilot supports H1 for a phase only when at least half of the MoE layers
meet both conditions. This threshold is an engineering gate, not a universal
statistical definition.

Interpretation:

| Outcome | Direction |
|---|---|
| Strong + stable | Continue static pinned-tier and replication analysis |
| Strong + unstable | Study short-horizon/adaptive residency; static caching is weak |
| Weak + stable | Skew is insufficient; do not justify a hot tier from popularity |
| Weak + unstable | Reject popularity-based residency for this scope |

Per-domain prefill results are primary for the data-conditioned claim.
Token-weighted mixed-domain results and decode results are secondary.

## Confounds and controls

- Analyze prefill and decode separately.
- Never combine expert IDs from different layers.
- Use greedy decoding and fixed seeds for the pilot.
- Report the exact prompt set and generation parameters in the manifest.
- Exclude padding positions.
- Do not use hooked runs for timing.
- Treat top-k selections, not router probability mass, as demand.
- Report both selection-level coverage and the top-k value. Top-8 routing
  makes full-token residency substantially harder than a top-1 model.
- If quantization is introduced, rerun router validation; a different kernel
  path can invalidate hook assumptions.

## Stop/go decision

After the pilot:

1. If integrity fails, stop and fix tracing.
2. If H1 clearly fails in both phases and all domains, record a negative result
   and skip static hot-tier claims. H2 can still proceed because conditional
   locality may exist without global skew.
3. If H1 is clearly supported, run the confirmation workload before making an
   architectural claim.
4. If mixed, expand only the scopes near the threshold; do not scale every
   experiment indiscriminately.

The next hypothesis will be designed only after this gate is reviewed.
