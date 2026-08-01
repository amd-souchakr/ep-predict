# C0 protocol: post-training and routing trajectories

**Status:** preregistered before Base checkpoint collection
**Checkpoints:** `allenai/OLMoE-1B-7B-0125` and its
`allenai/OLMoE-1B-7B-0125-Instruct` descendant
**Scope:** matched-token prefill only; Base versus final Instruct endpoint

## Decisive question

Does post-training materially change long-horizon conditional routing
predictability when Base and Instruct process exactly the same input tokens?

The primary metric is transition-table selection-coverage gain over static
target-layer popularity. This subtracts the easier prediction that can arise
merely because a checkpoint has more skewed marginal expert demand.

## Primary gate

Use held-out prefill requests, source layer 0, target layer 15, and K=16.
Support a post-training predictability effect only when:

1. the absolute Instruct-minus-Base change in transition-over-static selection
   gain is at least 5 percentage points; and
2. at least three of four domains have the direction of the aggregate change.

This gate detects either increased or decreased predictability. It does not
require post-training to improve predictability.

## Workload and split

- Reuse the exact 128 `h1-standard-small` source prompts: 32 each from
  WikiText-2, GSM8K, HumanEval, and MT-Bench.
- Force checkpoint-independent raw prompt serialization.
- Truncate both checkpoints at 384 tokens.
- Generate only one token so collection records the externally supplied
  prefill trajectory without paying for divergent free-running decode.
- Require identical request IDs, sample IDs, token positions, input token IDs,
  routing geometry, and top-k across checkpoints.
- Reuse the deterministic 96/32 H2 request split, with 24/8 requests per
  domain. Fit all popularity and transition tables on the 96 train requests.

The previous Instruct H1 trace is not a matched input because its tokenizer
chat template was applied automatically. Both endpoint traces are therefore
collected again under the frozen raw serialization.

## Analyses

### Formal

- Static and transition prediction at K=8/16/32.
- Fixed source layer 0 through every horizon Δ=1…15.
- Selection coverage, complete top-8 coverage, and gain over static.
- Apply the frozen layer-0→15 gate before interpreting other results.

### Descriptive

- Matched per-token expert-selection agreement, route Jaccard, and exact top-8
  route equality by layer and domain.
- Per-layer expert-popularity Jensen–Shannon divergence and K=16 hot-set
  overlap.
- Cross-checkpoint policy transfer: learn transition tables on Base and
  evaluate Instruct, and vice versa, using the same held-out requests.
- Separate per-checkpoint H1 skew and H2 trajectory reports.

The cross-transfer analysis assumes expert IDs retain their identity across
the direct fine-tuning lineage. It measures policy portability, not language
model quality.

## Stop rule and interpretation

- If the primary gate fails, do not download SFT or DPO. Conclude that
  post-training may remap routes but did not materially change the frozen
  long-range conditional-predictability metric.
- If it passes, human review decides whether the effect is large and
  domain-consistent enough to justify adding SFT and DPO to locate its stage.
- Do not collect hidden states, train H3 predictors, replay H4–H6, or make
  hardware-benefit claims in C0.
- This experiment cannot prove that explicit predictability regularization
  preserves language-model quality.

## Primary figures

1. Base versus Instruct layer-0 prediction quality through Δ=15.
2. Matched Base–Instruct expert-selection agreement through network depth.

Both figures use held-out requests and are saved as PDF and 450-DPI PNG with
hashed inputs under the comparison run.
