# GPT-OSS MTP-style future-route heads

**Development:** passed 3/3 on the existing 32-request development set

**Confirmation:** `CONFIRMATION_PASS`, 3/3 on 64 fresh requests

**Checkpoint:** `openai/gpt-oss-20b` revision
`6cee5e81ee83917806bbde320786a8fb61efebee`

**Scope:** route-only expert-demand prediction, not latency or language quality

## What changed

The failed Milestone F model concatenated the current weighted route with
source-layer, target-layer, and phase embeddings, then forced every layer pair
through one 64-unit hidden layer. That architecture had only 5,864 parameters
to represent 276 distinct source-target conditional maps. It learned
well-calibrated marginal popularity and failed at ranking.

The revised model follows the multi-token-prediction analogy more literally:
every source-target layer pair selects its own linear 32-logit prediction head.
The selected input concatenates:

- a 32-dimensional dispatch-weighted current route; and
- a 32-dimensional binary current-route indicator.

The binary channel matters because the strong transition table treats each
selected source expert equally, while the weighted channel retains routed-mass
information. Their combination lets a linear head represent both effects.

The all-horizon decode model has 276 heads, 574,080 FP32 parameters (2.19 MiB),
and 2,048 multiply-accumulates per forecast. An actionable delta 1--3-only
implementation needs 66 heads, 137,280 parameters (0.52 MiB FP32), and the same
per-forecast cost. This is roughly 98 times more parameters than the failed
MLP, but still negligible beside a 20B-parameter model and cheaper per forecast
than the MLP's 5,376 MACs.

## Architecture selection without development leakage

The original 96 Milestone F fitting requests were partitioned into four
request-held-out folds, six validation requests per domain per fold. Weighted,
binary, and weighted+binary heads were compared at 8, 16, 32, 48, and 72
training requests. The existing 32-request development set did not participate
in this selection.

At 72 requests, averaged across folds and delta 1--3:

| Predictor | Decode tokens | Selection coverage | Complete-route coverage |
|---|---:|---:|---:|
| Weighted route heads | 1,152 | 87.2% ± 0.7 pp | 63.1% ± 1.4 pp |
| Binary route heads | 1,152 | 89.9% ± 0.6 pp | 70.1% ± 1.3 pp |
| Weighted + binary heads | 1,152 | 90.5% ± 0.6 pp | 71.7% ± 1.2 pp |
| Transition table | 1,152 | 84.6% ± 0.7 pp | 57.9% ± 1.2 pp |

The weighted+binary model was selected, refit on all 96 original fitting
requests, and evaluated on the existing 32-request development set:

| Δ | Head selection | Transition | Head complete | Transition complete | Gate-equivalent |
|---:|---:|---:|---:|---:|:---:|
| 1 | 91.9% | 86.3% | 74.8% | 60.7% | pass |
| 2 | 91.6% | 85.5% | 74.4% | 59.4% | pass |
| 3 | 90.5% | 84.4% | 71.8% | 56.8% | pass |

This development result was adaptive: the architecture was motivated by the
failed shared MLP. The weights and pipeline were therefore frozen before a new
confirmation workload was observed.

## Fresh confirmation

The confirmation workload contains 64 previously unused requests, 16 per
domain. It was constructed from the same revision-pinned datasets by extending
the deterministic per-domain pool from 32 to 48 and excluding every prior
sample ID and prompt hash. Both overlap counts are zero.

The frozen checkpoint was evaluated without refitting:

| Δ | Head selection | Transition | Difference (95% paired-request CI) | Head complete | Transition complete | Pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 91.7% | 86.0% | +5.7 pp [+5.5, +6.0] | 74.1% | 60.0% | yes |
| 2 | 91.1% | 85.4% | +5.8 pp [+5.5, +6.1] | 73.3% | 59.6% | yes |
| 3 | 90.0% | 84.1% | +5.9 pp [+5.7, +6.2] | 70.8% | 56.8% | yes |

All absolute, transition-noninferiority, cheap-baseline-gain, and four-domain
breadth checks pass at all three lookaheads. The head beats the strongest cheap
comparator by 23.1--25.5 selection points and transition by 13.7--14.1
complete-route points. The result is stable rather than a threshold-edge pass.

The confirmed learned frontier for Milestone G is:

| K | Candidate amplification | Selection Δ=1 / 2 / 3 | Complete route Δ=1 / 2 / 3 |
|---:|---:|---:|---:|
| 4 | 1× | 76.7% / 76.0% / 74.1% | 36.7% / 36.2% / 34.9% |
| 8 | 2× | 91.7% / 91.1% / 90.0% | 74.1% / 73.3% / 70.8% |
| 12 | 3× | 95.8% / 95.5% / 94.9% | 85.9% / 85.1% / 83.2% |
| 16 | 4× | 97.8% / 97.6% / 97.3% | 92.1% / 91.5% / 90.3% |

Fresh-request probability calibration is also coherent: mean predicted
positive probability is 0.129 against the exact 0.125 label frequency,
ten-bin ECE is 1.44--1.53 pp, and Brier score is 0.0416--0.0461 across the
three primary lookaheads. Unlike the failed shared MLP, low calibration error
now accompanies strong top-k discrimination.

Fresh trace integrity also passes: 64/64 requests, 1,024 generated tokens,
309,408 token-layer records, 1,237,632 dispatch pairs, zero ID mismatches, zero
weight mismatches, and complete coverage.

## What the sample-size result actually says

Calling the original run “96 samples” obscures the unit structure:

- 96 requests produced 18,126 training tokens overall;
- only 1,536 were decode tokens;
- expanding each token over 276 layer pairs created 5,002,776 examples, but
  those pairs are not independent observations;
- each pair-specific decode head nevertheless sees all 1,536 unique decode
  tokens.

For this 64-dimensional route-only problem, more data was not the primary
missing ingredient. Weighted+binary heads already reach 85.2% selection and
61.2% complete-route coverage with 128 decode tokens in the four-fold study.
The learning-curve schedule uses 100 full epochs and crosses minibatch-count
boundaries, so its slope is not a clean asymptotic sample-complexity estimate.
It is still enough to reject the claim that the 96-request dataset inherently
precluded a useful route-only predictor.

The original model failed because it imposed the wrong parameter sharing. The
per-pair heads succeed because layer identity chooses a conditional map rather
than entering additively through a narrow shared bottleneck.

## What a future gate-attached predictor should look like

For a production model, predicting from the current executed route is useful
but conservative: it becomes available only after the current gate. A more
MTP-like design should attach to the normalized router input or a small shared
projection of the layer hidden state:

1. Compute a shared 128-dimensional predictor state from the current layer's
   2,880-dimensional hidden state.
2. Select separate `(source layer, delta)` heads for delta 1--3, each emitting
   32 future-expert logits.
3. Train against future complete top-4 sets, optionally distilling future
   router probabilities as a secondary target.
4. Optimize ranking/coverage directly in addition to BCE; report probability
   calibration separately because marginal calibration did not imply useful
   top-k ranking.
5. If trained jointly with the base model, track language loss, expert load
   balance, specialization, and routing stability. Stop-gradient predictor
   training and joint routing-regularized training are different claims.

For delta 1--3, a direct hidden-to-head design would use about 6.1 million
weights (roughly 12 MiB BF16). A shared 2,880→128 projection plus 66
128→32 heads uses about 641 thousand weights, roughly 1.22 MiB BF16. The latter
is the better first frozen-model experiment.

## Data required for the hidden-state experiment

The route-head result cannot determine hidden-state sample complexity. A
practical collection should target at least 100,000 unique decode tokens,
balanced across domains and held out by request, then plot learning curves at
10k, 30k, 100k, 300k, and 1M tokens. One hundred thousand raw BF16 hidden-state
tokens across all 24 layers and width 2,880 require about 12.9 GiB; storing a
frozen 128-dimensional projection requires about 0.57 GiB. Online projection
or online detached-head training is therefore preferable.

This 100k target is an engineering starting point, not a statistically proven
minimum. The stopping rule should be empirical: proceed to confirmation when
the lower request-bootstrap bound and all-domain breadth plateau above the
same K=8 gate for two successive learning-curve sizes.

Future joint pretraining would expose the heads to orders of magnitude more
tokens, so predictor optimization is unlikely to be data-limited. The open
scientific question is whether adding that objective preserves model quality,
specialization, and load balance—not whether cross-layer routing contains a
predictive signal. The fresh confirmation now answers the latter positively.

## Artifacts

- [Exploratory learning curve and checkpoint](../artifacts/runs/gpt-oss-20b-mtp-head-exploratory/)
- [Fresh confirmation report and traces](../artifacts/runs/gpt-oss-20b-mtp-head-confirmation/)
- [Frozen exploratory configuration](../configs/experiment/gpt-oss-20b-mtp-head-exploratory.toml)
- [Frozen confirmation configuration](../configs/experiment/gpt-oss-20b-mtp-head-confirmation.toml)
- [Single-token lookahead visual aids](GPT_OSS_LOOKAHEAD_VISUALS.md)
- [Post-hoc Δ=1–23 horizon trade-off](GPT_OSS_LONG_HORIZON_RESULTS.md)
