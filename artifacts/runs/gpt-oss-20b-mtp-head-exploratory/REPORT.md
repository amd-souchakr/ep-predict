# GPT-OSS 20B MTP-style route-head exploratory result

**Evidence role:** adaptive architecture/learning-curve development; this does
not replace the failed frozen shared-MLP result or the separate fresh
confirmation.

The original shared 64-unit MLP failed because one additive-context bottleneck
had to represent 276 different source-target transition maps. Replacing that
bottleneck with one linear head per source-target pair changes the result
decisively.

Four request-held-out folds were formed only from the original 96 fitting
requests. At the largest 72-request training point, averaged over folds and
delta 1--3:

| Input | Selection | Complete route |
|---|---:|---:|
| Weighted route | 87.2% | 63.1% |
| Binary route | 89.9% | 70.1% |
| Weighted + binary route | 90.5% | 71.7% |
| Transition table | 84.6% | 57.9% |

The weighted+binary representation was selected before evaluating the existing
32-request development set. Refitting it on all 96 requests produced 91.9%,
91.6%, and 90.5% selection coverage and 74.8%, 74.4%, and 71.8%
complete-route coverage at decode K=8 and delta 1/2/3. It passed all three
counterfactual Milestone F gate rows and exceeded transition selection by
5.7--6.1 points.

The learning curve does not support the claim that 96 requests left this
route-only model severely undertrained. With 8 training requests (128 unique
decode tokens), weighted+binary heads already average 85.2% selection and
61.2% complete-route coverage. The fixed 100-epoch schedule crosses minibatch
boundaries between 32 and 48 requests, so the exact slope should not be read as
a pure sample-complexity law. The qualitative architecture ranking is stable
across all five sizes.

The selected decode model contains 276 heads, 574,080 FP32 parameters (2.19
MiB), and 2,048 multiply-accumulates per forecast. Restricting deployment to
delta 1--3 would require only 66 heads and 137,280 parameters.

This model sees only the current executed route's expert IDs and dispatch
weights. It proves that existing GPT-OSS routing contains strong actionable
cross-layer structure; it does not test a predictor attached to hidden states
or joint base-model training.
