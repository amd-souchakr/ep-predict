# C0 Base–Instruct trajectory findings

## Plain-language conclusion

Post-training did **not** materially change how predictable OLMoE's routing
trajectory is. It mostly preserved the Base model's expert-routing structure
while making small local changes to which experts appear in each top-8 set.

The formal C0 gate is:

> **PILOT_DOES_NOT_SUPPORT_POSTTRAINING_PREDICTABILITY_EFFECT**

For this OLMoE lineage, the structured trajectory appears to be largely a
pretrained computational scaffold rather than something created by SFT, DPO,
or RLVR.

## Scope and integrity

- Base revision: `9b0c1aa87e34a20052389dce1f0cf01da783f654`.
- Instruct revision: `caada7d7b70f4b852b14108479e0812223a8794f`.
- Both expose the same 16-layer, 64-expert, top-8 routing geometry and exact
  12 MiB BF16 experts.
- Reused the 128 balanced WikiText-2, GSM8K, HumanEval, and MT-Bench prompts.
- Forced raw checkpoint-independent serialization; no chat template.
- Verified identical input token IDs for all 13,918 tokens and 222,688
  token-layer records across both checkpoints.
- Preserved the same 96/32 request split, with 24/8 per domain.
- All popularity and transition tables were fit on train requests only.
- Held-out route comparison covered 55,520 token-layer events.
- Self-transition metrics were reproduced inside the cross-checkpoint
  evaluator with maximum absolute difference 0.0.
- No hidden states, learned sidecar, model modification, inference-library
  modification, free-running decode comparison, or hardware replay.

## Formal result

The preregistered gate used prefill, source layer 0, target layer 15, and K=16.
It compared transition-table selection gain over static popularity so that
changes in marginal skew could not masquerade as conditional predictability.

| Layer 0→15, K=16 | Base | Instruct | Change |
|---|---:|---:|---:|
| Transition selection coverage | 50.7% | 50.9% | +0.2 pp |
| Static selection coverage | 39.8% | 38.4% | −1.4 pp |
| Transition gain over static | +11.0 pp | +12.6 pp | **+1.6 pp** |
| Complete top-8 coverage | 5.2% | 4.9% | −0.4 pp |

The required absolute conditional-gain change was 5 points across at least
three domains. The observed aggregate change was only +1.6 points. Code
increased by +5.0 points, conversation by +1.0, general prose by +1.4, and
mathematics decreased by −1.0. The endpoint effect is neither large nor
domain-general.

## Routing changed less than expected

Across held-out domains and all 16 layers:

- 89.7% of selected expert IDs are shared between Base and Instruct;
- mean route-set Jaccard is 82.6%;
- 35.7% of complete top-8 sets are exactly identical;
- K=16 per-layer hot-set Jaccard is 84.6%;
- expert-popularity JS divergence is only 0.0029 nats.

Selection agreement stays between 85.0% and 91.6% across depth. The largest
change occurs at the final layer, but even there 85% of selections remain the
same. General prose is most stable at 90.3%; mathematics is lowest at 89.2%.

The apparent tension between 89.7% marginal agreement and 35.7% exact-set
agreement is combinatorial: replacing one of eight experts preserves 87.5% of
selections but makes the complete set different. Post-training makes frequent
small edits, not a wholesale rerouting of computation.

## Skew is also stable

The all-layer mixed-domain prefill top-8 popularity coverage changes only from
24.8% for Base to 25.2% for Instruct. Mean Gini changes from 0.256 to 0.264.
Both checkpoints fail the original global H1 stability gate in all 16 layers.

Domain structure is preserved:

- code remains strongly skewed: 46.3% Base versus 46.6% Instruct top-8
  popularity coverage;
- math remains intermediate: 30.9% versus 30.8%;
- conversation remains weak: 22.6% for both.

Post-training did not create a new universal static-placement opportunity.

## Prediction and policy portability

The fixed-source horizon curves nearly overlap. At K=16:

| Horizon | Base gain over static | Instruct gain over static |
|---:|---:|---:|
| Δ=1 | +20.8 pp | +21.4 pp |
| Δ=3 | +17.5 pp | +16.4 pp |
| Δ=6 | +13.1 pp | +13.3 pp |
| Δ=9 | +21.5 pp | +21.1 pp |
| Δ=12 | +16.4 pp | +15.6 pp |
| Δ=15 | +11.0 pp | +12.6 pp |

The non-monotonic curve reproduces the earlier conclusion that target-layer
identity matters more than distance alone.

Transition tables also transfer well between checkpoints. Averaged over all
eligible layer/domain scopes at K=16, cross-checkpoint selection penalties are
generally below one percentage point through Δ=12. At the primary layer-0→15
pair, the Base table loses 3.3 points on Instruct, while the Instruct table is
0.04 points better than the Base self-table on Base. The symmetric mean
penalty is 1.6 points.

The asymmetry suggests that post-training mostly refines a shared transition
graph. The final table is somewhat more portable backward than the Base table
is forward, but this is a single-family pilot rather than evidence for
universal checkpoint portability.

## Interpretation

### Directly supported

1. OLMoE's Base checkpoint already contains strong short- and long-range
   routing structure.
2. SFT+DPO+RLVR preserve most expert identities, hot sets, and conditional
   predictability under identical inputs.
3. Post-training changes complete route sets much more often than marginal
   agreement suggests, usually through one or a few expert substitutions.
4. Existing routing-transition policies are largely reusable across the two
   endpoints.

### Inference

The evidence is consistent with expert specialization and trajectory geometry
being established mainly during pretraining. Post-training appears to adjust
the path locally rather than invent a new computational decomposition.

This makes trajectory predictability look more like a stable property of the
pretrained architecture/checkpoint family than an accidental artifact of
instruction tuning.

### Not established

- that post-training never changes routing predictability in other families;
- which of SFT, DPO, or RLVR caused the small code-specific change;
- that stable routing implies stable language-model outputs or quality;
- that explicit predictability training would preserve loss or load balance;
- that these transition tables are useful for H6-style temporal residency;
- that the result transfers to top-1/top-2 MoE architectures.

## Decision and recommendation

Do not download or run the SFT and DPO checkpoints now. The endpoint gate
failed, so stage localization would add cost without changing the current
research decision.

The high-value follow-up remains one newer, more sparsely routed MoE checkpoint
when explicitly approved. C0 strengthens the OLMoE trajectory claim across a
training lineage, but it does not broaden it across architectures.

## Figures and evidence

Canonical evidence is under
`artifacts/runs/olmoe-c0-base-instruct/analysis/c0/`:

- `fig1_base_instruct_predictability`: fixed layer-0 prediction through every
  future layer;
- `fig2_matched_route_agreement`: expert-ID agreement through depth;
- `matched_route_overlap.csv`;
- `predictability_by_horizon.csv`;
- `cross_checkpoint_transfer.csv`;
- `REPORT.md`, `gate.json`, and `summary.json`.

The plots were programmatically inspected for axes, units, aggregation, and
headline-value consistency. Researcher visual review and the decision not to
add SFT/DPO remain the human checkpoint.
