# H1 pilot findings

## Outcome

The preregistered mixed-workload decode gate is **not supported**:

- 2 of 16 layers passed, below the required 8 of 16;
- 7 of 16 layers had at least 2x top-8 concentration;
- only layers 6 and 9 also met both stability thresholds.

This is not a complete rejection of H1. The result is strongly
workload-dependent: code and mathematics have concentrated, stable expert
demand, while conversation does not. The correct pilot conclusion is:

> A single model-wide, workload-agnostic top-8 tier is weak for this mixture,
> but workload-conditioned expert placement is promising enough to test H2.

## Run and integrity

- Model: `allenai/OLMoE-1B-7B-0125-Instruct`, pinned commit
  `caada7d7b70f4b852b14108479e0812223a8794f`
- Workload: 128 requests, 32 each from WikiText-2, GSM8K, HumanEval, and
  MT-Bench
- Input token positions per layer: 15,582 prefill and 8,011 decode
- Trace records: 377,488 across 16 MoE layers
- Expert selections: 3,019,904
- Observed routing: top-8 of 64 at every layer
- Hook validation: collection completed with mismatch failures enabled; trace
  integrity found no schema, layer, or expert-ID errors

Two requests were resumed from the successful smoke run. Of the remaining
requests, most generated 64 tokens; early EOS produced shorter valid sequences
for a few requests.

## Skew and stability

The table reports mean top-8 selection coverage across layers. `Skew` is the
number of layers at or above 25% coverage. `Stable` applies the configured
512-token top-8 Jaccard and lagged/oracle thresholds. The formal gate is the
mixed decode row; per-domain prefill is the primary data-conditioned evidence.

| Phase | Scope | Top-8 coverage | Lift over uniform | Skew | Stable | Both |
|---|---|---:|---:|---:|---:|---:|
| Prefill | Mixed | 25.8% | 2.06x | 12/16 | 0/16 | 0/16 |
| Prefill | Code | 45.9% | 3.67x | 16/16 | 16/16 | 16/16 |
| Prefill | Conversation | 23.7% | 1.90x | 2/16 | 0/16 | 0/16 |
| Prefill | General prose | 25.0% | 2.00x | 9/16 | 1/16 | 1/16 |
| Prefill | Mathematics | 28.2% | 2.26x | 14/16 | 16/16 | 14/16 |
| Decode | Mixed | 24.2% | 1.94x | 7/16 | 2/16 | 2/16 |
| Decode | Code | 46.5% | 3.72x | 16/16 | 16/16 | 16/16 |
| Decode | Conversation | 21.8% | 1.74x | 1/16 | 1/16 | 0/16 |
| Decode | General prose | 27.2% | 2.17x | 12/16 | 11/16 | 10/16 |
| Decode | Mathematics | 30.5% | 2.44x | 14/16 | 13/16 | 13/16 |

Domain window counts are small, especially in decode (three complete
512-token windows per domain), so the per-domain stability classifications are
pilot evidence rather than confidence-bound estimates.

## Is routing data-conditioned?

Yes, for this checkpoint and formatting. The between-domain difference is
much larger than same-domain split-half drift:

| Phase | Mean between-domain JSD | Mean within-domain JSD | Ratio | Between-domain hot-set Jaccard | Within-domain hot-set Jaccard |
|---|---:|---:|---:|---:|---:|
| Prefill | 0.0807 | 0.00725 | 11.1x | 0.222 | 0.667 |
| Decode | 0.1142 | 0.00744 | 15.3x | 0.156 | 0.646 |

All 96 layer-matched domain-pair comparisons in each phase had greater JSD
than the average split-half drift of the two participating domains.
Between-domain hot-set overlap was lower in 93/96 prefill comparisons and
96/96 decode comparisons.

This supports a joint **checkpoint x input distribution x formatting x
generation phase** effect. It does not establish that semantic domain alone
causes the routing difference. HumanEval's repeated Python syntax and the
prompt prefixes are legitimate workload properties but possible explanatory
confounds.

## Placement implications

An exploratory oracle comparison uses each domain's own static top-8 set
instead of one mixed-workload static set:

| Phase | Mixed top-8 coverage | Domain top-8 coverage | Gain |
|---|---:|---:|---:|
| Prefill | 25.4% | 30.7% | +5.3 percentage points |
| Decode | 24.2% | 31.5% | +7.3 percentage points |

These are domain-balanced means across layers. General prose has the largest
loss from the mixed set: 14.4% to 25.0% in prefill and 11.0% to 27.2% in
decode. The domain-conditioned set improved 63/64 prefill and 64/64 decode
layer-domain scopes.

Each BF16 expert is 12 MiB. A top-8 tier across all 16 layers is 1.5 GiB,
one eighth of the 12 GiB expert weights. However, OLMoE routes every token to
eight experts:

- a domain-conditioned top-8 tier covers about 31% of expert selections;
- it touches at least one resident expert for about 90% of prefill and 92% of
  decode layer-token records;
- it contains all eight selected experts for only about 0.03% of records.

Therefore, a small tier can reduce slow-memory traffic or replication pressure,
but it cannot usually remove the slow tier from the token's critical path.
Post-hoc capacity points reinforce the distinction: domain-conditioned tiers
of 16 and 32 experts cover about 50% and 75% of selections, but contain all
eight selected experts for only 5.2%/22.4% of prefill and 2.7%/20.5% of decode
records. These capacity results are exploratory, not part of the preregistered
gate.

## Decision and recommendations

1. Record H1 as **mixed; global gate not supported**. Do not claim that one
   static top-8 placement works model-wide.
2. Proceed to H2 without expanding H1 first. Test whether request/context and
   recent routing history can select a compact expert set better than global
   popularity. Include global-static, domain-oracle, lagged-window, and
   current-window-oracle baselines.
3. Evaluate capacities 8, 16, and 32 and report both selection coverage and
   all-selected-experts coverage. Charge tier switches and expert movement;
   coverage alone is not a hardware benefit.
4. Keep the learned component external to the frozen model and inference
   library. The defensible future claim is that a lightweight sidecar can
   anticipate routing demand and manage a hierarchy, not yet that the base
   model itself learned resource management.
5. Defer additional H1 seeds until H2 shows useful conditional gain. Before a
   strong or publication claim, repeat H1 with more subset/order seeds,
   request-bootstrap intervals, and a second MoE checkpoint.

## Figures and human review

Generate the three H1 figures with:

```bash
uv run ep-predict plot-h1 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h1-standard-small.toml
```

The generated `analysis/h1/figures/FIGURES.md` contains captions and the
human-review checklist. H2 should not begin until that visual review is
complete.

Generated outputs:

- `fig1_capacity_coverage`: fast-tier capacity versus selection coverage;
- `fig2_skew_stability`: layer-level skew/stability operating map;
- `fig3_domain_shift`: between-domain divergence versus within-domain drift.

Each is available as a 450-DPI PNG and vector PDF under
`artifacts/runs/h1-standard-small/analysis/h1/figures`.

## Post-run protocol audit

`docs/H1_PROTOCOL.md` describes 128/256-token windows and mentions a
256-token gate, while the frozen experiment config specifies 256/512-token
windows and a 512-token gate. The run and formal decision used the config.
Rechecking the available 256-token decode rows does not rescue H1: no layer
reaches the 0.50 mean-Jaccard threshold at that window. The mismatch should be
fixed when H2 is preregistered, but the H1 text is retained as an audit trail.
