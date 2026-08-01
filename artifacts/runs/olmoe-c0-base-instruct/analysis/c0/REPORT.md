# C0 Base–Instruct trajectory comparison

## Formal decision

**PILOT_DOES_NOT_SUPPORT_POSTTRAINING_PREDICTABILITY_EFFECT**.

At the preregistered prefill layer-0→15, K=16 gate, the change in transition-over-static selection gain is +1.6 percentage points. 3/4 domains have the aggregate direction.

## Matched-token trajectory change

Across held-out domains and layers, Base and Instruct select 89.7% of the same experts on average; the full top-8 sets are identical for 35.7% of token-layer events.

## Long-range prediction

| Checkpoint | Transition selection | Gain over static | Complete top-8 |
|---|---:|---:|---:|
| Base | 50.7% | +11.0 pp | 5.2% |
| Instruct | 50.9% | +12.6 pp | 4.9% |

A transition table learned on the other checkpoint loses 1.6 selection-coverage points on average at the same primary layer pair. This transfer result tests policy portability, not language-model quality.

## Scope

The primary comparison uses identical externally supplied tokens, the same 96/32 held-out request split, and no hidden-state predictor. Free-running decode, SFT/DPO localization, hardware replay, and model-quality claims are outside this pilot.
