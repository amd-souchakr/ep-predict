# C0 Base–Instruct figure review

## Automated headline

Formal decision: **PILOT_DOES_NOT_SUPPORT_POSTTRAINING_PREDICTABILITY_EFFECT**. At layer 0→15 and K=16, the Instruct-minus-Base change in conditional selection gain is +1.6 percentage points. Matched Base/Instruct routes retain 89.7% of selected expert IDs on average across depth.

Figure 1 holds the source layer fixed at layer 0, avoiding the changing-source composition problem in a global horizon mean. Figure 2 uses only held-out requests with exactly matching input token IDs.

## Human visual-review checkpoint

- [x] Programmatic check: exact token matching and the 96/32 request split pass.
- [x] Programmatic/visual check: conditional gain is distinguished from raw coverage/skew.
- [x] Visual check: complete top-8 coverage is not described as hardware benefit.
- [x] Visual check: domain and layer heterogeneity in Figure 2 is visible and recorded in `docs/C0_RESULTS.md`.
- [ ] The Base–Instruct endpoint result justifies or rejects adding SFT/DPO.
