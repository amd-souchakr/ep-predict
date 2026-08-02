# H4 figure review

## Automated headline

Formal decision: `PILOT_SUPPORTS`. With 16 experts kept on GPU per layer and the measured copy speed, the best tested advance notice is 3 layers: 83.9% of required cold data arrives on time and 86.5% of transfer waiting is removed.

## Human review checklist

- [x] The grid reads as copy speed × advance notice × experts kept on GPU without requiring H4 terminology.
- [x] The black-outlined 16-expert, measured-speed, 1–3 layer cells agree with `gate.json`.
- [x] The two-panel chart keeps on-time data distinct from waiting removed.
- [x] Every capacity and lookahead remains visible in both measured-link trends.
- [x] The effective-average layer-time approximation and single-copy-engine limitation are accepted for regime-space exploration.
- [x] The testbed forward time is not presented as an inherent MI355X hardware characteristic.

**Review completed:** 2026-08-01. The researcher accepted the regime-space
result with a software/testbed interpretation for the forward-time gap and
requested the final plain-language figure redesign retained here. Milestone C
router/dispatch qualification is the next gated experiment.
