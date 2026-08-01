# H4 figure review

## Automated headline

Formal decision: `PILOT_DOES_NOT_SUPPORT`. The best frozen K=16, measured-bandwidth short horizon is Δ=3: 32.8% deadline-feasible cold bytes and 38.9% oracle stall reduction.

## Human review checklist

- [x] Heatmap axes, bandwidth multipliers, capacities, and 12 MiB semantics are correct.
- [x] The K=16, Δ=1–3 cells agree with `gate.json`.
- [x] Deadline-feasible bytes are not conflated with resident hit bytes.
- [x] Capacity, bandwidth, and lookahead boundaries are recorded.
- [x] The first-order timing and single-copy-engine assumptions are accepted as sufficient for analytical window finding, not a latency claim.
- [x] Next action: H5-A/H5-B first-order profitability sweep and inverse predictor requirements.

**Review completed:** 2026-08-01. The formal K=16 short-horizon failure
stands; the broader capacity–bandwidth–lookahead region motivates analytical
co-design exploration rather than additional timing fidelity.
