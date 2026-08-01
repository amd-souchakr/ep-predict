# H5 first-order result

**Decision:** `PILOT_SUPPORTS_DESIGN_REGION_BUT_NOT_EXISTING_POLICY`

A controlled analytical profitability region exists, but none of the eight
unchanged transition/linear policy placements passes the frozen screen.

At K=32, the policies cover 67–81% of complete cold sets but transfer
6.3–6.7 candidate bytes per useful cold byte. The long-range K=32, Δ=9
policies have enough first-order headroom and coverage; speculative traffic is
their only frozen-gate failure.

This redirects the next experiment toward prediction-guided admission and
residency, not predictor expansion. See `docs/H5_RESULTS.md` for the full
interpretation and limitations.
