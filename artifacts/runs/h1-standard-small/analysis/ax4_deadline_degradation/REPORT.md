# AX4 deadline-bounded graceful expert degradation

**State:** complete; pending human figure review

## Plain-language result

AX4 passes the preregistered architecture gate under the selected trace-ordered FCFS candidates. Hard commit makes transfer latency irrelevant after the deadline; the price is missing routed mass, whose language-quality meaning remains unvalidated.

## Weight-semantics finding

OLMoE does **not** renormalize its selected top-8 weights. The router softmaxes over all 64 experts, selects eight, and the expert block uses those probabilities directly. AX4 therefore uses normalized-within-top-8 mass only as the primary architecture contract and preserves absolute missing router probability as a secondary integrity metric.

Across decode waves, the mean raw selected-weight sum is 0.406; P99 is 0.717.

## Frozen latency prediction

The measured all-local anchor is 10.23 ms. A fixed 10% commit/fallback allowance gives 11.25 ms and 88.9 token/s. Against the 66.83 ms K=16 reactive projection, this is 83.2% lower TPOT and 5.94× throughput.

This bound is conditional on zero waiting after commit, reserved fallback compute, bounded local work, and isolation of speculative traffic.

## Selected FCFS candidates

| K | Δ | C | A | order | slack | BW | P99 missing mass | fallback waves | gate |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | 99.0% | 1.0× | mass_adversarial | 0.50 | 24.1 GB/s | 100.0% | 6.76% | fail |
| 8 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 24.1 GB/s | 100.0% | 5.93% | fail |
| 8 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 64.0 GB/s | 66.3% | 0.00% | fail |
| 8 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 128.0 GB/s | 27.2% | 0.00% | fail |
| 8 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 256.0 GB/s | 0.0% | 0.00% | pass |
| 8 | 1 | 99.0% | 1.0× | random_within_route | 0.50 | 24.1 GB/s | 100.0% | 6.76% | fail |
| 16 | 3 | 99.0% | 1.0× | mass_adversarial | 0.50 | 24.1 GB/s | 100.0% | 1.79% | fail |
| 16 | 3 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 24.1 GB/s | 100.0% | 1.52% | fail |
| 16 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 64.0 GB/s | 60.6% | 0.00% | fail |
| 16 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 128.0 GB/s | 20.4% | 0.00% | fail |
| 16 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 256.0 GB/s | 0.0% | 0.00% | pass |
| 16 | 9 | 99.0% | 1.5× | mass_priority_oracle | 0.50 | 24.1 GB/s | 100.0% | 1.56% | fail |
| 16 | 3 | 99.0% | 1.0× | random_within_route | 0.50 | 24.1 GB/s | 100.0% | 1.79% | fail |
| 32 | 3 | 99.0% | 1.0× | mass_adversarial | 0.50 | 24.1 GB/s | 83.8% | 0.16% | fail |
| 32 | 3 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 24.1 GB/s | 81.4% | 0.14% | fail |
| 32 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 64.0 GB/s | 42.9% | 0.00% | fail |
| 32 | 1 | 99.0% | 1.5× | mass_priority_oracle | 1.00 | 128.0 GB/s | 7.2% | 0.00% | pass |
| 32 | 1 | 99.0% | 1.5× | mass_priority_oracle | 0.50 | 256.0 GB/s | 0.0% | 0.00% | fail |
| 32 | 3 | 99.9% | 1.0× | random_within_route | 0.50 | 24.1 GB/s | 83.5% | 0.16% | fail |

## Headline candidate

The best selected FCFS point is K=8, Δ=1, C=99.0%, A=1.5×, mass_priority_oracle. Its P99 normalized missing mass is 0.0% and its full-fallback wave rate is 0.00%.

Null, present-renormalized, and shared-residual execution have the same availability and TPOT in this model. They differ only in the assumed quality response and perturbation bound: mB, 2mB, and mD.

## Interpretation

A gate pass would justify a tightly scoped future training test of availability-conditioned expert erasure. It would not show that current OLMoE maintains perplexity or task quality. A gate failure would mean that even controlled approximation needs either more resident capacity, better mass ranking, more lookahead/bandwidth, or a looser missing-mass contract.

Latency is trace-calibrated and nonblocking by construction. Missing mass is trace-derived under an assumed predictor. Robustness, quality preservation, larger-model routing, and non-measured hardware points remain assumptions.
