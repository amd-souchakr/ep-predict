# H4 oracle feasibility result

**Decision:** `PILOT_SUPPORTS`

## Calibration

- Hook-free cached-token forward median: 15.638 ms.
- Effective inter-MoE-layer budget: 0.977 ms.
- Exact 12 MiB pinned-host transfer median: 0.285 ms.
- Fitted effective bandwidth: 44.69 GB/s.
- Decode P10--P90: 15.251--15.858 ms.
- Exact-copy P10--P90: 0.284--0.286 ms.

## Direct calibration comparison

| Quantity | RTX 3090 Ti | MI355X | MI355X / RTX |
|---|---:|---:|---:|
| cached_token_forward_median_ms | 10.228976 | 15.638086 | 1.529× |
| effective_inter_moe_layer_ms | 0.639311 | 0.977380 | 1.529× |
| exact_12mib_copy_median_ms | 0.524096 | 0.284944 | 0.544× |
| fitted_effective_bandwidth_gbps | 24.135426 | 44.688902 | 1.852× |

## Descriptive calibration-factor attribution

The MI355X demand trace is held fixed while measured testbed layer time and copy time are substituted independently. This post-hoc factorial view does not assign the forward-time difference to inherent hardware and does not alter the frozen gate.

| Scenario | Δ | Copies/layer | On-time cold bytes | Stall reduction | Pass |
|---|---:|---:|---:|---:|:---:|
| reference_calibration | 1 | 1.22 | 26.6% | 31.7% | no |
| reference_calibration | 2 | 1.22 | 29.8% | 35.8% | no |
| reference_calibration | 3 | 1.22 | 32.8% | 38.9% | no |
| mi_copy_only | 1 | 2.24 | 50.0% | 54.6% | no |
| mi_copy_only | 2 | 2.24 | 56.3% | 61.4% | yes |
| mi_copy_only | 3 | 2.24 | 61.4% | 65.7% | yes |
| measured_testbed_slack_only | 1 | 1.86 | 26.6% | 46.8% | no |
| measured_testbed_slack_only | 2 | 1.86 | 36.0% | 52.7% | no |
| measured_testbed_slack_only | 3 | 1.86 | 43.0% | 56.7% | no |
| measured_testbed_combined | 1 | 3.43 | 68.8% | 74.8% | yes |
| measured_testbed_combined | 2 | 3.43 | 78.7% | 82.7% | yes |
| measured_testbed_combined | 3 | 3.43 | 83.9% | 86.5% | yes |

## Frozen primary gate

| Δ | Deadline-feasible cold bytes | Oracle stall reduction |
|---:|---:|---:|
| 1 | 68.8% | 74.8% |
| 2 | 78.7% | 82.7% |
| 3 | 83.9% | 86.5% |

The gate requires both metrics to reach 50% for at least one short horizon at measured bandwidth and K=16.

## Descriptive feasibility boundary

| K | Δ | Bandwidth | Resident hits | On-time cold bytes | Stall reduction |
|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 1× | 39.9% | 72.0% | 76.0% |
| 16 | 3 | 1× | 56.8% | 83.9% | 86.5% |
| 32 | 3 | 1× | 79.8% | 95.9% | 96.7% |
| 16 | 9 | 1× | 53.3% | 99.9% | 100.0% |
| 16 | 1 | 2× | 55.5% | 97.1% | 98.9% |

The broader scan quantifies the capacity--lead-time--bandwidth boundary. Its descriptive cells do not retroactively change the formal short-horizon gate.

## Interpretation boundary

This is a trace-driven, single-copy-engine oracle calculation. It establishes a calibrated feasibility region, not end-to-end speedup or overlap correctness in the live model.
The measured forward interval is testbed-stack slack for this regime exploration; cross-platform differences are not assigned to inherent GPU compute capability.
