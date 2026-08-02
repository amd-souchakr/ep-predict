# MI355X OLMoE derived-platform comparison

**Evidence grade:** descriptive matched workload platform comparison  
**Raw-trace interchangeability:** `NOT_ESTABLISHED_NO_NVIDIA_RAW_TRACE`

## Integrity

- MI355X requests: 128.
- MI355X routed records: 222,688.
- Model geometry matches the preserved NVIDIA report: True.
- Router validation mismatches: 0.
- Requests with bad router-call counts: 0.
- Input-token hashes recorded: 128/128.
- Matched request keys, prompt hash, collection settings, and H2 split: True.
- Comparable NVIDIA/MI355X input hashes: 126; mismatches: 0.

## H1 derived trends

- Mean top-8 coverage: 25.18% MI355X versus 25.17% NVIDIA.
- Layerwise top-8 coverage Pearson/Spearman: 0.999962/1.000000.
- Mean absolute top-8 coverage difference: 0.013 points.
- Mean top-8 hot-set overlap: 7.875/8 experts; Jaccard 0.9722.
- Layerwise Gini Pearson/Spearman: 0.999978/1.000000.

## H2 derived trends

- Horizon-wise transition selection-gain Pearson/Spearman: 0.999989/1.000000.
- Mean absolute selection-gain difference: 0.040 points.
- Horizon-wise complete-route-gain Pearson/Spearman: 0.999922/1.000000.
- MI355X descriptive passing horizons: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].
- NVIDIA descriptive passing horizons: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].

## Interpretation boundary

These artifacts compare matched requests and analysis splits when the scope
integrity above is true. They still cannot establish per-record route
agreement or authorize raw-trace interchangeability because the NVIDIA raw
records are unavailable. Small remaining aggregate differences are consistent
with platform-dependent numerical routing changes, but cannot localize them.
