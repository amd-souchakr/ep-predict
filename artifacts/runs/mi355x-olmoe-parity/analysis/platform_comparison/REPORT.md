# MI355X OLMoE derived-platform comparison

**Evidence grade:** descriptive nested-workload comparison  
**Raw-trace interchangeability:** `NOT_ESTABLISHED_NO_NVIDIA_RAW_TRACE`

## Integrity

- MI355X requests: 16.
- MI355X routed records: 26,752.
- Model geometry matches the preserved NVIDIA report: True.
- Router validation mismatches: 0.
- Requests with bad router-call counts: 0.
- Input-token hashes recorded: 16/16.

## H1 derived trends

- Mean top-8 coverage: 22.37% MI355X versus 25.17% NVIDIA.
- Layerwise top-8 coverage Pearson/Spearman: 0.839/0.841.
- Mean absolute top-8 coverage difference: 2.80 points.
- Mean top-8 hot-set overlap: 4.94/8 experts; Jaccard 0.456.
- Layerwise Gini Pearson/Spearman: 0.947/0.959.

## H2 derived trends

- Horizon-wise transition selection-gain Pearson/Spearman: 0.999/1.000.
- Mean absolute selection-gain difference: 0.86 points.
- Horizon-wise complete-route-gain Pearson/Spearman: 0.996/1.000.
- MI355X descriptive passing horizons: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14].
- NVIDIA descriptive passing horizons: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].

## Interpretation boundary

These artifacts can show whether layer and horizon trends are qualitatively
stable. They cannot attribute differences to hardware, establish per-record
route agreement, or authorize raw-trace interchangeability. The MI355X H2
split contains one held-out request per domain and is a tracer bullet rather
than a confirmation experiment.
