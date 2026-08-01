from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ep_predict.analysis.h1 import (
    distribution_metrics,
    gini,
    window_stability,
)
from ep_predict.analysis.h1 import analyze_h1
from ep_predict.tracing.schema import TraceRecord
from ep_predict.tracing.storage import RequestTraceStore, write_json


class MetricTests(unittest.TestCase):
    def test_uniform_distribution(self) -> None:
        metrics = distribution_metrics([10, 10, 10, 10], [1, 2])
        self.assertAlmostEqual(metrics["gini"], 0.0)
        self.assertAlmostEqual(metrics["normalized_entropy"], 1.0)
        self.assertAlmostEqual(metrics["top_1_coverage"], 0.25)
        self.assertAlmostEqual(metrics["top_2_coverage"], 0.5)

    def test_maximally_skewed_distribution(self) -> None:
        self.assertAlmostEqual(gini([40, 0, 0, 0]), 0.75)
        metrics = distribution_metrics([40, 0, 0, 0], [1])
        self.assertAlmostEqual(metrics["top_1_coverage"], 1.0)

    def test_stable_windows(self) -> None:
        metrics = window_stability(
            [[0], [0], [0], [1], [0], [0], [0], [1]],
            window_size=4,
            hotset_size=1,
        )
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertAlmostEqual(metrics["mean_jaccard"], 1.0)
        self.assertAlmostEqual(metrics["mean_lagged_oracle_ratio"], 1.0)


class H1IntegrationTest(unittest.TestCase):
    def test_synthetic_trace_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            records = [
                TraceRecord(
                    run_id="synthetic",
                    request_id=0,
                    sample_id="sample",
                    phase="decode",
                    token_position=position,
                    input_token_id=position,
                    layer_id=0,
                    moe_layer_index=0,
                    selected_expert_ids=[0 if position % 4 != 3 else 1],
                    selected_expert_weights=[1.0],
                    batch_id=position,
                    batch_size=1,
                    dataset_name="synthetic",
                    domain="general",
                )
                for position in range(8)
            ]
            RequestTraceStore(run_dir).write_request(0, "sample", records)
            write_json(
                run_dir / "run_manifest.json",
                {"run_id": "synthetic"},
            )
            write_json(
                run_dir / "model_report.json",
                {
                    "routers": [
                        {"layer_id": 0, "num_experts": 4, "top_k": 1}
                    ]
                },
            )
            config = {
                "analysis": {"window_sizes": [4], "top_n": [1, 2]},
                "decision_gate": {
                    "phase": "decode",
                    "capacity_experts": 1,
                    "min_coverage_lift_over_uniform": 2.0,
                    "window_size": 4,
                    "min_hotset_jaccard": 0.5,
                    "min_lagged_oracle_ratio": 0.8,
                    "min_passing_layer_fraction": 0.5,
                },
            }
            result = analyze_h1(run_dir, config)
            self.assertEqual(result["gate"]["decision"], "PILOT_SUPPORT")
            report = run_dir / "analysis" / "h1" / "REPORT.md"
            self.assertTrue(report.is_file())
            self.assertIn("PILOT_SUPPORT", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
