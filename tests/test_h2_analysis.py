from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ep_predict.analysis.h2 import _transition_candidates, analyze_h2
from ep_predict.tracing.schema import TraceRecord
from ep_predict.tracing.storage import RequestTraceStore, write_json


class H2MetricTests(unittest.TestCase):
    def test_transition_candidates_use_source_condition(self) -> None:
        from collections import Counter

        rows = {
            0: Counter({3: 9, 2: 1}),
            1: Counter({2: 8, 3: 2}),
        }
        candidates = _transition_candidates(
            (0,),
            rows=rows,
            marginal=Counter({2: 10, 3: 10}),
            capacity=1,
            num_experts=4,
        )
        self.assertEqual(candidates, (3,))


class H2IntegrationTest(unittest.TestCase):
    def test_synthetic_transition_passes_gate_without_token_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            store = RequestTraceStore(run_dir)
            for request_id in range(6):
                records: list[TraceRecord] = []
                for position in range(12):
                    route = (request_id + position) % 2
                    for layer in range(4):
                        expert = route if layer % 2 == 0 else 1 - route
                        records.append(
                            TraceRecord(
                                run_id="synthetic",
                                request_id=request_id,
                                sample_id=f"sample-{request_id}",
                                phase="decode",
                                token_position=position,
                                input_token_id=position,
                                layer_id=layer,
                                moe_layer_index=layer,
                                selected_expert_ids=[expert],
                                selected_expert_weights=[1.0],
                                batch_id=position,
                                batch_size=1,
                                dataset_name="synthetic",
                                domain="general",
                            )
                        )
                store.write_request(request_id, f"sample-{request_id}", records)
            write_json(run_dir / "run_manifest.json", {"run_id": "synthetic"})
            write_json(
                run_dir / "model_report.json",
                {
                    "routers": [
                        {"layer_id": layer, "num_experts": 4, "top_k": 1}
                        for layer in range(4)
                    ]
                },
            )
            config = {
                "split_seed": 3,
                "test_requests_per_domain": 2,
                "capacities": [1, 2],
                "lookaheads": [1, 2],
                "previous_window_tokens": 8,
                "decision_gate": {
                    "phase": "decode",
                    "baseline": "transition",
                    "comparator": "static",
                    "capacity_experts": 1,
                    "min_mean_selection_coverage_gain": 0.2,
                    "min_mean_complete_token_coverage_gain": 0.2,
                    "min_positive_scope_fraction": 0.5,
                    "min_positive_domains": 1,
                },
            }
            result = analyze_h2(run_dir, config)
            self.assertEqual(result["gate"]["decision"], "PILOT_SUPPORT")
            self.assertEqual(result["train_requests"], 4)
            self.assertEqual(result["test_requests"], 2)
            with (run_dir / "analysis" / "h2" / "metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(
                {row["domain"] for row in rows},
                {"general"},
            )


if __name__ == "__main__":
    unittest.main()
