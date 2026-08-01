from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ep_predict.analysis.codesign import analyze_codesign_map
from ep_predict.tracing.storage import write_json


class CodesignMapTest(unittest.TestCase):
    def test_combines_physics_and_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h4 = root / "h4"
            h4.mkdir()
            physical = [
                {
                    "capacity": 16,
                    "lookahead": 1,
                    "bandwidth_scale": 1.0,
                    "eligible_waves": 10,
                    "cold_demand_experts": 20,
                    "effective_inter_moe_layer_ms": 2.0,
                    "expert_transfer_ms": 0.5,
                    "deadline_feasible_cold_fraction": 0.6,
                    "oracle_stall_reduction": 0.7,
                }
            ]
            predictions = [
                {
                    "phase": "decode",
                    "domain": "__domain_balanced__",
                    "baseline": "transition",
                    "capacity": 16,
                    "delta": 1,
                    "mean_selection_coverage": 0.8,
                    "mean_complete_token_coverage": 0.6,
                }
            ]
            prediction_path = root / "prediction.csv"
            for path, rows in (
                (h4 / "oracle_metrics.csv", physical),
                (prediction_path, predictions),
            ):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            output = root / "codesign"
            config = {
                "output_dir": str(h4),
                "simulation": {
                    "capacities": [16],
                    "lookaheads": [1],
                },
                "decision_gate": {
                    "bandwidth_scale": 1.0,
                    "min_deadline_feasible_cold_fraction": 0.5,
                    "min_oracle_stall_reduction": 0.5,
                },
                "codesign_map": {
                    "prediction_summary": str(prediction_path),
                    "output_dir": str(output),
                    "policies": ["transition"],
                    "min_complete_route_coverage": 0.5,
                    "physical_headroom_ratio": 1.0,
                },
            }
            summary = analyze_codesign_map(config)
            self.assertEqual(
                summary["category_counts"]["candidate_codesign_region"], 1
            )
            with (output / "codesign_points.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["cold_service_headroom_ratio"]), 2.0)


if __name__ == "__main__":
    unittest.main()
