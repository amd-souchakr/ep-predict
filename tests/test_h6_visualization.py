from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ep_predict.tracing.storage import write_json


class H6VisualizationTest(unittest.TestCase):
    def test_h6_primary_comparison_is_generated(self) -> None:
        try:
            from ep_predict.visualize.h6 import plot_h6
        except ImportError:
            self.skipTest("visualization dependencies are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            analysis = Path(temporary) / "h6"
            analysis.mkdir(parents=True)
            scope_rows = [
                {
                    "phase": "decode",
                    "domain": "code",
                    "source_layer": 0,
                    "target_layer": 3,
                    "delta": 3,
                    "capacity": 16,
                    "policy": "lru",
                    "complete_resident_set_hit_coverage": 0.03,
                }
            ]
            summary_rows = [
                {
                    "phase": "decode",
                    "domain": "__domain_balanced__",
                    "delta": 3,
                    "capacity": 16,
                    "policy": policy,
                    "mean_residual_cold_expert_fraction": residual,
                    "mean_useful_movement_fraction": useful,
                }
                for policy, residual, useful in (
                    ("oracle", 0.31, 0.95),
                    ("lru", 0.48, 0.70),
                    ("linear", 0.49, 0.66),
                    ("transition", 0.50, 0.60),
                )
            ]
            for name, rows in (
                ("scope_metrics.csv", scope_rows),
                ("summary.csv", summary_rows),
            ):
                with (analysis / name).open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write_json(
                analysis / "gate.json",
                {
                    "decision": "PILOT_SUPPORT",
                    "primary_scope": {
                        "phase": "decode",
                        "capacity_experts": 16,
                        "lookahead": 3,
                    },
                },
            )
            manifest = plot_h6(
                {
                    "output_dir": str(analysis),
                    "replay": {"capacities": [8, 16, 32]},
                }
            )
            self.assertTrue(
                (
                    analysis
                    / "figures"
                    / "fig1_h6_residency_gain_heatmap.pdf"
                ).is_file()
            )
            self.assertEqual(len(manifest["outputs"]), 3)


if __name__ == "__main__":
    unittest.main()
