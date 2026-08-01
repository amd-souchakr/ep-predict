from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ep_predict.tracing.storage import write_json


class H6VisualizationTest(unittest.TestCase):
    def test_h6_heatmap_is_generated(self) -> None:
        try:
            from ep_predict.visualize.h6 import plot_h6
        except ImportError:
            self.skipTest("visualization dependencies are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            analysis = Path(temporary) / "h6"
            analysis.mkdir(parents=True)
            rows = []
            for capacity in (8, 16, 32):
                for policy, coverage in (
                    ("static", 0.20),
                    ("domain", 0.25),
                    ("lru", 0.30),
                    ("transition", 0.35),
                    ("linear", 0.40),
                ):
                    rows.append(
                        {
                            "phase": "decode",
                            "domain": "code",
                            "source_layer": 0,
                            "target_layer": 1,
                            "delta": 1,
                            "capacity": capacity,
                            "policy": policy,
                            "complete_resident_set_hit_coverage": coverage,
                        }
                    )
            for name in ("scope_metrics.csv", "summary.csv"):
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
            self.assertEqual(len(manifest["outputs"]), 4)


if __name__ == "__main__":
    unittest.main()
