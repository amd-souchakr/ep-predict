from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ep_predict.tracing.storage import write_json


class H4VisualizationTest(unittest.TestCase):
    def test_two_h4_figures_are_generated(self) -> None:
        try:
            from ep_predict.visualize.h4 import plot_h4
        except ImportError:
            self.skipTest("visualization dependencies are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis" / "h4"
            analysis.mkdir(parents=True)
            rows = []
            for capacity in (8, 16, 32):
                for bandwidth in (0.5, 1.0, 2.0):
                    for delta in (1, 2, 3):
                        rows.append(
                            {
                                "capacity": capacity,
                                "lookahead": delta,
                                "bandwidth_scale": bandwidth,
                                "oracle_stall_reduction": 0.6,
                            }
                        )
            with (analysis / "oracle_metrics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(
                analysis / "gate.json",
                {
                    "decision": "PILOT_SUPPORTS",
                    "best_primary_row": {
                        "lookahead": 2,
                        "deadline_feasible_cold_fraction": 0.7,
                        "oracle_stall_reduction": 0.6,
                    },
                },
            )
            write_json(analysis / "measurement.json", {"state": "complete"})
            config = {
                "output_dir": str(analysis),
                "simulation": {
                    "capacities": [8, 16, 32],
                    "lookaheads": [1, 2, 3],
                    "bandwidth_scales": [0.5, 1.0, 2.0],
                },
            }
            manifest = plot_h4(root, config)
            self.assertEqual(len(manifest["outputs"]), 5)
            self.assertTrue(
                (
                    analysis
                    / "figures"
                    / "fig1_h4_oracle_feasibility_heatmap.pdf"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
