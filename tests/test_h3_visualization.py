from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ep_predict.tracing.storage import write_json


class H3VisualizationTest(unittest.TestCase):
    def test_plot_h3_writes_two_figures_and_manifest(self) -> None:
        try:
            from ep_predict.visualize.h3 import plot_h3
        except ImportError:
            self.skipTest("visualization dependencies are not installed")

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            analysis = run / "analysis" / "h3"
            analysis.mkdir(parents=True)
            summary_rows = []
            metric_rows = []
            for delta in (1, 2, 3):
                for policy, selection, complete in (
                    ("transition", 0.78, 0.24),
                    ("linear", 0.79, 0.28),
                ):
                    summary_rows.append(
                        {
                            "phase": "decode",
                            "domain": "__domain_balanced__",
                            "delta": delta,
                            "capacity": 16,
                            "baseline": policy,
                            "mean_selection_coverage": selection,
                            "mean_complete_token_coverage": complete,
                        }
                    )
                for domain in ("code", "conversation", "general", "math"):
                    for layer in range(15):
                        for policy, selection, complete in (
                            ("transition", 0.78, 0.24),
                            ("linear", 0.79, 0.28),
                        ):
                            metric_rows.append(
                                {
                                    "phase": "decode",
                                    "domain": domain,
                                    "source_layer": layer,
                                    "delta": delta,
                                    "capacity": 16,
                                    "baseline": policy,
                                    "selection_coverage": selection,
                                    "complete_token_coverage": complete,
                                }
                            )
            for path, rows in (
                (analysis / "summary.csv", summary_rows),
                (analysis / "metrics.csv", metric_rows),
            ):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write_json(
                analysis / "gate.json",
                {
                    "decision": "PILOT_DOES_NOT_SUPPORT",
                    "mean_selection_coverage_gain": 0.01,
                    "mean_complete_token_coverage_gain": 0.04,
                },
            )
            manifest = plot_h3(run, {})
            self.assertEqual(len(manifest["outputs"]), 5)
            self.assertTrue(
                (analysis / "figures" / "fig1_h3_lookahead_comparison.pdf").is_file()
            )
            self.assertFalse(manifest["human_review_complete"])


if __name__ == "__main__":
    unittest.main()
