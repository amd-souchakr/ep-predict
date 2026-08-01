from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ep_predict.tracing.storage import write_json


class ExtendedHorizonVisualizationTest(unittest.TestCase):
    def test_plots_cover_all_valid_source_target_pairs(self) -> None:
        try:
            from ep_predict.visualize.extended_horizon import (
                plot_extended_horizon,
            )
        except ImportError:
            self.skipTest("visualization dependencies are not installed")

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            analysis = run / "analysis" / "h23_extended_horizon"
            analysis.mkdir(parents=True)
            summaries = []
            metrics = []
            for delta in range(1, 16):
                for policy, selection, complete in (
                    ("transition", 0.75, 0.20),
                    ("linear", 0.78, 0.25),
                ):
                    summaries.append(
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
                for source in range(16 - delta):
                    for domain in ("code", "conversation", "general", "math"):
                        for policy, selection, complete in (
                            ("transition", 0.75, 0.20),
                            ("linear", 0.78, 0.25),
                        ):
                            metrics.append(
                                {
                                    "phase": "decode",
                                    "domain": domain,
                                    "source_layer": source,
                                    "target_layer": source + delta,
                                    "delta": delta,
                                    "capacity": 16,
                                    "baseline": policy,
                                    "selection_coverage": selection,
                                    "complete_token_coverage": complete,
                                }
                            )
            for path, rows in (
                (analysis / "summary.csv", summaries),
                (analysis / "metrics.csv", metrics),
            ):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write_json(analysis / "gate.json", {"decision": "unchanged"})
            manifest = plot_extended_horizon(
                run,
                {"analysis": {"output_name": "h23_extended_horizon"}},
            )
            self.assertEqual(len(manifest["outputs"]), 5)
            self.assertTrue(
                (
                    analysis
                    / "figures"
                    / "fig2_source_target_gain_heatmap.pdf"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
