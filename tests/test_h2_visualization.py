from __future__ import annotations

import unittest

from ep_predict.visualize.h2 import _lookahead_series, _summary_lookup


class H2VisualizationTest(unittest.TestCase):
    def test_summary_lookup_preserves_phase_and_lookahead(self) -> None:
        rows = [
            {
                "phase": "decode",
                "domain": "__domain_balanced__",
                "delta": "2",
                "capacity": "16",
                "baseline": "transition",
                "mean_selection_coverage": "0.75",
            }
        ]
        lookup = _summary_lookup(rows)
        self.assertEqual(
            lookup[
                ("decode", "__domain_balanced__", 2, 16, "transition")
            ]["mean_selection_coverage"],
            "0.75",
        )

    def test_lookahead_series_orders_future_layers(self) -> None:
        rows = []
        for delta, static, transition in (
            (1, 0.40, 0.79),
            (2, 0.41, 0.78),
            (3, 0.42, 0.77),
        ):
            for baseline, value in (
                ("static", static),
                ("transition", transition),
            ):
                rows.append(
                    {
                        "phase": "decode",
                        "domain": "__domain_balanced__",
                        "delta": str(delta),
                        "capacity": "16",
                        "baseline": baseline,
                        "mean_selection_coverage": str(value),
                    }
                )
        series = _lookahead_series(
            rows,
            metric="mean_selection_coverage",
            capacity=16,
            lookaheads=[1, 2, 3],
        )
        self.assertEqual(series["transition"], [0.79, 0.78, 0.77])


if __name__ == "__main__":
    unittest.main()
