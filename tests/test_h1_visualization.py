from __future__ import annotations

import unittest

from ep_predict.visualize.h1 import _capacity_curves


class H1VisualizationTest(unittest.TestCase):
    def test_capacity_curve_averages_layer_local_ranks(self) -> None:
        rows = []
        for phase in ("prefill", "decode"):
            for domain in ("__all__", "code", "math", "general", "conversation"):
                rows.extend(
                    [
                        {
                            "phase": phase,
                            "domain": domain,
                            "layer_id": "0",
                            "rank": "1",
                            "probability": "0.75",
                        },
                        {
                            "phase": phase,
                            "domain": domain,
                            "layer_id": "0",
                            "rank": "2",
                            "probability": "0.25",
                        },
                        {
                            "phase": phase,
                            "domain": domain,
                            "layer_id": "1",
                            "rank": "1",
                            "probability": "0.50",
                        },
                        {
                            "phase": phase,
                            "domain": domain,
                            "layer_id": "1",
                            "rank": "2",
                            "probability": "0.50",
                        },
                    ]
                )

        curves = _capacity_curves(rows, layers=[0, 1], num_experts=2)

        self.assertAlmostEqual(curves[("prefill", "code")][0], 0.625)
        self.assertAlmostEqual(curves[("prefill", "code")][1], 1.0)


if __name__ == "__main__":
    unittest.main()
