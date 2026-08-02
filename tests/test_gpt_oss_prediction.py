from __future__ import annotations

import unittest

from ep_predict.analysis.gpt_oss_prediction import (
    CoverageAccumulator,
    WeightedRoute,
    _bootstrap_gate,
    _copy_candidates,
    _transition_candidates,
)


class GptOssPredictionTest(unittest.TestCase):
    def test_weighted_coverage_separates_set_mass_and_complete_metrics(self) -> None:
        accumulator = CoverageAccumulator(candidate_count=4, top_k=2)
        accumulator.add((0, 2, 3, 4), WeightedRoute((0, 1), (0.8, 0.2)))
        metrics = accumulator.metrics()
        self.assertEqual(metrics["selection_coverage"], 0.5)
        self.assertAlmostEqual(metrics["routed_mass_coverage"], 0.8)
        self.assertEqual(metrics["complete_route_coverage"], 0.0)
        self.assertEqual(metrics["candidate_set_fraction"], 0.125)
        self.assertNotIn("resident_fraction", metrics)

    def test_copy_and_transition_candidates_have_fixed_candidate_count(self) -> None:
        marginal = {0: 10, 1: 8, 2: 5, 3: 1}
        copied = _copy_candidates((3, 1), marginal, candidate_count=3, num_experts=4)
        self.assertEqual(copied, (3, 1, 0))
        self.assertEqual(
            _copy_candidates((3, 1), marginal, candidate_count=2, num_experts=4),
            (3, 1),
        )
        transitioned = _transition_candidates(
            (0,), {0: {2: 8, 3: 2}}, marginal, candidate_count=2, num_experts=4
        )
        self.assertEqual(transitioned, (2, 3))

    def test_bootstrap_gate_uses_requests_and_strongest_baseline(self) -> None:
        rows = []
        for domain_index, domain in enumerate(("a", "b", "c", "d")):
            for request_index in range(3):
                request_id = domain_index * 3 + request_index
                for delta in (1, 2, 3):
                    for baseline, selection, complete in (
                        ("domain_static", 0.50, 0.30),
                        ("source_copy", 0.55, 0.32),
                        ("transition", 0.65, 0.42),
                    ):
                        rows.append(
                            {
                                "request_id": request_id,
                                "domain": domain,
                                "phase": "decode",
                                "delta": delta,
                                "candidate_count": 8,
                                "baseline": baseline,
                                "selection_coverage": selection,
                                "routed_mass_coverage": selection + 0.05,
                                "complete_route_coverage": complete,
                            }
                        )
        config = {
            "analysis": {
                "primary_phase": "decode",
                "primary_capacity": 8,
                "primary_lookaheads": [1, 2, 3],
                "bootstrap_resamples": 50,
                "bootstrap_seed": 7,
                "confidence_level": 0.95,
            },
            "decision_gate": {
                "min_selection_coverage_gain": 0.03,
                "require_positive_selection_ci": True,
                "min_complete_route_coverage_gain": 0.02,
                "require_nonnegative_complete_ci": True,
                "min_positive_domains": 3,
                "min_passing_lookaheads": 2,
            },
        }
        gate_rows, decision = _bootstrap_gate(rows, config)
        self.assertTrue(
            all(row["selection_comparator"] == "source_copy" for row in gate_rows)
        )
        self.assertTrue(all(row["pass"] for row in gate_rows))
        self.assertEqual(decision["decision"], "PILOT_SUPPORTS_20B_ROUTE_PREDICTION")


if __name__ == "__main__":
    unittest.main()
