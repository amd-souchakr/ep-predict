from __future__ import annotations

import unittest

import numpy as np
import torch

from ep_predict.analysis.gpt_oss_learned import (
    SharedRouteMLP,
    _coverage_arrays,
    development_gate,
    layer_pairs,
    model_accounting,
)


class GptOssLearnedTest(unittest.TestCase):
    def test_shared_model_has_frozen_parameter_and_operation_counts(self) -> None:
        model = SharedRouteMLP(
            layers=24,
            experts=32,
            source_embedding_width=8,
            target_embedding_width=8,
            phase_embedding_width=4,
            hidden_width=64,
        )
        config = {
            "model": {
                "route_width": 32,
                "source_layer_embedding_width": 8,
                "target_layer_embedding_width": 8,
                "phase_embedding_width": 4,
                "hidden_width": 64,
                "output_width": 32,
            }
        }
        accounting = model_accounting(model, config)
        self.assertEqual(accounting["parameters"], 5864)
        self.assertEqual(accounting["serialized_parameter_bytes"], 23456)
        self.assertEqual(accounting["multiply_accumulates_per_forecast"], 5376)
        logits = model(
            torch.zeros(3, 32),
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 1, 0]),
        )
        self.assertEqual(tuple(logits.shape), (3, 32))

    def test_layer_pairs_cover_every_forward_pair_once(self) -> None:
        source, target = layer_pairs(24, list(range(1, 24)))
        self.assertEqual(len(source), 276)
        self.assertTrue(torch.all(target > source))
        self.assertEqual(len(set(zip(source.tolist(), target.tolist()))), 276)

    def test_coverage_arrays_keep_mass_aligned_with_dispatch_order(self) -> None:
        ranks = np.asarray([[7, 5, 2, 0]], dtype=np.uint8)
        target_ids = np.asarray([[5, 7, 1, 3]])
        weights = np.asarray([[0.1, 0.6, 0.2, 0.1]])
        _, selections, masses, completes = _coverage_arrays(
            ranks, target_ids, weights, 2
        )
        self.assertEqual(selections[0], 2)
        self.assertAlmostEqual(masses[0], 0.7)
        self.assertEqual(completes[0], 0)

    def test_development_gate_applies_absolute_and_noninferiority_rules(self) -> None:
        rows = []
        for domain_index, domain in enumerate(("a", "b", "c", "d")):
            for request_index in range(2):
                request_id = domain_index * 2 + request_index
                for delta in (1, 2, 3):
                    for baseline, selection, complete in (
                        ("domain_static", 0.65, 0.25),
                        ("source_copy", 0.70, 0.30),
                        ("transition", 0.86, 0.58),
                        ("learned", 0.83, 0.53),
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
            "evaluation": {
                "primary_phase": "decode",
                "primary_candidate_count": 8,
                "primary_lookaheads": [1, 2, 3],
                "bootstrap_resamples": 20,
                "bootstrap_seed": 3,
                "confidence_level": 0.95,
            },
            "decision_gate": {
                "min_selection_coverage": 0.82,
                "min_complete_route_coverage": 0.50,
                "max_selection_deficit_vs_transition": 0.03,
                "max_complete_route_deficit_vs_transition": 0.05,
                "min_selection_gain_vs_cheap_comparator": 0.10,
                "require_positive_selection_gain_all_domains": True,
                "min_passing_lookaheads": 2,
            },
        }
        gate_rows, decision = development_gate(rows, config)
        self.assertTrue(all(row["pass"] for row in gate_rows))
        self.assertEqual(decision["decision"], "DEVELOPMENT_PASS")


if __name__ == "__main__":
    unittest.main()
