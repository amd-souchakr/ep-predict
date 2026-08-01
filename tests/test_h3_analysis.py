from __future__ import annotations

import unittest

import numpy as np

from ep_predict.analysis.h3 import _evaluate_gate, _fit_linear_head


class H3AnalysisTest(unittest.TestCase):
    def test_linear_head_learns_separable_multilabel_scores(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(256, 6)).astype(np.float32)
        y = np.zeros((256, 4), dtype=np.float32)
        y[:, 0] = x[:, 0] > 0
        y[:, 1] = x[:, 0] <= 0
        y[:, 2] = x[:, 1] > 0
        y[:, 3] = x[:, 1] <= 0
        head = _fit_linear_head(
            x,
            y,
            phase="decode",
            source_layer=0,
            delta=1,
            config={
                "device": "cpu",
                "training_seed": 23,
                "learning_rate": 0.05,
                "weight_decay": 0.0001,
                "positive_class_weight": 1.0,
                "epochs": 30,
                "batch_size": 64,
            },
        )
        predictions = head.scores(x).argmax(axis=1)
        expected_first_pair = np.where(x[:, 0] > 0, 0, 1)
        self.assertGreater(np.mean(predictions == expected_first_pair), 0.45)

    def test_gate_requires_both_metrics_across_scopes(self) -> None:
        rows = []
        for domain in ("a", "b", "c", "d"):
            for layer in range(4):
                for baseline, selection, complete in (
                    ("transition", 0.70, 0.20),
                    ("linear", 0.75, 0.24),
                ):
                    rows.append(
                        {
                            "phase": "decode",
                            "domain": domain,
                            "source_layer": layer,
                            "delta": 1,
                            "capacity": 16,
                            "baseline": baseline,
                            "selection_coverage": selection,
                            "complete_token_coverage": complete,
                        }
                    )
        gate = _evaluate_gate(
            rows,
            {
                "phase": "decode",
                "lookahead": 1,
                "capacity_experts": 16,
                "baseline": "linear",
                "comparator": "transition",
                "min_mean_selection_coverage_gain": 0.03,
                "min_mean_complete_token_coverage_gain": 0.02,
                "min_positive_selection_scope_fraction": 0.75,
                "min_positive_complete_scope_fraction": 0.75,
                "min_positive_domains": 3,
            },
        )
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["positive_domains_both_metrics"], 4)


if __name__ == "__main__":
    unittest.main()
