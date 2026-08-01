from __future__ import annotations

import unittest

import numpy as np

from ep_predict.analysis.admission import (
    _average_tie_auroc,
    _histogram_divergence,
    _threshold_sweep,
)


class AdmissionAnalysisTest(unittest.TestCase):
    def test_perfect_score_separation_has_unit_auroc(self) -> None:
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        labels = np.array([False, False, True, True])
        self.assertEqual(_average_tie_auroc(scores, labels), 1.0)

    def test_threshold_sweep_counts_useful_and_false_candidates(self) -> None:
        scores = np.array([[2.0, 1.0, 0.5, -1.0]], dtype=np.float32)
        cold = np.array([[True, True, False, False]])
        resident = np.array([[False, False, False, True]])
        row = _threshold_sweep(
            scores=scores,
            cold=cold,
            resident=resident,
            thresholds=[0.75],
        )[0]
        self.assertEqual(row["useful_admitted_experts"], 2)
        self.assertEqual(row["false_admitted_experts"], 0)
        self.assertEqual(row["complete_cold_set_coverage"], 1.0)
        self.assertEqual(row["candidate_transfer_amplification"], 1.0)

    def test_histogram_divergence_distinguishes_overlap_and_information(
        self,
    ) -> None:
        identical = _histogram_divergence(
            np.array([5, 10, 5]),
            np.array([50, 100, 50]),
        )
        self.assertAlmostEqual(identical["score_js_divergence_bits"], 0.0)
        self.assertAlmostEqual(identical["score_distribution_overlap"], 1.0)
        self.assertAlmostEqual(
            identical["score_label_mutual_information_bits"], 0.0
        )

        separated = _histogram_divergence(
            np.array([10, 0]),
            np.array([0, 10]),
        )
        self.assertAlmostEqual(separated["score_js_divergence_bits"], 1.0)
        self.assertAlmostEqual(separated["score_distribution_overlap"], 0.0)
        self.assertAlmostEqual(
            separated["score_label_mutual_information_bits"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
