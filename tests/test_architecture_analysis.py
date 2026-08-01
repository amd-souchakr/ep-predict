from __future__ import annotations

import unittest

import numpy as np

from ep_predict.analysis.architecture import (
    MIB,
    _correlated_complete_mask,
    _local_stall_metrics,
    _predicted_counts,
    _transfer_ms,
)


class ArchitectureAnalysisTest(unittest.TestCase):
    def test_correlated_coverage_and_amplification_are_realized(self) -> None:
        cold = np.ones((10, 10), dtype=np.int16)
        complete, coverage = _correlated_complete_mask(
            cold, 0.90, seed=17, block_waves=4
        )
        predicted, amplification = _predicted_counts(cold, complete, 2.0)
        self.assertAlmostEqual(
            coverage["realized_complete_cold_set_coverage"], 0.90
        )
        self.assertGreaterEqual(coverage["max_incomplete_run_waves"], 2)
        self.assertAlmostEqual(
            amplification["realized_predicted_to_useful_amplification"], 2.0
        )
        self.assertEqual(int(predicted.sum()), 2 * int(complete.sum()))

    def test_more_lookahead_hides_an_isolated_transfer(self) -> None:
        cold = np.zeros((2, 4), dtype=np.int16)
        cold[:, 3] = 1
        complete = cold > 0
        predicted = cold.astype(np.int64)
        short = _local_stall_metrics(
            cold_counts=cold,
            complete=complete,
            predicted=predicted,
            lookahead=1,
            layer_ms=1.0,
            transfer_ms=2.0,
            base_tpot_ms=10.0,
        )
        long = _local_stall_metrics(
            cold_counts=cold,
            complete=complete,
            predicted=predicted,
            lookahead=3,
            layer_ms=1.0,
            transfer_ms=2.0,
            base_tpot_ms=10.0,
        )
        self.assertGreater(
            short["p99_predictive_stall_ms"],
            long["p99_predictive_stall_ms"],
        )

    def test_transfer_fit_uses_decimal_gigabytes(self) -> None:
        result = _transfer_ms(12 * MIB, 24.135425594434384, 2.755010047474471)
        self.assertAlmostEqual(result, 0.5240960121154785, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
