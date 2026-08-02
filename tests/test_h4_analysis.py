from __future__ import annotations

import unittest
from collections import OrderedDict

from ep_predict.analysis.h4 import (
    DemandWave,
    _calibration_comparison_rows,
    _cold_sets,
    _simulate,
)


class H4AnalysisTest(unittest.TestCase):
    def test_calibration_comparison_reports_ratio_and_change(self) -> None:
        reference = {
            "decode": {
                "median_forward_ms": 10.0,
                "effective_inter_moe_layer_ms": 0.625,
            },
            "transfer": {
                "exact_expert_median_ms": 0.5,
                "fit": {"effective_bandwidth_gbps": 24.0},
            },
        }
        current = {
            "decode": {
                "median_forward_ms": 5.0,
                "effective_inter_moe_layer_ms": 0.3125,
            },
            "transfer": {
                "exact_expert_median_ms": 0.25,
                "fit": {"effective_bandwidth_gbps": 48.0},
            },
        }
        rows = _calibration_comparison_rows(reference, current)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["current_over_reference"], 0.5)
        self.assertEqual(rows[0]["percent_change"], -50.0)
        self.assertEqual(rows[-1]["current_over_reference"], 2.0)

    def test_lru_capacity_reuse_and_cold_accounting(self) -> None:
        waves = [
            DemandWave(0, 0, "x", 0, (0, 1)),
            DemandWave(1, 0, "x", 0, (0, 2)),
            DemandWave(2, 0, "x", 0, (0, 1)),
        ]
        cold, compulsory, demanded, misses, compulsory_misses = _cold_sets(
            waves, capacity=2
        )
        self.assertEqual(demanded, 6)
        self.assertEqual(misses, 4)
        self.assertEqual(compulsory_misses, 3)
        self.assertEqual(cold[(0, 0)], (0, 1))
        self.assertEqual(compulsory[(0, 0)], (0, 1))
        self.assertEqual(cold[(1, 0)], (2,))
        self.assertEqual(cold[(2, 0)], (1,))

    def test_oracle_hides_transfer_with_sufficient_slack(self) -> None:
        waves = [
            DemandWave(0, 0, "x", 0, (0,)),
            DemandWave(0, 0, "x", 1, (1,)),
        ]
        metrics = _simulate(
            waves=waves,
            token_count=1,
            cold={(0, 0): (0,), (0, 1): (1,)},
            compulsory={(0, 0): (0,), (0, 1): (1,)},
            all_cold_misses=2,
            all_compulsory_misses=2,
            demanded_experts=2,
            layers=2,
            delta=1,
            layer_ms=2.0,
            transfer_ms=1.0,
            expert_bytes=12,
        )
        self.assertEqual(metrics["deadline_feasible_cold_fraction"], 1.0)
        self.assertEqual(metrics["oracle_stall_ms"], 0.0)
        self.assertEqual(metrics["oracle_stall_reduction"], 1.0)

    def test_serial_copies_create_residual_wave_stall(self) -> None:
        waves = [
            DemandWave(0, 0, "x", 0, (0, 1)),
            DemandWave(0, 0, "x", 1, (2, 3)),
        ]
        metrics = _simulate(
            waves=waves,
            token_count=1,
            cold={(0, 0): (0, 1), (0, 1): (2, 3)},
            compulsory={(0, 0): (0, 1), (0, 1): (2, 3)},
            all_cold_misses=4,
            all_compulsory_misses=4,
            demanded_experts=4,
            layers=2,
            delta=1,
            layer_ms=1.0,
            transfer_ms=1.0,
            expert_bytes=12,
        )
        self.assertEqual(metrics["deadline_feasible_cold_fraction"], 0.5)
        self.assertEqual(metrics["oracle_stall_ms"], 1.0)
        self.assertEqual(metrics["reactive_stall_ms"], 2.0)
        self.assertEqual(metrics["oracle_stall_reduction"], 0.5)


if __name__ == "__main__":
    unittest.main()
