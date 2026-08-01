from __future__ import annotations

import unittest

from ep_predict.analysis.h5 import _controlled_sweep


class H5AnalysisTest(unittest.TestCase):
    def test_first_order_gate_and_inverse_requirement(self) -> None:
        physical = [
            {
                "capacity": 16,
                "lookahead": 3,
                "bandwidth_scale": 1.0,
                "eligible_waves": 10,
                "cold_demand_experts": 20,
                "mean_cold_experts_per_wave": 2.0,
                "available_lead_time_ms": 3.0,
                "measured_expert_transfer_ms": 1.0,
                "cold_service_headroom": 1.0,
                "cold_service_pressure": 1.0,
                "first_order_oracle_stall_reduction": 1.0,
            }
        ]
        gate = {
            "min_modeled_stall_reduction": 0.25,
            "min_oracle_recovery": 0.50,
            "max_predicted_to_useful_bytes": 2.0,
        }
        design, windows, inverse = _controlled_sweep(
            physical_rows=physical,
            coverage_step=0.5,
            amplifications=[1.0, 2.0, 4.0],
            gate=gate,
            expert_bytes=12,
        )
        lookup = {
            (
                row["complete_cold_set_coverage"],
                row["candidate_transfer_amplification"],
            ): row
            for row in design
        }
        self.assertTrue(lookup[(0.5, 1.0)]["profitable"])
        self.assertTrue(lookup[(0.5, 2.0)]["profitable"])
        self.assertFalse(lookup[(1.0, 4.0)]["profitable"])
        inverse_lookup = {
            row["candidate_transfer_amplification"]: row for row in inverse
        }
        self.assertEqual(
            inverse_lookup[1.0]["minimum_complete_cold_set_coverage"], 0.5
        )
        self.assertFalse(inverse_lookup[4.0]["inverse_window_exists"])
        self.assertTrue(any(row["window_exists"] for row in windows))


if __name__ == "__main__":
    unittest.main()
