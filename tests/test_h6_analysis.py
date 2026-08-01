from __future__ import annotations

import unittest

import numpy as np

from ep_predict.analysis.h6 import _dynamic_replay


class H6AnalysisTest(unittest.TestCase):
    def test_oracle_retains_nearer_reuse_under_equal_budget(self) -> None:
        demands = [(0,), (1,), (0,), (1,)]
        common = {
            "demands": demands,
            "initial_residents": (0,),
            "capacity": 1,
            "movement_budget": 1,
            "expert_bytes": 12,
        }
        lru = _dynamic_replay(policy="lru", **common)
        oracle = _dynamic_replay(policy="oracle", **common)
        self.assertEqual(lru["residual_cold_expert_demand"], 3)
        self.assertEqual(oracle["residual_cold_expert_demand"], 2)
        self.assertLess(
            oracle["runtime_movement_experts"],
            lru["runtime_movement_experts"],
        )

    def test_prediction_never_loads_an_undemanded_candidate(self) -> None:
        scores = np.asarray(
            [
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=np.float32,
        )
        result = _dynamic_replay(
            demands=[(0,), (0,)],
            initial_residents=(0,),
            capacity=1,
            movement_budget=1,
            expert_bytes=12,
            policy="transition",
            score_vectors=scores,
            initial_belief=np.zeros(3, dtype=np.float32),
        )
        self.assertEqual(result["runtime_movement_experts"], 0)
        self.assertEqual(result["residual_cold_expert_demand"], 0)


if __name__ == "__main__":
    unittest.main()
