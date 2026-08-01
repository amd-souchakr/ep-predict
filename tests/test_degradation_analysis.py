from __future__ import annotations

import unittest

import numpy as np

from ep_predict.analysis.degradation import (
    WeightedRoutes,
    _candidate_profile,
    _lru_cold_mask,
    _simulate_local_deadline,
)


def _routes() -> WeightedRoutes:
    raw = np.asarray(
        [
            [0.40, 0.10],
            [0.30, 0.20],
            [0.35, 0.15],
            [0.25, 0.25],
        ],
        dtype=np.float64,
    )
    return WeightedRoutes(
        token_count=2,
        layers=2,
        top_k=2,
        request_ids=np.asarray([0, 0, 0, 0]),
        domains=np.asarray(["code", "code", "code", "code"]),
        token_indices=np.asarray([0, 0, 1, 1]),
        layer_ids=np.asarray([0, 1, 0, 1]),
        expert_ids=np.asarray([[0, 1], [2, 3], [0, 1], [2, 3]]),
        raw_weights=raw,
        normalized_weights=raw / raw.sum(axis=1, keepdims=True),
    )


class DeadlineDegradationTest(unittest.TestCase):
    def test_lru_cold_mask_and_normalized_mass(self) -> None:
        routes = _routes()
        cold = _lru_cold_mask(routes, 2)
        np.testing.assert_array_equal(
            cold,
            np.asarray(
                [
                    [True, True],
                    [True, True],
                    [False, False],
                    [False, False],
                ]
            ),
        )
        np.testing.assert_allclose(routes.normalized_weights.sum(axis=1), 1.0)
        self.assertTrue(np.all(routes.raw_weights.sum(axis=1) < 1.0))

    def test_importance_order_controls_incomplete_wave_omission(self) -> None:
        routes = _routes()
        cold = np.ones_like(routes.expert_ids, dtype=bool)
        complete = np.asarray([True, False, True, True])
        oracle = _candidate_profile(
            routes,
            cold,
            lookahead=1,
            complete=complete,
            importance_order="mass_priority_oracle",
            seed=17,
        )
        adversarial = _candidate_profile(
            routes,
            cold,
            lookahead=1,
            complete=complete,
            importance_order="mass_adversarial",
            seed=17,
        )
        # Wave 1 has normalized weights [0.6, 0.4].
        self.assertTrue(oracle.predicted_mask[1, 0])
        self.assertFalse(oracle.predicted_mask[1, 1])
        self.assertFalse(adversarial.predicted_mask[1, 0])
        self.assertTrue(adversarial.predicted_mask[1, 1])

    def test_hard_deadline_never_charges_post_commit_wait(self) -> None:
        routes = _routes()
        cold = _lru_cold_mask(routes, 2)
        profile = _candidate_profile(
            routes,
            cold,
            lookahead=1,
            complete=np.ones(4, dtype=bool),
            importance_order="mass_priority_oracle",
            seed=17,
        )
        metrics, missing, _raw = _simulate_local_deadline(
            routes=routes,
            profile=profile,
            amplification=1.0,
            importance_order="mass_priority_oracle",
            lookahead=1,
            slack_intervals=0.0,
            layer_ms=1.0,
            transfer_ms=0.4,
            concurrency=1,
            expert_bytes=12 * 1024**2,
            seed=17,
            renormalization_floor=0.05,
        )
        self.assertTrue(metrics["zero_post_commit_transfer_wait"])
        self.assertEqual(metrics["transfer_induced_post_commit_stall_ms"], 0.0)
        self.assertEqual(missing[0], 1.0)
        self.assertEqual(missing[2], 0.0)


if __name__ == "__main__":
    unittest.main()
