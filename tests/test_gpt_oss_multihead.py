from __future__ import annotations

import unittest

import torch

from ep_predict.analysis.gpt_oss_multihead import PairwiseRouteHeads


class GptOssMultiheadTest(unittest.TestCase):
    def test_pairwise_heads_have_independent_linear_maps(self) -> None:
        model = PairwiseRouteHeads(pairs=3, feature_width=32, experts=32)
        self.assertEqual(sum(value.numel() for value in model.parameters()), 3168)
        output = model(torch.zeros(5, 3, 32))
        self.assertEqual(tuple(output.shape), (5, 3, 32))

    def test_one_phase_all_pair_parameter_count(self) -> None:
        weighted = PairwiseRouteHeads(pairs=276, feature_width=32)
        combined = PairwiseRouteHeads(pairs=276, feature_width=64)
        self.assertEqual(sum(value.numel() for value in weighted.parameters()), 291456)
        self.assertEqual(sum(value.numel() for value in combined.parameters()), 574080)


if __name__ == "__main__":
    unittest.main()
