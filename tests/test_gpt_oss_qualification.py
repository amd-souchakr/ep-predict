from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from ep_predict.gpt_oss_qualification import compare_routes, decode_dispatch_inputs


@dataclass
class FakeRoutingData:
    gate_scal: torch.Tensor
    hist: torch.Tensor


@dataclass
class FakeGather:
    src_indx: torch.Tensor
    dst_indx: torch.Tensor


class GptOssQualificationTest(unittest.TestCase):
    def test_decode_dispatch_inputs_recovers_token_major_pairs(self) -> None:
        # Token-major selected IDs are [[0, 2], [1, 2]]. Dispatch is expert-major.
        routing = FakeRoutingData(
            gate_scal=torch.tensor([0.7, 0.6, 0.3, 0.4]),
            hist=torch.tensor([1, 1, 2]),
        )
        gather = FakeGather(
            src_indx=torch.tensor([0, 2, 1, 3]),
            dst_indx=torch.tensor([0, 2, 1, 3]),
        )
        observation = decode_dispatch_inputs(routing, gather, num_tokens=2, top_k=2)
        self.assertEqual(observation.expert_ids.tolist(), [[0, 2], [1, 2]])
        self.assertTrue(torch.allclose(observation.weights, torch.tensor([[0.7, 0.3], [0.6, 0.4]])))

    def test_compare_routes_preserves_id_weight_pairs_across_ordering(self) -> None:
        result = compare_routes(
            torch.tensor([[2, 0], [2, 1]]),
            torch.tensor([[0.3, 0.7], [0.4, 0.6]]),
            torch.tensor([[0, 2], [1, 2]]),
            torch.tensor([[0.7, 0.3], [0.6, 0.4]]),
        )
        self.assertEqual(result["id_mismatches"], 0)
        self.assertEqual(result["weight_mismatches"], 0)
        self.assertEqual(result["max_abs_weight_error"], 0.0)

    def test_compare_routes_detects_weight_attached_to_wrong_expert(self) -> None:
        result = compare_routes(
            torch.tensor([[2, 0]]),
            torch.tensor([[0.3, 0.7]]),
            torch.tensor([[0, 2]]),
            torch.tensor([[0.3, 0.7]]),
        )
        self.assertEqual(result["id_mismatches"], 0)
        self.assertEqual(result["weight_mismatches"], 2)


if __name__ == "__main__":
    unittest.main()
