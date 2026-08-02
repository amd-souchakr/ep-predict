from __future__ import annotations

import unittest

from ep_predict.gpt_oss_tracing import (
    compare_repeat_records,
    summarize_routing,
    validate_request_coverage,
)
from ep_predict.tracing.schema import TraceRecord


def record(position: int, layer: int, *, phase: str | None = None) -> TraceRecord:
    return TraceRecord(
        run_id="d",
        request_id=0,
        sample_id="sample",
        phase=phase or ("prefill" if position < 2 else "decode"),
        token_position=position,
        input_token_id=position + 10,
        layer_id=layer,
        moe_layer_index=layer,
        selected_expert_ids=[0, 1],
        selected_expert_weights=[0.75, 0.25],
        batch_id=position,
        batch_size=1,
        dataset_name="milestone-d",
        domain="test",
    )


class GptOssTracingTest(unittest.TestCase):
    def test_complete_prompt_and_terminal_decode_coverage(self) -> None:
        records = [record(position, layer) for position in range(3) for layer in range(2)]
        result = validate_request_coverage(
            records,
            prompt_tokens=2,
            generated_tokens=1,
            expected_layers=2,
            expected_top_k=2,
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["observed_records"], 6)

    def test_coverage_detects_duplicate_and_missing_key(self) -> None:
        records = [record(0, 0), record(0, 0), record(1, 0)]
        result = validate_request_coverage(
            records,
            prompt_tokens=2,
            generated_tokens=0,
            expected_layers=1,
            expected_top_k=2,
        )
        self.assertFalse(result["complete"])
        self.assertEqual(result["duplicate_token_layer_keys"], 1)

    def test_repeat_comparison_and_summary(self) -> None:
        records = [record(position, layer) for position in range(3) for layer in range(2)]
        comparison = compare_repeat_records(records, records, weight_atol=1e-6)
        self.assertTrue(comparison["identical"])
        summary = summarize_routing(records)
        self.assertEqual(len(summary), 4)
        self.assertTrue(all(row["top_expert"] == 0 for row in summary))

    def test_repeat_comparison_rejects_duplicate_key(self) -> None:
        records = [record(position, layer) for position in range(2) for layer in range(2)]
        comparison = compare_repeat_records(records, records + [records[0]], weight_atol=1e-6)
        self.assertFalse(comparison["identical"])
        self.assertEqual(comparison["repeated_duplicate_keys"], 1)


if __name__ == "__main__":
    unittest.main()
