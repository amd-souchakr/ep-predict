from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ep_predict.tracing.schema import TraceRecord
from ep_predict.tracing.storage import (
    RequestFeatureStore,
    RequestTraceStore,
    iter_trace_records,
)


class StorageTest(unittest.TestCase):
    def test_atomic_request_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            record = TraceRecord(
                run_id="run",
                request_id=3,
                sample_id="sample/unsafe",
                phase="prefill",
                token_position=0,
                input_token_id=42,
                layer_id=1,
                moe_layer_index=0,
                selected_expert_ids=[2, 4],
                selected_expert_weights=[0.6, 0.4],
                batch_id=0,
                batch_size=1,
                dataset_name="test",
                domain="code",
            )
            store = RequestTraceStore(run_dir)
            path = store.write_request(3, "sample/unsafe", [record])
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())
            restored = list(iter_trace_records(run_dir))
            self.assertEqual(restored[0]["selected_expert_ids"], [2, 4])
            self.assertEqual(restored[0]["metadata_version"], 1)

    def test_feature_store_round_trip_without_pickle(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("feature dependencies are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            record = TraceRecord(
                run_id="run",
                request_id=0,
                sample_id="sample",
                phase="prefill",
                token_position=2,
                input_token_id=11,
                layer_id=4,
                moe_layer_index=4,
                selected_expert_ids=[1, 3],
                selected_expert_weights=[0.6, 0.4],
                batch_id=0,
                batch_size=1,
                dataset_name="unit",
                domain="synthetic",
            )
            store = RequestFeatureStore(run_dir)
            path = store.write_request(
                0,
                "sample",
                [record],
                torch.tensor([[1.5, -2.0]], dtype=torch.float16),
            )
            with np.load(path, allow_pickle=False) as shard:
                self.assertEqual(shard["hidden_feature"].shape, (1, 2))
                self.assertEqual(int(shard["token_position"][0]), 2)
                self.assertEqual(int(shard["layer_id"][0]), 4)


if __name__ == "__main__":
    unittest.main()
