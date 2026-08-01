from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ep_predict.tracing.schema import TraceRecord
from ep_predict.tracing.storage import RequestTraceStore, iter_trace_records


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


if __name__ == "__main__":
    unittest.main()
