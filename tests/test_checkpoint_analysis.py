from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ep_predict.analysis.checkpoint import analyze_checkpoint_trajectories
from ep_predict.analysis.h2 import analyze_h2
from ep_predict.collect import _tokenize_prompt
from ep_predict.tracing.schema import TraceRecord
from ep_predict.tracing.storage import RequestTraceStore, write_json


class _Tokenizer:
    chat_template = "template"

    def __init__(self) -> None:
        self.raw_calls = 0
        self.chat_calls = 0

    def __call__(self, *_args, **_kwargs):
        self.raw_calls += 1
        return {"mode": "raw"}

    def apply_chat_template(self, *_args, **_kwargs):
        self.chat_calls += 1
        return {"mode": "chat"}


class SharedTokenizationTest(unittest.TestCase):
    def test_raw_mode_bypasses_checkpoint_chat_template(self) -> None:
        tokenizer = _Tokenizer()
        encoded = _tokenize_prompt(
            tokenizer,
            "same text",
            32,
            prompt_format="raw",
        )
        self.assertEqual(encoded["mode"], "raw")
        self.assertEqual(tokenizer.raw_calls, 1)
        self.assertEqual(tokenizer.chat_calls, 0)


class CheckpointAnalysisIntegrationTest(unittest.TestCase):
    def _make_run(self, root: Path, checkpoint: str) -> Path:
        run = root / checkpoint
        store = RequestTraceStore(run)
        domains = ("code", "conversation", "general", "math")
        for request_id in range(8):
            domain = domains[request_id % len(domains)]
            records: list[TraceRecord] = []
            for position in range(6):
                base_route = (request_id + position) % 2
                for layer in range(4):
                    expert = base_route
                    if checkpoint == "instruct" and layer >= 2:
                        expert = 1 - expert
                    records.append(
                        TraceRecord(
                            run_id=checkpoint,
                            request_id=request_id,
                            sample_id=f"sample-{request_id}",
                            phase="prefill",
                            token_position=position,
                            input_token_id=100 + position,
                            layer_id=layer,
                            moe_layer_index=layer,
                            selected_expert_ids=[expert],
                            selected_expert_weights=[1.0],
                            batch_id=0,
                            batch_size=1,
                            dataset_name="synthetic",
                            domain=domain,
                        )
                    )
            store.write_request(request_id, f"sample-{request_id}", records)
        write_json(run / "run_manifest.json", {"run_id": checkpoint})
        write_json(
            run / "model_report.json",
            {
                "model_commit": f"{checkpoint}-commit",
                "routers": [
                    {
                        "layer_id": layer,
                        "num_experts": 4,
                        "top_k": 1,
                    }
                    for layer in range(4)
                ],
            },
        )
        h2_config = {
            "trace_run": str(run),
            "analysis_id": f"{checkpoint}-h2",
            "split_seed": 17,
            "test_requests_per_domain": 1,
            "capacities": [1, 2],
            "lookaheads": [1, 2, 3],
            "previous_window_tokens": 8,
            "decision_gate": {
                "phase": "prefill",
                "baseline": "transition",
                "comparator": "static",
                "capacity_experts": 2,
                "min_mean_selection_coverage_gain": 0.0,
                "min_mean_complete_token_coverage_gain": 0.0,
                "min_positive_scope_fraction": 0.0,
                "min_positive_domains": 0,
            },
        }
        analyze_h2(run, h2_config)
        return run

    def test_matched_comparison_and_cross_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self._make_run(root, "base")
            instruct = self._make_run(root, "instruct")
            output = root / "comparison"
            result = analyze_checkpoint_trajectories(
                {
                    "analysis_id": "synthetic-c0",
                    "output_dir": str(output),
                    "phase": "prefill",
                    "primary_source_layer": 0,
                    "hotset_capacity": 2,
                    "capacities": [1, 2],
                    "lookaheads": [1, 2, 3],
                    "checkpoints": {
                        "base": {"run": str(base)},
                        "instruct": {"run": str(instruct)},
                    },
                    "decision_gate": {
                        "source_layer": 0,
                        "lookahead": 3,
                        "capacity_experts": 2,
                        "min_abs_selection_gain_difference": 0.05,
                        "min_consistent_domains": 3,
                    },
                }
            )
            self.assertEqual(
                result["integrity"]["input_token_id_mismatches"],
                0,
            )
            self.assertEqual(
                result["integrity"]["self_transfer_max_abs_difference"],
                0.0,
            )
            self.assertTrue(
                (output / "analysis" / "c0" / "matched_route_overlap.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
