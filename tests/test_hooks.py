from __future__ import annotations

import unittest

try:
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM
except ImportError:
    torch = None
    OlmoeConfig = None
    OlmoeForCausalLM = None


@unittest.skipIf(torch is None, "inference dependency group is not installed")
class HookIntegrationTest(unittest.TestCase):
    def test_tiny_random_olmoe_trace_matches_router(self) -> None:
        assert OlmoeConfig is not None
        assert OlmoeForCausalLM is not None
        from ep_predict.tracing.hooks import RouterTracer, discover_routers
        from ep_predict.tracing.schema import RequestContext

        config = OlmoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = OlmoeForCausalLM(config).eval()
        routers = discover_routers(model, [".mlp.gate"])
        self.assertEqual(len(routers), 2)

        with RouterTracer(model, routers) as tracer:
            tracer.start_request(
                RequestContext(
                    run_id="tiny",
                    request_id=0,
                    sample_id="sample",
                    dataset_name="unit",
                    domain="synthetic",
                )
            )
            with torch.inference_mode():
                model(
                    input_ids=torch.tensor([[1, 2, 3]]),
                    attention_mask=torch.ones((1, 3), dtype=torch.long),
                    use_cache=False,
                )
            records, features, summary = tracer.finish_request()

        self.assertEqual(summary["router_validation_mismatches"], 0)
        self.assertEqual(summary["router_calls_per_forward"], [2])
        self.assertEqual(len(records), 6)
        self.assertIsNone(features)
        self.assertTrue(all(len(record.selected_expert_ids) == 2 for record in records))

    def test_projected_router_inputs_align_with_records(self) -> None:
        assert OlmoeConfig is not None
        assert OlmoeForCausalLM is not None
        from ep_predict.tracing.hooks import (
            RouterInputProjector,
            RouterTracer,
            discover_routers,
        )
        from ep_predict.tracing.schema import RequestContext

        config = OlmoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = OlmoeForCausalLM(config).eval()
        routers = discover_routers(model, [".mlp.gate"])
        projector = RouterInputProjector(
            input_dimension=16,
            output_dimension=7,
            seed=31,
        )

        with RouterTracer(
            model,
            routers,
            feature_projector=projector,
        ) as tracer:
            tracer.start_request(
                RequestContext(
                    run_id="tiny-features",
                    request_id=0,
                    sample_id="sample",
                    dataset_name="unit",
                    domain="synthetic",
                )
            )
            with torch.inference_mode():
                model(
                    input_ids=torch.tensor([[1, 2, 3]]),
                    attention_mask=torch.ones((1, 3), dtype=torch.long),
                    use_cache=False,
                )
            records, features, summary = tracer.finish_request()

        assert features is not None
        self.assertEqual(tuple(features.shape), (len(records), 7))
        self.assertEqual(features.dtype, torch.float16)
        self.assertEqual(summary["hidden_feature_records"], len(records))
        duplicate = RouterInputProjector(
            input_dimension=16,
            output_dimension=7,
            seed=31,
        )
        self.assertEqual(projector.matrix_sha256, duplicate.matrix_sha256)


if __name__ == "__main__":
    unittest.main()
