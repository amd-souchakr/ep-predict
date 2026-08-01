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
            records, summary = tracer.finish_request()

        self.assertEqual(summary["router_validation_mismatches"], 0)
        self.assertEqual(summary["router_calls_per_forward"], [2])
        self.assertEqual(len(records), 6)
        self.assertTrue(all(len(record.selected_expert_ids) == 2 for record in records))


if __name__ == "__main__":
    unittest.main()
