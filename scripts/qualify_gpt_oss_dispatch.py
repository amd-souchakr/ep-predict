#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

from ep_predict.gpt_oss_qualification import (
    compare_routes,
    decode_dispatch_inputs,
    expected_from_logits,
    sha256_file,
    tensor_nbytes,
    write_json,
)

DEFAULT_SNAPSHOT = Path(
    "/home/souchakr/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
PROMPT = "Milestone C dispatch parity."


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def checkpoint_inspection(snapshot: Path) -> dict[str, Any]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    config = json.loads((snapshot / "config.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    layer0: dict[str, dict[str, Any]] = {}
    dtype_bytes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    layer0_payload = 0
    for shard in shards:
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            # ``safe_open`` exposes keys through its API but is not a dict.
            for key in handle.keys():  # noqa: SIM118
                if not key.startswith("model.layers.0.mlp.experts."):
                    continue
                view = handle.get_slice(key)
                shape = list(view.get_shape())
                dtype = view.get_dtype()
                elements = 1
                for size in shape:
                    elements *= size
                nbytes = elements * dtype_bytes[dtype]
                layer0[key.rsplit(".", 1)[-1]] = {"shape": shape, "dtype": dtype, "payload_bytes": nbytes}
                layer0_payload += nbytes
    num_experts = int(config["num_local_experts"])
    return {
        "config_sha256": sha256_file(snapshot / "config.json"),
        "tokenizer_json_sha256": sha256_file(snapshot / "tokenizer.json"),
        "index_sha256": sha256_file(snapshot / "model.safetensors.index.json"),
        "tensor_payload_bytes": int(index["metadata"]["total_size"]),
        "safetensors_file_bytes": sum((snapshot / shard).stat().st_size for shard in shards),
        "shards": shards,
        "expert_tensors_layer0": layer0,
        "expert_stored_bytes_per_layer": layer0_payload,
        "expert_stored_bytes_per_expert": layer0_payload // num_experts,
        "geometry": {
            "routed_layers": int(config["num_hidden_layers"]),
            "experts_per_layer": num_experts,
            "top_k": int(config["num_experts_per_tok"]),
            "hidden_size": int(config["hidden_size"]),
            "intermediate_size": int(config["intermediate_size"]),
            "shared_experts": 0,
        },
        "quantization_config": config.get("quantization_config"),
    }


def eager_control() -> dict[str, Any]:
    torch.manual_seed(20260801)
    config = GptOssConfig(
        hidden_size=8,
        intermediate_size=6,
        num_local_experts=5,
        num_experts_per_tok=3,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
    )
    config._experts_implementation = "eager"
    mlp = GptOssMLP(config).eval()
    with torch.no_grad():
        for ordinal, parameter in enumerate(mlp.parameters(), start=1):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(torch.sin(values * (0.013 * ordinal)))

    captured_router: list[tuple[torch.Tensor, torch.Tensor]] = []
    captured_expert: list[tuple[torch.Tensor, torch.Tensor]] = []

    def router_hook(_module, _args, output):
        captured_router.append((output[2].detach().clone(), output[1].detach().clone()))

    def expert_pre_hook(_module, args, kwargs):
        captured_expert.append((args[1].detach().clone(), args[2].detach().clone()))

    handles = [
        mlp.router.register_forward_hook(router_hook),
        mlp.experts.register_forward_pre_hook(expert_pre_hook, with_kwargs=True),
    ]
    hidden = torch.tensor(
        [[[0.2, -0.1, 0.4, 0.7, -0.3, 0.5, 0.9, -0.8], [-0.6, 0.3, 0.8, -0.2, 0.1, 0.4, -0.5, 0.7]]]
    )
    with torch.inference_mode():
        mlp(hidden)
    for handle in handles:
        handle.remove()

    comparison = compare_routes(*captured_router[0], *captured_expert[0])
    sums = captured_expert[0][1].float().sum(dim=-1)
    return {
        "router_hook_calls": len(captured_router),
        "expert_hook_calls": len(captured_expert),
        "tokens": int(hidden.shape[1]),
        **comparison,
        "selected_weight_sum_min": float(sums.min().item()),
        "selected_weight_sum_max": float(sums.max().item()),
        "qualified": comparison["id_mismatches"] == 0 and comparison["weight_mismatches"] == 0,
    }


def _storage_component_bytes(experts: torch.nn.Module) -> tuple[dict[str, int | None], int | None]:
    components: dict[str, int | None] = {}
    for name in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"):
        components[name] = tensor_nbytes(getattr(experts, name))
    for projection in ("gate_up_proj", "down_proj"):
        precision = getattr(experts, f"{projection}_precision_config", None)
        components[f"{projection}_scales"] = tensor_nbytes(getattr(precision, "weight_scale", None))
    known = [value for value in components.values() if value is not None]
    total = sum(known) if len(known) == len(components) else None
    return components, total


def _kernel_provenance(model: torch.nn.Module) -> dict[str, Any]:
    from transformers.integrations import mxfp4

    quantizer = getattr(model, "hf_quantizer", None)
    # Transformers removes/does not retain the quantizer on every load path,
    # but the MXFP4 integration receives the exact hub object as a module
    # global when it installs the replacement forward.
    hub = getattr(quantizer, "triton_kernels_hub", None) or getattr(mxfp4, "triton_kernels_hub", None)
    paths: set[str] = set()
    for child_name in ("routing", "matmul_ogs", "tensor", "swiglu"):
        child = getattr(hub, child_name, None)
        path = getattr(child, "__file__", None)
        if path:
            paths.add(str(Path(path)))
    source_files = [
        {"path": path, "sha256": sha256_file(Path(path))} for path in sorted(paths) if Path(path).is_file()
    ]
    hub_path = getattr(hub, "__file__", None)
    provenance_paths = [Path(item["path"]) for item in source_files]
    if hub_path:
        provenance_paths.append(Path(hub_path))
    snapshot_revisions = sorted(
        {
            path.parts[path.parts.index("snapshots") + 1]
            for path in provenance_paths
            if "snapshots" in path.parts
        }
    )
    return {
        "kernels_package": package_version("kernels"),
        "kernels_data_package": package_version("kernels-data"),
        "hub_object_type": type(hub).__name__ if hub is not None else None,
        "hub_repr": repr(hub) if hub is not None else None,
        "hub_module_path": hub_path,
        "snapshot_revisions": snapshot_revisions,
        "source_files": source_files,
    }


def native_checkpoint_run(snapshot: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("native qualification requires one visible ROCm/CUDA device")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype="auto",
        device_map={"": "cuda:0"},
        use_kernels=False,
    ).eval()

    expected: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    consumed: dict[int, list[Any]] = defaultdict(list)
    ordinary_router_calls: dict[int, int] = defaultdict(int)
    handles = []

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        top_k = int(mlp.router.top_k)

        def mlp_pre_hook(module, args, kwargs, layer_idx=layer_idx):
            hidden = args[0].reshape(-1, module.router.hidden_dim)
            logits = F.linear(hidden, module.router.weight, module.router.bias)
            expected[layer_idx].append(tuple(value.detach().clone() for value in expected_from_logits(logits, module.router.top_k)))

        def dispatch_pre_hook(module, args, kwargs, layer_idx=layer_idx, top_k=top_k):
            routing_data, gather_idx = args[1], args[2]
            num_tokens = expected[layer_idx][-1][0].shape[0]
            consumed[layer_idx].append(
                decode_dispatch_inputs(routing_data, gather_idx, num_tokens=num_tokens, top_k=top_k)
            )

        def router_hook(_module, _args, _output, layer_idx=layer_idx):
            ordinary_router_calls[layer_idx] += 1

        handles.extend(
            [
                mlp.register_forward_pre_hook(mlp_pre_hook, with_kwargs=True),
                mlp.experts.register_forward_pre_hook(dispatch_pre_hook, with_kwargs=True),
                mlp.router.register_forward_hook(router_hook),
            ]
        )

    encoded = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids.to("cuda:0")
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()

    rows: list[dict[str, Any]] = []
    all_weight_sums: list[torch.Tensor] = []
    for layer_idx in range(model.config.num_hidden_layers):
        row: dict[str, Any] = {
            "layer": layer_idx,
            "expected_calls": len(expected[layer_idx]),
            "dispatch_hook_calls": len(consumed[layer_idx]),
            "ordinary_router_hook_calls": ordinary_router_calls[layer_idx],
            "tokens": int(expected[layer_idx][0][0].shape[0]) if expected[layer_idx] else 0,
            "id_mismatches": -1,
            "weight_mismatches": -1,
            "max_abs_weight_error": float("inf"),
        }
        if len(expected[layer_idx]) == len(consumed[layer_idx]) == 1:
            observation = consumed[layer_idx][0]
            comparison = compare_routes(*expected[layer_idx][0], observation.expert_ids, observation.weights)
            row.update(comparison)
            all_weight_sums.append(observation.weights.float().sum(dim=-1))
        rows.append(row)

    first_experts = model.model.layers[0].mlp.experts
    loaded_components, loaded_layer_bytes = _storage_component_bytes(first_experts)
    device = next(model.parameters()).device
    weight_sums = torch.cat(all_weight_sums) if all_weight_sums else torch.tensor([])
    native = {
        "prompt": PROMPT,
        "input_token_ids": input_ids.detach().cpu().reshape(-1).tolist(),
        "input_tokens": int(input_ids.numel()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "compute_input_dtype": str(model.model.embed_tokens.weight.dtype),
        "expert_class": type(first_experts).__name__,
        "mlp_forward_module": inspect.getmodule(model.model.layers[0].mlp.forward).__name__,
        "loaded_expert_components_layer0_bytes": loaded_components,
        "loaded_expert_bytes_per_layer": loaded_layer_bytes,
        "loaded_expert_bytes_per_expert": loaded_layer_bytes // model.config.num_local_experts if loaded_layer_bytes else None,
        "router_ordering": "descending router logit (torch.topk; tie order unspecified)",
        "dispatch_ordering": "selected IDs sorted ascending per token, then stable expert-major gather",
        "selected_weight_normalization": "softmax over selected top-k only; sums to one",
        "selected_weight_sum_min": float(weight_sums.min().item()) if weight_sums.numel() else None,
        "selected_weight_sum_max": float(weight_sums.max().item()) if weight_sums.numel() else None,
        "ordinary_router_hook_calls_total": sum(ordinary_router_calls.values()),
        "dispatch_hook_calls_total": sum(len(value) for value in consumed.values()),
        "kernel_provenance": _kernel_provenance(model),
    }
    del model
    torch.cuda.empty_cache()
    return native, rows


def source_provenance() -> dict[str, Any]:
    import transformers.models.gpt_oss.modeling_gpt_oss as modeling
    from transformers.integrations import mxfp4

    paths = [Path(inspect.getfile(modeling)), Path(inspect.getfile(mxfp4))]
    return {str(path): sha256_file(path) for path in paths}


def gate(eager: dict[str, Any], native: dict[str, Any], rows: list[dict[str, Any]], routed_layers: int) -> tuple[str, dict[str, bool]]:
    checks = {
        "eager_parity": bool(eager["qualified"]),
        "native_all_ids_match": all(row["id_mismatches"] == 0 for row in rows),
        "native_all_weights_match": all(row["weight_mismatches"] == 0 for row in rows),
        "native_weight_tolerance": all(row["max_abs_weight_error"] <= 1e-6 for row in rows),
        "complete_dispatch_hook_coverage": len(rows) == routed_layers and all(row["dispatch_hook_calls"] == 1 for row in rows),
        "complete_expected_call_coverage": len(rows) == routed_layers and all(row["expected_calls"] == 1 for row in rows),
        "ordinary_router_bypass_observed": native["ordinary_router_hook_calls_total"] == 0,
        "native_mxfp4_experts": native["expert_class"] == "Mxfp4GptOssExperts",
        # BF16 stores each of four softmax outputs independently, so their
        # represented sum need not be exactly one. Four BF16 epsilons is a
        # conservative representation-only bound; pairwise parity remains 1e-6.
        "weights_normalized": native["selected_weight_sum_min"] is not None
        and abs(native["selected_weight_sum_min"] - 1.0) <= 4 * torch.finfo(torch.bfloat16).eps
        and abs(native["selected_weight_sum_max"] - 1.0) <= 4 * torch.finfo(torch.bfloat16).eps,
        "loaded_bytes_recorded": native["loaded_expert_bytes_per_expert"] is not None,
        "kernel_provenance_recorded": bool(native["kernel_provenance"]["hub_repr"]),
        "custom_mlp_forward_observed": native["mlp_forward_module"] == "transformers.integrations.mxfp4",
    }
    return ("QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED"), checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/gpt-oss-20b-milestone-c"))
    args = parser.parse_args()

    inspection = checkpoint_inspection(args.snapshot)
    eager = eager_control()
    native, rows = native_checkpoint_run(args.snapshot)
    decision, checks = gate(eager, native, rows, inspection["geometry"]["routed_layers"])

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "dispatch_parity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": 1,
        "milestone": "C",
        "decision": decision,
        "checkpoint": {"model_id": "openai/gpt-oss-20b", "revision": REVISION, "snapshot": str(args.snapshot)},
        "environment": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "torch_hip": torch.version.hip,
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "accelerate": package_version("accelerate"),
            "safetensors": package_version("safetensors"),
        },
        "transformers_source_sha256": source_provenance(),
        "checkpoint_inspection": inspection,
        "eager_control": eager,
        "native_mxfp4": native,
        "gate_checks": checks,
        "claim_boundary": "router/dispatch instrumentation qualification only; no routing-distribution or performance claim",
    }
    write_json(args.output / "qualification.json", result)
    print(json.dumps({"decision": decision, "gate_checks": checks}, indent=2))
    if decision != "QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
