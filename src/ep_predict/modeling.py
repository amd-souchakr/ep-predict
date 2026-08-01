from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from ep_predict.tracing.hooks import RouterSpec, discover_routers


def _torch_dtype(name: str):
    import torch

    aliases = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return aliases[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}; choose {sorted(aliases)}") from error


def load_model_and_tokenizer(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"configured device {device}, but CUDA is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_id"],
        revision=config.get("tokenizer_revision"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config.get("revision"),
        torch_dtype=_torch_dtype(config.get("dtype", "bfloat16")),
        device_map={"": device},
        low_cpu_mem_usage=True,
        attn_implementation=config.get("attention_implementation", "sdpa"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    model.eval()
    return model, tokenizer


def inspect_loaded_model(
    model: Any,
    *,
    router_name_contains: list[str] | None = None,
) -> tuple[dict[str, Any], list[RouterSpec]]:
    routers = discover_routers(model, router_name_contains)
    if not routers:
        raise RuntimeError("no explicit router modules discovered")

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    router_reports: list[dict[str, Any]] = []
    expert_parameter_ids: set[int] = set()

    for spec in routers:
        parent_name = spec.name.rsplit(".", 1)[0]
        parent = model.get_submodule(parent_name)
        experts = getattr(parent, "experts", None)
        expert_bytes = 0
        expert_parameters = 0
        expert_shapes: list[dict[str, Any]] = []
        if experts is not None:
            for name, parameter in experts.named_parameters():
                identity = id(parameter)
                if identity in expert_parameter_ids:
                    continue
                expert_parameter_ids.add(identity)
                bytes_for_tensor = parameter.numel() * parameter.element_size()
                expert_bytes += bytes_for_tensor
                expert_parameters += parameter.numel()
                expert_shapes.append(
                    {
                        "name": name,
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                        "bytes": bytes_for_tensor,
                    }
                )

        router_reports.append(
            {
                "name": spec.name,
                "class": spec.module.__class__.__name__,
                "layer_id": spec.layer_id,
                "moe_layer_index": spec.moe_layer_index,
                "num_experts": spec.num_experts,
                "top_k": spec.top_k,
                "expert_parameters_total": expert_parameters,
                "expert_bytes_total": expert_bytes,
                "expert_bytes_each": (
                    expert_bytes // spec.num_experts if expert_bytes else None
                ),
                "expert_tensors": expert_shapes,
            }
        )

    expert_bytes_total = sum(report["expert_bytes_total"] for report in router_reports)
    config = model.config
    report = {
        "model_class": model.__class__.__name__,
        "model_type": getattr(config, "model_type", None),
        "model_commit": getattr(config, "_commit_hash", None),
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "expert_parameter_bytes": expert_bytes_total,
        "non_expert_parameter_bytes": parameter_bytes - expert_bytes_total,
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "router_count": len(routers),
        "routers": router_reports,
    }
    return report, routers


def environment_report() -> dict[str, Any]:
    import torch
    import transformers

    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
        }
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    packages = {}
    for name in ("accelerate", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "gpu": gpu,
        "git_commit": git_commit,
        "transformers_version": transformers.__version__,
    }


def print_model_summary(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
