from __future__ import annotations

import importlib.metadata
import os
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _base_architecture(name: str | None) -> str | None:
    if not name:
        return None
    return name.split(":", 1)[0]


def verify_rocm_runtime(
    *,
    device_index: int = 0,
    expected_architecture: str = "gfx950",
    expected_visible_devices: int = 1,
) -> dict[str, Any]:
    """Exercise the ROCm wheel, BF16 kernels, and project router hooks.

    PyTorch deliberately exposes ROCm devices through ``torch.cuda``. This
    verifier rejects a CUDA or CPU wheel even if the rest of the Python stack
    imports successfully.
    """

    import torch
    import transformers

    errors: list[str] = []
    checks: dict[str, Any] = {}
    report: dict[str, Any] = {
        "packages": {
            name: _package_version(name)
            for name in (
                "accelerate",
                "pytorch-triton-rocm",
                "torch",
                "transformers",
                "triton-rocm",
            )
        },
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": torch.version.hip,
        "transformers_version": transformers.__version__,
        "device_nodes": {
            "/dev/kfd": os.path.exists("/dev/kfd"),
            "/dev/dri": os.path.exists("/dev/dri"),
        },
        "visibility": {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "checks": checks,
        "errors": errors,
    }

    checks["rocm_wheel"] = torch.version.hip is not None
    if not checks["rocm_wheel"]:
        errors.append(
            "torch.version.hip is empty: the installed torch wheel is not a ROCm build"
        )
    checks["not_cuda_wheel"] = torch.version.cuda is None
    if not checks["not_cuda_wheel"]:
        errors.append(
            f"torch.version.cuda={torch.version.cuda}: a CUDA wheel was installed"
        )

    try:
        accelerator_available = bool(torch.cuda.is_available())
    except Exception as error:  # pragma: no cover - hardware/driver dependent
        accelerator_available = False
        errors.append(f"ROCm initialization failed: {error}")
    checks["accelerator_available"] = accelerator_available
    if not accelerator_available:
        if not os.path.exists("/dev/kfd"):
            errors.append("/dev/kfd is not exposed to this process")
        if not os.path.exists("/dev/dri"):
            errors.append("/dev/dri is not exposed to this process")
        if not errors or not any("initialization failed" in item for item in errors):
            errors.append("torch.cuda.is_available() is false for the ROCm runtime")
        report["state"] = "failed"
        return report

    device_count = int(torch.cuda.device_count())
    report["visible_device_count"] = device_count
    checks["visible_device_count"] = device_count == expected_visible_devices
    if not checks["visible_device_count"]:
        errors.append(
            f"expected {expected_visible_devices} visible GPU, found {device_count}; "
            "set HIP_VISIBLE_DEVICES=0 for single-GPU experiments"
        )
    if device_index < 0 or device_index >= device_count:
        errors.append(
            f"device index {device_index} is outside the {device_count} visible devices"
        )
        report["state"] = "failed"
        return report

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    architecture = _base_architecture(getattr(properties, "gcnArchName", None))
    report["device"] = {
        "logical_index": device_index,
        "name": properties.name,
        "architecture": architecture,
        "architecture_detail": getattr(properties, "gcnArchName", None),
        "memory_bytes": int(properties.total_memory),
    }
    checks["expected_architecture"] = architecture == expected_architecture
    if not checks["expected_architecture"]:
        errors.append(
            f"expected architecture {expected_architecture}, found {architecture!r}"
        )

    try:
        left = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
        right = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
        product = left @ right
        torch.cuda.synchronize(device)
        checks["bf16_matmul"] = bool(torch.isfinite(product).all().item())
        if not checks["bf16_matmul"]:
            errors.append("BF16 matrix multiplication produced non-finite values")
    except Exception as error:  # pragma: no cover - hardware/driver dependent
        checks["bf16_matmul"] = False
        errors.append(f"BF16 matrix multiplication failed: {error}")

    try:
        host = torch.arange(4096, dtype=torch.float32).pin_memory()
        copied = host.to(device=device, non_blocking=True)
        torch.cuda.synchronize(device)
        checks["pinned_host_to_device_copy"] = bool(
            torch.equal(copied.cpu(), host)
        )
        if not checks["pinned_host_to_device_copy"]:
            errors.append("pinned host-to-device copy changed tensor values")
    except Exception as error:  # pragma: no cover - hardware/driver dependent
        checks["pinned_host_to_device_copy"] = False
        errors.append(f"pinned host-to-device copy failed: {error}")

    try:
        from transformers import OlmoeConfig, OlmoeForCausalLM

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
        model = (
            OlmoeForCausalLM(config)
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )
        routers = discover_routers(model, [".mlp.gate"])
        with RouterTracer(model, routers) as tracer:
            tracer.start_request(
                RequestContext(
                    run_id="rocm-sanity",
                    request_id=0,
                    sample_id="tiny-random-olmoe",
                    dataset_name="runtime-verification",
                    domain="synthetic",
                )
            )
            with torch.inference_mode():
                output = model(
                    input_ids=torch.tensor([[1, 2, 3]], device=device),
                    attention_mask=torch.ones(
                        (1, 3), dtype=torch.long, device=device
                    ),
                    use_cache=False,
                )
            records, features, summary = tracer.finish_request()
        torch.cuda.synchronize(device)
        model_ok = (
            tuple(output.logits.shape) == (1, 3, 64)
            and len(routers) == 2
            and len(records) == 6
            and features is None
            and summary["router_validation_mismatches"] == 0
            and summary["router_calls_per_forward"] == [2]
        )
        checks["tiny_olmoe_router_trace"] = model_ok
        report["tiny_olmoe"] = {
            "dtype": str(output.logits.dtype),
            "logits_shape": list(output.logits.shape),
            "router_count": len(routers),
            "trace_records": len(records),
            "router_validation_mismatches": summary[
                "router_validation_mismatches"
            ],
        }
        if not model_ok:
            errors.append("tiny OLMoE forward or router-trace integrity check failed")
    except Exception as error:  # pragma: no cover - hardware/driver dependent
        checks["tiny_olmoe_router_trace"] = False
        errors.append(f"tiny OLMoE router-trace test failed: {error}")

    report["state"] = "ready" if not errors else "failed"
    return report
