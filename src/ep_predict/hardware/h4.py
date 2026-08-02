from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ep_predict.config import config_fingerprint
from ep_predict.modeling import environment_report, load_model_and_tokenizer
from ep_predict.tracing.storage import write_json


MIB = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_hardware_report(
    report: dict[str, Any], hardware: dict[str, Any]
) -> None:
    """Reject calibration on a platform outside the frozen hardware scope."""

    if not hardware:
        return
    if report.get("torch_backend") != "rocm":
        raise RuntimeError(
            "configured H4 hardware requires the ROCm PyTorch backend"
        )
    gpu = report.get("gpu") or {}
    expected_count = int(hardware.get("visible_device_count", 1))
    observed_count = int(gpu.get("device_count", 0))
    if observed_count != expected_count:
        raise RuntimeError(
            f"expected {expected_count} visible GPU, found {observed_count}"
        )
    expected_architecture = hardware.get("required_architecture")
    observed_architecture = str(gpu.get("architecture") or "").split(":", 1)[0]
    if expected_architecture and observed_architecture != expected_architecture:
        raise RuntimeError(
            f"expected architecture {expected_architecture}, found "
            f"{observed_architecture or None!r}"
        )


def _load_prompt(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records[index]


def _tokenize(tokenizer: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_transfer(samples: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for row in samples:
        grouped.setdefault(int(row["bytes"]), []).append(float(row["elapsed_ms"]))
    points = [
        (float(size), statistics.median(values))
        for size, values in sorted(grouped.items())
    ]
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    slope_ms_per_byte = (
        sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    )
    intercept_ms = mean_y - slope_ms_per_byte * mean_x
    nonnegative_startup_ms = max(0.0, intercept_ms)
    bandwidth_gbps = 1e-6 / slope_ms_per_byte
    return {
        "raw_intercept_ms": intercept_ms,
        "startup_ms": nonnegative_startup_ms,
        "slope_ms_per_byte": slope_ms_per_byte,
        "effective_bandwidth_gbps": bandwidth_gbps,
    }


def measure_h4(
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "H4 timing calibration requires a GPU through PyTorch's "
            "torch.cuda API (CUDA or ROCm)"
        )
    environment = environment_report()
    _validate_hardware_report(environment, experiment_config.get("hardware", {}))
    output_dir = Path(experiment_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(model_config, experiment_config)
    definition_path = output_dir / "measurement_definition.json"
    if definition_path.exists():
        existing = json.loads(definition_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError("H4 measurement directory has a different config")
    else:
        write_json(
            definition_path,
            {
                "config_fingerprint": fingerprint,
                "model_config": model_config,
                "experiment_config": experiment_config,
                "timing_semantics": "hook_free_cached_token_full_forward",
            },
        )

    timing = experiment_config["timing"]
    prompt_path = Path(timing["prompt_file"])
    prompt_sha256 = _sha256(prompt_path)
    model, tokenizer = load_model_and_tokenizer(model_config)
    warmups = int(timing["warmup_decode_steps"])
    measured = int(timing["measured_decode_steps"])
    timing_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for prompt_index in timing["prompt_indices"]:
            prompt = _load_prompt(prompt_path, int(prompt_index))
            encoded = _tokenize(
                tokenizer,
                str(prompt["prompt"]),
                int(timing["max_prompt_tokens"]),
            )
            encoded = {
                name: tensor.to(model.device) for name, tensor in encoded.items()
            }
            output = model(**encoded, use_cache=True)
            past = output.past_key_values
            next_token = output.logits[:, -1:].argmax(dim=-1)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(encoded["input_ids"])
            total_steps = warmups + measured
            for step in range(total_steps):
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                end.record()
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
                past = output.past_key_values
                next_token = output.logits[:, -1:].argmax(dim=-1)
                if step >= warmups:
                    timing_rows.append(
                        {
                            "prompt_index": int(prompt_index),
                            "sample_id": prompt["sample_id"],
                            "domain": prompt["domain"],
                            "decode_step": step - warmups,
                            "context_tokens": int(attention_mask.shape[1]),
                            "elapsed_ms": elapsed_ms,
                        }
                    )

    transfer = experiment_config["transfer"]
    transfer_rows: list[dict[str, Any]] = []
    for size_mib in transfer["sizes_mib"]:
        size = int(size_mib) * MIB
        host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        device = torch.empty(size, dtype=torch.uint8, device=model.device)
        host.fill_(int(size_mib) % 251)
        for _ in range(int(transfer["warmup_copies"])):
            device.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        for repetition in range(int(transfer["measured_copies"])):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            device.copy_(host, non_blocking=True)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            transfer_rows.append(
                {
                    "size_mib": int(size_mib),
                    "bytes": size,
                    "repetition": repetition,
                    "elapsed_ms": float(start.elapsed_time(end)),
                    "wall_completion_ms": wall_ms,
                }
            )
        del host, device

    _write_csv(output_dir / "decode_timing_samples.csv", timing_rows)
    _write_csv(output_dir / "transfer_samples.csv", transfer_rows)
    decode_values = [float(row["elapsed_ms"]) for row in timing_rows]
    transfer_fit = _fit_transfer(transfer_rows)
    by_size: dict[str, dict[str, float]] = {}
    for size_mib in transfer["sizes_mib"]:
        values = [
            float(row["elapsed_ms"])
            for row in transfer_rows
            if int(row["size_mib"]) == int(size_mib)
        ]
        by_size[str(int(size_mib))] = {
            "median_ms": statistics.median(values),
            "p10_ms": sorted(values)[max(0, int(0.10 * (len(values) - 1)))],
            "p90_ms": sorted(values)[int(0.90 * (len(values) - 1))],
        }
    expert_size = 12 * MIB
    exact_values = [
        float(row["elapsed_ms"])
        for row in transfer_rows
        if int(row["bytes"]) == expert_size
    ]
    if not exact_values:
        raise ValueError("transfer sweep must include the exact 12 MiB expert")
    result = {
        "state": "complete",
        "config_fingerprint": fingerprint,
        "environment": environment,
        "provenance": {
            "run_id": experiment_config["run_id"],
            "prompt_file": str(prompt_path),
            "prompt_file_sha256": prompt_sha256,
            "model_id": model_config["model_id"],
            "model_revision": model_config.get("revision"),
            "tokenizer_id": model_config["tokenizer_id"],
            "tokenizer_revision": model_config.get("tokenizer_revision"),
            "hooks_installed": False,
        },
        "decode": {
            "semantics": "hook_free_cached_token_full_forward",
            "samples": len(decode_values),
            "median_forward_ms": statistics.median(decode_values),
            "mean_forward_ms": statistics.fmean(decode_values),
            "p10_forward_ms": sorted(decode_values)[
                max(0, int(0.10 * (len(decode_values) - 1)))
            ],
            "p90_forward_ms": sorted(decode_values)[
                int(0.90 * (len(decode_values) - 1))
            ],
            "moe_layers": 16,
            "effective_inter_moe_layer_ms": statistics.median(decode_values) / 16,
        },
        "transfer": {
            "semantics": "pinned_host_to_device_async_single_copy",
            "samples": len(transfer_rows),
            "by_size_mib": by_size,
            "fit": transfer_fit,
            "exact_expert_bytes": expert_size,
            "exact_expert_median_ms": statistics.median(exact_values),
        },
    }
    write_json(output_dir / "measurement.json", result)
    return result
