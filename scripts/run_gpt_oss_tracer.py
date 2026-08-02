#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ep_predict.gpt_oss_qualification import (
    compare_routes,
    decode_dispatch_inputs,
    expected_from_logits,
    sha256_file,
)
from ep_predict.gpt_oss_tracing import (
    canonical_pairs,
    compare_repeat_records,
    summarize_routing,
    validate_request_coverage,
)
from ep_predict.tracing.schema import TraceRecord
from ep_predict.tracing.storage import RequestTraceStore, write_json


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class DispatchTraceSession:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        run_id: str,
        dataset_name: str | None = None,
        expected_layers: int,
        top_k: int,
        weight_atol: float,
    ) -> None:
        self.model = model
        self.run_id = run_id
        self.dataset_name = dataset_name or run_id
        self.expected_layers = expected_layers
        self.top_k = top_k
        self.weight_atol = weight_atol
        self.handles: list[Any] = []
        self.active: dict[str, Any] | None = None
        self.expected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.records: list[TraceRecord] = []
        self.forward_index = 0
        self.layer_stats: dict[int, dict[str, int | float]] = defaultdict(
            lambda: {
                "expected_calls": 0,
                "dispatch_calls": 0,
                "tokens": 0,
                "pairs": 0,
                "id_mismatches": 0,
                "weight_mismatches": 0,
                "max_abs_weight_error": 0.0,
            }
        )

    def __enter__(self) -> Self:
        if len(self.model.model.layers) != self.expected_layers:
            raise RuntimeError(
                f"loaded {len(self.model.model.layers)} layers, expected {self.expected_layers}"
            )
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp

            def mlp_pre_hook(module, args, kwargs, layer_idx=layer_idx):
                if self.active is None:
                    return
                if layer_idx in self.expected:
                    raise RuntimeError(
                        f"layer {layer_idx} MLP called twice in one forward"
                    )
                hidden = args[0].reshape(-1, module.router.hidden_dim)
                logits = F.linear(hidden, module.router.weight, module.router.bias)
                self.expected[layer_idx] = tuple(
                    value.detach().clone()
                    for value in expected_from_logits(logits, module.router.top_k)
                )
                self.layer_stats[layer_idx]["expected_calls"] += 1

            def dispatch_pre_hook(module, args, kwargs, layer_idx=layer_idx):
                if self.active is None:
                    return
                if layer_idx not in self.expected:
                    raise RuntimeError(
                        f"layer {layer_idx} dispatch preceded independent route"
                    )
                token_ids = self.active["token_ids"]
                positions = self.active["positions"]
                observation = decode_dispatch_inputs(
                    args[1], args[2], num_tokens=len(token_ids), top_k=self.top_k
                )
                comparison = compare_routes(
                    *self.expected[layer_idx],
                    observation.expert_ids,
                    observation.weights,
                    weight_atol=self.weight_atol,
                )
                stats = self.layer_stats[layer_idx]
                stats["dispatch_calls"] += 1
                stats["tokens"] += len(token_ids)
                stats["pairs"] += len(token_ids) * self.top_k
                stats["id_mismatches"] += comparison["id_mismatches"]
                stats["weight_mismatches"] += comparison["weight_mismatches"]
                stats["max_abs_weight_error"] = max(
                    float(stats["max_abs_weight_error"]),
                    float(comparison["max_abs_weight_error"]),
                )
                ids, weights = canonical_pairs(
                    observation.expert_ids, observation.weights
                )
                ids = ids.cpu()
                weights = weights.float().cpu()
                for row, (token_id, position) in enumerate(zip(token_ids, positions)):
                    self.records.append(
                        TraceRecord(
                            run_id=self.run_id,
                            request_id=self.active["request_id"],
                            sample_id=self.active["sample_id"],
                            phase=self.active["phase"],
                            token_position=position,
                            input_token_id=token_id,
                            layer_id=layer_idx,
                            moe_layer_index=layer_idx,
                            selected_expert_ids=[
                                int(value) for value in ids[row].tolist()
                            ],
                            selected_expert_weights=[
                                float(value) for value in weights[row].tolist()
                            ],
                            batch_id=self.forward_index,
                            batch_size=1,
                            dataset_name=self.dataset_name,
                            domain=self.active["domain"],
                        )
                    )

            self.handles.extend(
                [
                    mlp.register_forward_pre_hook(mlp_pre_hook, with_kwargs=True),
                    mlp.experts.register_forward_pre_hook(
                        dispatch_pre_hook, with_kwargs=True
                    ),
                ]
            )
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start_request(self) -> None:
        self.records = []
        self.forward_index = 0
        self.layer_stats.clear()

    def start_forward(
        self,
        *,
        request_id: int,
        sample_id: str,
        domain: str,
        phase: str,
        token_ids: list[int],
        positions: list[int],
    ) -> None:
        if self.active is not None:
            raise RuntimeError("previous traced forward is still active")
        if len(token_ids) != len(positions):
            raise ValueError("token IDs and positions must align")
        self.expected = {}
        self.active = {
            "request_id": request_id,
            "sample_id": sample_id,
            "domain": domain,
            "phase": phase,
            "token_ids": token_ids,
            "positions": positions,
        }

    def finish_forward(self) -> None:
        if self.active is None:
            raise RuntimeError("no active traced forward")
        expected = set(range(self.expected_layers))
        if set(self.expected) != expected:
            raise RuntimeError(
                f"forward covered {len(self.expected)}/{self.expected_layers} independent routes"
            )
        self.active = None
        self.expected = {}
        self.forward_index += 1


def render_prompt(tokenizer: Any, prompt: str, current_date: str) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        current_date=current_date,
    )
    if not isinstance(rendered, torch.Tensor):
        rendered = rendered["input_ids"]
    if (
        not isinstance(rendered, torch.Tensor)
        or rendered.ndim != 2
        or rendered.shape[0] != 1
    ):
        raise TypeError("chat template did not return one rank-2 tensor")
    return rendered


def trace_request(
    model: torch.nn.Module,
    tokenizer: Any,
    tracer: DispatchTraceSession,
    *,
    request_id: int,
    sample: dict[str, Any],
    current_date: str,
    max_new_tokens: int,
    max_prompt_tokens: int | None = None,
    device: str,
) -> tuple[list[TraceRecord], dict[str, Any]]:
    input_ids_cpu = render_prompt(tokenizer, sample["prompt"], current_date)
    if max_prompt_tokens is not None and input_ids_cpu.numel() > max_prompt_tokens:
        raise ValueError(
            f"rendered prompt has {input_ids_cpu.numel()} tokens, exceeds "
            f"frozen limit {max_prompt_tokens}: {sample['sample_id']}"
        )
    input_ids = input_ids_cpu.to(device)
    prompt_ids = [int(value) for value in input_ids_cpu.reshape(-1).tolist()]
    attention_mask = torch.ones_like(input_ids)
    tracer.start_request()
    tracer.start_forward(
        request_id=request_id,
        sample_id=sample["sample_id"],
        domain=sample["domain"],
        phase="prefill",
        token_ids=prompt_ids,
        positions=list(range(len(prompt_ids))),
    )
    with torch.inference_mode():
        result = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
    tracer.finish_forward()
    past = result.past_key_values
    logits = result.logits[:, -1, :]
    generated: list[int] = []

    for index in range(max_new_tokens):
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_token)
        token = torch.tensor([[next_token]], device=device, dtype=input_ids.dtype)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )
        tracer.start_forward(
            request_id=request_id,
            sample_id=sample["sample_id"],
            domain=sample["domain"],
            phase="decode",
            token_ids=[next_token],
            positions=[len(prompt_ids) + index],
        )
        with torch.inference_mode():
            result = model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
        tracer.finish_forward()
        past = result.past_key_values
        logits = result.logits[:, -1, :]

    torch.cuda.synchronize()
    records = list(tracer.records)
    output = {
        "request_id": request_id,
        "sample_id": sample["sample_id"],
        "domain": sample["domain"],
        "prompt": sample["prompt"],
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": json_sha256(prompt_ids),
        "generated_token_count": len(generated),
        "generated_token_ids": generated,
        "generated_token_ids_sha256": json_sha256(generated),
        "generated_text_with_special_tokens": tokenizer.decode(
            generated, skip_special_tokens=False
        ),
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "terminal_decode_forward": True,
        "trace_records": len(records),
    }
    del result, past, logits
    return records, output


def model_inspection(
    model: torch.nn.Module, tokenizer: Any, config: dict[str, Any]
) -> dict[str, Any]:
    first = model.model.layers[0].mlp.experts
    device = next(model.parameters()).device
    return {
        "model_class": type(model).__name__,
        "expert_class": type(first).__name__,
        "mlp_forward_module": inspect.getmodule(
            model.model.layers[0].mlp.forward
        ).__name__,
        "routed_layers": len(model.model.layers),
        "experts_per_layer": int(model.config.num_local_experts),
        "top_k": int(model.config.num_experts_per_tok),
        "shared_experts": 0,
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
        "native_mxfp4": type(first).__name__ == "Mxfp4GptOssExperts",
        "geometry_matches_config": (
            len(model.model.layers) == config["trace"]["expected_layers"]
            and int(model.config.num_local_experts)
            == config["trace"]["expected_experts"]
            and int(model.config.num_experts_per_tok)
            == config["trace"]["expected_top_k"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/gpt-oss-20b-milestone-d.toml"),
    )
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    output_dir = Path(config["output_dir"])
    snapshot = Path(config["snapshot"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Milestone D requires exactly one visible accelerator")
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    torch.manual_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype="auto",
        device_map={"": config["device"]},
        use_kernels=False,
    ).eval()
    inspection = model_inspection(model, tokenizer, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "model_inspection.json", inspection)

    references: dict[int, tuple[list[TraceRecord], dict[str, Any]]] = {}
    repeat_comparisons: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    primary_records: list[TraceRecord] = []
    layer_integrity: list[dict[str, Any]] = []
    trace_store = RequestTraceStore(output_dir)

    with DispatchTraceSession(
        model,
        run_id=config["run_id"],
        expected_layers=config["trace"]["expected_layers"],
        top_k=config["trace"]["expected_top_k"],
        weight_atol=config["trace"]["dispatch_weight_atol"],
    ) as tracer:
        for repeat in range(config["repeat_count"]):
            for request_id, sample in enumerate(config["workload"]):
                records, output = trace_request(
                    model,
                    tokenizer,
                    tracer,
                    request_id=request_id,
                    sample=sample,
                    current_date=config["chat_template_current_date"],
                    max_new_tokens=config["max_new_tokens"],
                    device=config["device"],
                )
                request_coverage = validate_request_coverage(
                    records,
                    prompt_tokens=output["prompt_token_count"],
                    generated_tokens=output["generated_token_count"],
                    expected_layers=config["trace"]["expected_layers"],
                    expected_top_k=config["trace"]["expected_top_k"],
                )
                if repeat == 0:
                    references[request_id] = (records, output)
                    coverage.append(
                        {
                            "request_id": request_id,
                            "sample_id": sample["sample_id"],
                            **request_coverage,
                        }
                    )
                    primary_records.extend(records)
                    trace_store.write_request(request_id, sample["sample_id"], records)
                    for layer, stats in sorted(tracer.layer_stats.items()):
                        layer_integrity.append(
                            {
                                "request_id": request_id,
                                "sample_id": sample["sample_id"],
                                "layer": layer,
                                **stats,
                            }
                        )
                else:
                    reference_records, reference_output = references[request_id]
                    comparison = compare_repeat_records(
                        reference_records,
                        records,
                        weight_atol=config["trace"]["repeat_weight_atol"],
                    )
                    comparison.update(
                        {
                            "request_id": request_id,
                            "sample_id": sample["sample_id"],
                            "repeat": repeat,
                            "prompt_ids_identical": output["prompt_token_ids"]
                            == reference_output["prompt_token_ids"],
                            "generated_ids_identical": output["generated_token_ids"]
                            == reference_output["generated_token_ids"],
                            "repeat_coverage_complete": request_coverage["complete"],
                            "repeat_missing_token_layer_keys": request_coverage[
                                "missing_token_layer_keys"
                            ],
                            "repeat_unexpected_token_layer_keys": request_coverage[
                                "unexpected_token_layer_keys"
                            ],
                            "repeat_duplicate_token_layer_keys": request_coverage[
                                "duplicate_token_layer_keys"
                            ],
                        }
                    )
                    repeat_comparisons.append(comparison)

    outputs = [references[index][1] for index in sorted(references)]
    with (output_dir / "outputs.jsonl").open("w", encoding="utf-8") as handle:
        for output in outputs:
            handle.write(json.dumps(output, sort_keys=True) + "\n")

    routing_rows = summarize_routing(primary_records)
    for name, rows in (
        ("layer_integrity.csv", layer_integrity),
        ("routing_summary.csv", routing_rows),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    total_pairs = sum(int(row["pairs"]) for row in layer_integrity)
    integrity = {
        "schema_version": 1,
        "milestone": "D",
        "coverage": coverage,
        "repeat_comparisons": repeat_comparisons,
        "totals": {
            "requests": len(outputs),
            "prompt_tokens": sum(output["prompt_token_count"] for output in outputs),
            "generated_tokens": sum(
                output["generated_token_count"] for output in outputs
            ),
            "trace_records": len(primary_records),
            "dispatch_consumed_pairs": total_pairs,
            "dispatch_id_mismatches": sum(
                int(row["id_mismatches"]) for row in layer_integrity
            ),
            "dispatch_weight_mismatches": sum(
                int(row["weight_mismatches"]) for row in layer_integrity
            ),
            "dispatch_max_abs_weight_error": max(
                float(row["max_abs_weight_error"]) for row in layer_integrity
            ),
        },
    }
    checks = {
        "native_mxfp4_path": inspection["native_mxfp4"],
        "geometry_matches_frozen_config": inspection["geometry_matches_config"],
        "exact_generated_token_count": all(
            output["generated_token_count"] == config["max_new_tokens"]
            for output in outputs
        ),
        "complete_layer_token_coverage": all(row["complete"] for row in coverage),
        "zero_dispatch_id_mismatches": integrity["totals"]["dispatch_id_mismatches"]
        == 0,
        "zero_dispatch_weight_mismatches": integrity["totals"][
            "dispatch_weight_mismatches"
        ]
        == 0,
        "dispatch_weight_tolerance": integrity["totals"][
            "dispatch_max_abs_weight_error"
        ]
        <= config["trace"]["dispatch_weight_atol"],
        "repeat_input_ids_identical": all(
            row["prompt_ids_identical"] for row in repeat_comparisons
        ),
        "repeat_output_ids_identical": all(
            row["generated_ids_identical"] for row in repeat_comparisons
        ),
        "repeat_routes_identical": all(row["identical"] for row in repeat_comparisons),
        "repeat_layer_token_coverage": all(
            row["repeat_coverage_complete"] for row in repeat_comparisons
        ),
        "repeat_weight_tolerance": all(
            row["max_abs_weight_error"] <= config["trace"]["repeat_weight_atol"]
            for row in repeat_comparisons
        ),
    }
    integrity["gate_checks"] = checks
    integrity["decision"] = (
        "TRACE_QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED"
    )
    write_json(output_dir / "integrity.json", integrity)

    run_definition = {
        "schema_version": 1,
        "config": config,
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "checkpoint": {
            "model_id": config["model_id"],
            "revision": config["revision"],
            "snapshot": str(snapshot),
            "config_sha256": sha256_file(snapshot / "config.json"),
            "tokenizer_json_sha256": sha256_file(snapshot / "tokenizer.json"),
            "safetensors_index_sha256": sha256_file(
                snapshot / "model.safetensors.index.json"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "torch_hip": torch.version.hip,
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "kernels": package_version("kernels"),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "visible_device_count": torch.cuda.device_count(),
        },
        "git_commit": git_commit(),
        "milestone_c_qualification": {
            "path": "artifacts/runs/gpt-oss-20b-milestone-c/qualification.json",
            "sha256": sha256_file(
                Path("artifacts/runs/gpt-oss-20b-milestone-c/qualification.json")
            ),
        },
        "claim_boundary": "end-to-end tracing workflow qualification only; no routing-distribution comparison or timing claim",
    }
    write_json(output_dir / "run_definition.json", run_definition)
    print(
        json.dumps(
            {
                "decision": integrity["decision"],
                "totals": integrity["totals"],
                "gate_checks": checks,
            },
            indent=2,
        )
    )
    if integrity["decision"] != "TRACE_QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
