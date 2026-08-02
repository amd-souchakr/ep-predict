#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from run_gpt_oss_tracer import DispatchTraceSession, model_inspection, trace_request
from transformers import AutoModelForCausalLM, AutoTokenizer

from ep_predict.gpt_oss_qualification import sha256_file
from ep_predict.gpt_oss_tracing import summarize_routing, validate_request_coverage
from ep_predict.tracing.storage import RequestTraceStore, write_json


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_workload(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(config["prompt_file"])
    if sha256_file(path) != config["prompt_file_sha256"]:
        raise ValueError("prompt file SHA-256 does not match the frozen config")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    domains = Counter(str(row["domain"]) for row in rows)
    if len(rows) != int(config["expected_requests"]):
        raise ValueError(
            f"loaded {len(rows)} requests, expected {config['expected_requests']}"
        )
    if len(domains) != int(config["expected_domains"]):
        raise ValueError(
            f"loaded {len(domains)} domains, expected {config['expected_domains']}"
        )
    if set(domains.values()) != {int(config["expected_requests_per_domain"])}:
        raise ValueError(f"unbalanced frozen workload: {dict(domains)}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/gpt-oss-20b-milestone-e.toml"),
    )
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    snapshot = Path(config["snapshot"])
    workload = _load_workload(config)
    for path_key, hash_key in (
        ("prompt_manifest", "prompt_manifest_sha256"),
        ("predictor_checkpoint", "predictor_checkpoint_sha256"),
        ("predictor_config", "predictor_config_sha256"),
    ):
        if path_key in config and sha256_file(Path(config[path_key])) != config[hash_key]:
            raise ValueError(f"{path_key} SHA-256 does not match the frozen config")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Milestone E requires exactly one visible accelerator")
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    if (output_dir / "trace").exists():
        raise FileExistsError(
            f"refusing to mix a new collection with existing traces: {output_dir / 'trace'}"
        )
    torch.manual_seed(int(config["seed"]))

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
    write_json(
        output_dir / "model_report.json",
        {
            "model_id": config["model_id"],
            "revision": config["revision"],
            "routers": [
                {
                    "layer_id": layer,
                    "moe_layer_index": layer,
                    "num_experts": inspection["experts_per_layer"],
                    "top_k": inspection["top_k"],
                }
                for layer in range(inspection["routed_layers"])
            ],
        },
    )

    outputs: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    layer_integrity: list[dict[str, Any]] = []
    routing_records = []
    trace_store = RequestTraceStore(output_dir)
    with DispatchTraceSession(
        model,
        run_id=config["run_id"],
        dataset_name=config["run_id"],
        expected_layers=int(config["trace"]["expected_layers"]),
        top_k=int(config["trace"]["expected_top_k"]),
        weight_atol=float(config["trace"]["dispatch_weight_atol"]),
    ) as tracer:
        for request_id, sample in enumerate(workload):
            records, output = trace_request(
                model,
                tokenizer,
                tracer,
                request_id=request_id,
                sample=sample,
                current_date=config["chat_template_current_date"],
                max_new_tokens=int(config["max_new_tokens"]),
                max_prompt_tokens=int(config["max_prompt_tokens"]),
                device=config["device"],
            )
            request_coverage = validate_request_coverage(
                records,
                prompt_tokens=output["prompt_token_count"],
                generated_tokens=output["generated_token_count"],
                expected_layers=int(config["trace"]["expected_layers"]),
                expected_top_k=int(config["trace"]["expected_top_k"]),
            )
            if not request_coverage["complete"]:
                raise RuntimeError(
                    f"incomplete trace for {sample['sample_id']}: {request_coverage}"
                )
            trace_store.write_request(request_id, sample["sample_id"], records)
            outputs.append(output)
            coverage.append(
                {
                    "request_id": request_id,
                    "sample_id": sample["sample_id"],
                    **request_coverage,
                }
            )
            routing_records.extend(records)
            for layer, stats in sorted(tracer.layer_stats.items()):
                layer_integrity.append(
                    {
                        "request_id": request_id,
                        "sample_id": sample["sample_id"],
                        "layer": layer,
                        **stats,
                    }
                )
            print(
                json.dumps(
                    {
                        "request": request_id + 1,
                        "of": len(workload),
                        "sample_id": sample["sample_id"],
                        "prompt_tokens": output["prompt_token_count"],
                    }
                ),
                flush=True,
            )

    with (output_dir / "outputs.jsonl").open("w", encoding="utf-8") as handle:
        for output in outputs:
            handle.write(json.dumps(output, sort_keys=True) + "\n")
    _write_csv(output_dir / "layer_integrity.csv", layer_integrity)
    _write_csv(output_dir / "routing_summary.csv", summarize_routing(routing_records))
    total_pairs = sum(int(row["pairs"]) for row in layer_integrity)
    totals = {
        "requests": len(outputs),
        "prompt_tokens": sum(int(row["prompt_token_count"]) for row in outputs),
        "generated_tokens": sum(int(row["generated_token_count"]) for row in outputs),
        "trace_records": sum(int(row["trace_records"]) for row in outputs),
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
    }
    checks = {
        "native_mxfp4_path": bool(inspection["native_mxfp4"]),
        "geometry_matches_frozen_config": bool(inspection["geometry_matches_config"]),
        "exact_request_count": len(outputs) == int(config["expected_requests"]),
        "exact_generated_token_count": all(
            int(row["generated_token_count"]) == int(config["max_new_tokens"])
            for row in outputs
        ),
        "prompt_bound_respected": all(
            int(row["prompt_token_count"]) <= int(config["max_prompt_tokens"])
            for row in outputs
        ),
        "complete_layer_token_coverage": all(row["complete"] for row in coverage),
        "zero_dispatch_id_mismatches": totals["dispatch_id_mismatches"] == 0,
        "zero_dispatch_weight_mismatches": totals["dispatch_weight_mismatches"] == 0,
        "dispatch_weight_tolerance": totals["dispatch_max_abs_weight_error"]
        <= float(config["trace"]["dispatch_weight_atol"]),
    }
    integrity = {
        "schema_version": 1,
        "milestone": str(config.get("milestone", "E")),
        "coverage": coverage,
        "totals": totals,
        "gate_checks": checks,
        "decision": "TRACE_COMPLETE" if all(checks.values()) else "TRACE_INVALID",
    }
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
        "prompt_file": {
            "path": config["prompt_file"],
            "sha256": config["prompt_file_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "torch_hip": torch.version.hip,
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "kernels": _package_version("kernels"),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "visible_device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        },
        "git_commit": _git_commit(),
        "milestone_d_result": {
            "path": "artifacts/runs/gpt-oss-20b-milestone-d/result.json",
            "sha256": sha256_file(
                Path("artifacts/runs/gpt-oss-20b-milestone-d/result.json")
            ),
        },
        "claim_boundary": str(
            config.get(
                "claim_boundary",
                "GPT-OSS 20B routing and held-out route-prediction quality only; "
                "no timing, language-quality, 120B, or cross-model claim",
            )
        ),
    }
    write_json(output_dir / "run_definition.json", run_definition)
    print(json.dumps({"decision": integrity["decision"], "totals": totals}, indent=2))
    if integrity["decision"] != "TRACE_COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
