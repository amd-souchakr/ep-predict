from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ep_predict.config import config_fingerprint
from ep_predict.modeling import (
    environment_report,
    inspect_loaded_model,
    load_model_and_tokenizer,
)
from ep_predict.tracing.hooks import RouterInputProjector, RouterTracer
from ep_predict.tracing.schema import RequestContext, TRACE_SCHEMA_VERSION
from ep_predict.tracing.storage import (
    RequestFeatureStore,
    RequestTraceStore,
    write_json,
)


def _load_prompts(path: str | Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"sample_id", "domain", "prompt"} - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields {sorted(missing)}")
            prompts.append(record)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    *,
    prompt_format: str = "auto",
):
    if prompt_format not in {"auto", "raw"}:
        raise ValueError("prompt_format must be 'auto' or 'raw'")
    if prompt_format == "raw":
        return tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
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


def _input_ids_report(encoded: dict[str, Any]) -> dict[str, Any]:
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise ValueError("tokenizer output does not contain input_ids")
    flattened = [int(value) for value in input_ids.detach().cpu().reshape(-1).tolist()]
    serialized = ",".join(str(value) for value in flattened).encode("ascii")
    return {
        "input_token_count": len(flattened),
        "input_token_ids_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def collect_run(
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    import torch

    run_id = experiment_config["run_id"]
    run_dir = Path(experiment_config["output_dir"])
    prompt_path = Path(experiment_config["prompt_file"])
    trace_config = experiment_config.get("trace", {})
    prompt_file_sha256 = _file_sha256(prompt_path)
    fingerprint = config_fingerprint(
        model_config,
        experiment_config,
        {"prompt_file_sha256": prompt_file_sha256},
    )
    manifest_path = run_dir / "run_manifest.json"
    definition_path = run_dir / "run_definition.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    if definition_path.exists():
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        if definition.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{run_dir} contains traces from a different configuration; "
                "choose a new run_id"
            )
    else:
        write_json(
            definition_path,
            {
                "run_id": run_id,
                "config_fingerprint": fingerprint,
                "model_config": model_config,
                "experiment_config": experiment_config,
                "prompt_file_sha256": prompt_file_sha256,
            },
        )

    existing_request_summaries: dict[tuple[int, str], dict[str, Any]] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{run_dir} belongs to a different configuration; choose a new run_id"
            )
        existing_request_summaries = {
            (int(summary["request_id"]), str(summary["sample_id"])): summary
            for summary in existing.get("requests", [])
        }

    prompts = _load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]

    seed = int(experiment_config.get("seed", 0))
    prompt_format = str(experiment_config.get("prompt_format", "auto"))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_model_and_tokenizer(model_config)
    model_report, routers = inspect_loaded_model(
        model,
        router_name_contains=model_config.get("router_name_contains"),
    )
    write_json(run_dir / "model_report.json", model_report)

    feature_config = experiment_config.get("hidden_features", {})
    features_enabled = bool(feature_config.get("enabled", False))
    feature_projector: RouterInputProjector | None = None
    feature_store: RequestFeatureStore | None = None
    if features_enabled:
        if feature_config.get("point", "router_input") != "router_input":
            raise ValueError("only the explicit router_input hook point is supported")
        hidden_size = model_report.get("hidden_size")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("model report does not expose a valid hidden size")
        feature_projector = RouterInputProjector(
            input_dimension=hidden_size,
            output_dimension=int(feature_config["dimension"]),
            seed=int(feature_config["projection_seed"]),
            storage_dtype=str(feature_config.get("storage_dtype", "float16")),
        )
        feature_store = RequestFeatureStore(run_dir)
        write_json(run_dir / "projection_report.json", feature_projector.report())

    store = RequestTraceStore(run_dir)
    request_summaries: list[dict[str, Any]] = []
    with RouterTracer(
        model,
        routers,
        fail_on_router_mismatch=bool(
            trace_config.get("fail_on_router_mismatch", True)
        ),
        fail_on_missing_router=bool(
            trace_config.get("fail_on_missing_router", True)
        ),
        feature_projector=feature_projector,
    ) as tracer:
        for request_id, prompt_record in enumerate(prompts):
            sample_id = prompt_record["sample_id"]
            trace_complete = store.completed(request_id, sample_id)
            feature_complete = (
                feature_store.completed(request_id, sample_id)
                if feature_store is not None
                else True
            )
            if trace_complete and feature_complete:
                previous_summary = existing_request_summaries.get(
                    (request_id, sample_id)
                )
                if previous_summary is not None:
                    request_summaries.append(
                        {
                            **previous_summary,
                            "state": "already_complete",
                        }
                    )
                else:
                    request_summaries.append(
                        {
                            "request_id": request_id,
                            "sample_id": sample_id,
                            "state": "already_complete",
                        }
                    )
                continue

            encoded = _tokenize_prompt(
                tokenizer,
                prompt_record["prompt"],
                int(experiment_config.get("max_prompt_tokens", 384)),
                prompt_format=prompt_format,
            )
            input_report = _input_ids_report(encoded)
            encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
            tracer.start_request(
                RequestContext(
                    run_id=run_id,
                    request_id=request_id,
                    sample_id=sample_id,
                    dataset_name=prompt_record.get("dataset_name", prompt_path.stem),
                    domain=prompt_record["domain"],
                )
            )
            generation_args: dict[str, Any] = {
                "max_new_tokens": int(experiment_config.get("max_new_tokens", 64)),
                "do_sample": bool(experiment_config.get("do_sample", False)),
                "use_cache": bool(model_config.get("use_cache", True)),
                "pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            }
            if generation_args["do_sample"]:
                generation_args.update(
                    {
                        "temperature": float(experiment_config.get("temperature", 1.0)),
                        "top_p": float(experiment_config.get("top_p", 1.0)),
                    }
                )
            with torch.inference_mode():
                model.generate(**encoded, **generation_args)
            records, hidden_features, summary = tracer.finish_request()
            feature_path: Path | None = None
            if feature_store is not None:
                if hidden_features is None:
                    raise RuntimeError("feature collection enabled but no features exist")
                feature_path = feature_store.write_request(
                    request_id,
                    sample_id,
                    records,
                    hidden_features,
                )
            trace_path = store.write_request(request_id, sample_id, records)
            request_summaries.append(
                {
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "domain": prompt_record["domain"],
                    "state": "complete",
                    "trace": str(trace_path),
                    "features": str(feature_path) if feature_path else None,
                    "prompt_format": prompt_format,
                    **input_report,
                    **summary,
                }
            )
            print(
                f"[{request_id + 1}/{len(prompts)}] {sample_id}: "
                f"{summary['records']} records, "
                f"{summary['model_forward_calls']} forwards"
            )

    manifest = {
        "run_id": run_id,
        "state": "complete",
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "config_fingerprint": fingerprint,
        "model_config": model_config,
        "experiment_config": experiment_config,
        "prompt_file_sha256": prompt_file_sha256,
        "prompt_format": prompt_format,
        "model_report_path": str(run_dir / "model_report.json"),
        "projection_report": (
            feature_projector.report() if feature_projector is not None else None
        ),
        "environment": environment_report(),
        "requests": request_summaries,
    }
    write_json(manifest_path, manifest)
    return manifest
