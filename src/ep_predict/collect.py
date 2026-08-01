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
from ep_predict.tracing.hooks import RouterTracer
from ep_predict.tracing.schema import RequestContext, TRACE_SCHEMA_VERSION
from ep_predict.tracing.storage import RequestTraceStore, write_json


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


def _tokenize_prompt(tokenizer: Any, prompt: str, max_tokens: int):
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

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{run_dir} belongs to a different configuration; choose a new run_id"
            )

    prompts = _load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]

    seed = int(experiment_config.get("seed", 0))
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
    ) as tracer:
        for request_id, prompt_record in enumerate(prompts):
            sample_id = prompt_record["sample_id"]
            if store.completed(request_id, sample_id):
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
            )
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
            records, summary = tracer.finish_request()
            trace_path = store.write_request(request_id, sample_id, records)
            request_summaries.append(
                {
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "domain": prompt_record["domain"],
                    "state": "complete",
                    "trace": str(trace_path),
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
        "model_report_path": str(run_dir / "model_report.json"),
        "environment": environment_report(),
        "requests": request_summaries,
    }
    write_json(manifest_path, manifest)
    return manifest
