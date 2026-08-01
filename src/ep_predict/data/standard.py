from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


def _normalized_text(value: Any, *, preserve_whitespace: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    if preserve_whitespace:
        return value.strip()
    return " ".join(value.split())


def _extract_source_value(source: dict[str, Any], row: dict[str, Any]) -> Any:
    value = row.get(source["field"])
    if source.get("value_mode") == "first":
        if not isinstance(value, (list, tuple)) or not value:
            return ""
        return value[0]
    return value


def _source_sample_id(source: dict[str, Any], row: dict[str, Any], index: int) -> str:
    id_field = source.get("id_field")
    if id_field and row.get(id_field) is not None:
        identifier = str(row[id_field])
    else:
        identifier = f"{index:05d}"
    safe_identifier = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in identifier
    )
    return f"{source['key']}-{safe_identifier}"


def materialize_standard_workload(config: dict[str, Any]) -> dict[str, Any]:
    """Create a deterministic, reviewable JSONL workload from pinned datasets."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "dataset preparation requires: uv sync --extra data"
        ) from error

    output_path = Path(config["output_file"])
    manifest_path = Path(config["manifest_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    sample_count = int(config.get("samples_per_domain", 32))
    shuffle_buffer = int(config.get("shuffle_buffer", 10_000))
    records: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []

    for source_index, source in enumerate(config["sources"]):
        load_args: dict[str, Any] = {
            "path": source["dataset_name"],
            "split": source["split"],
            "revision": source["revision"],
            "streaming": True,
        }
        if source.get("subset"):
            load_args["name"] = source["subset"]
        dataset = load_dataset(**load_args)
        shuffled = dataset.shuffle(
            seed=seed + source_index,
            buffer_size=shuffle_buffer,
        )

        accepted = 0
        inspected = 0
        seen_text: set[str] = set()
        prompt_lengths: list[int] = []
        for row in shuffled:
            inspected += 1
            text = _normalized_text(
                _extract_source_value(source, row),
                preserve_whitespace=bool(source.get("preserve_whitespace", False)),
            )
            if len(text) < int(source.get("min_chars", 1)):
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen_text:
                continue
            seen_text.add(digest)
            prompt = f"{source.get('prefix', '')}{text}"
            records.append(
                {
                    "sample_id": _source_sample_id(source, row, accepted),
                    "domain": source["domain"],
                    "prompt": prompt,
                    "dataset_name": source["dataset_name"],
                    "dataset_subset": source.get("subset") or None,
                    "dataset_split": source["split"],
                    "dataset_revision": source["revision"],
                    "source_key": source["key"],
                }
            )
            prompt_lengths.append(len(prompt))
            accepted += 1
            if accepted == sample_count:
                break
        if accepted != sample_count:
            raise RuntimeError(
                f"{source['key']} yielded only {accepted}/{sample_count} usable rows"
            )
        source_reports.append(
            {
                "key": source["key"],
                "dataset_name": source["dataset_name"],
                "subset": source.get("subset") or None,
                "split": source["split"],
                "revision": source["revision"],
                "domain": source["domain"],
                "accepted": accepted,
                "inspected": inspected,
                "prompt_chars_min": min(prompt_lengths),
                "prompt_chars_mean": round(
                    sum(prompt_lengths) / len(prompt_lengths), 1
                ),
                "prompt_chars_max": max(prompt_lengths),
            }
        )

    # Interleave domains deterministically so windows do not coincide with one
    # source. Per-domain analysis still uses the explicit domain field.
    records.sort(
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['sample_id']}".encode()
        ).hexdigest()
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, output_path)

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "name": "h1-standard-small",
        "seed": seed,
        "samples_per_domain": sample_count,
        "record_count": len(records),
        "output_file": str(output_path),
        "output_sha256": digest,
        "sources": source_reports,
    }
    write_json(manifest_path, manifest)
    return manifest
