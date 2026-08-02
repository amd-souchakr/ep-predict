#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from ep_predict.tracing.storage import write_json

ORIGINAL = Path("artifacts/datasets/h1-standard-small/prompts.jsonl")
POOL = Path(
    "artifacts/datasets/gpt-oss-milestone-f-confirmation-pool/"
    "prompts-48-per-domain.jsonl"
)
OUTPUT = Path("artifacts/datasets/gpt-oss-milestone-f-confirmation/prompts.jsonl")
MANIFEST = Path("artifacts/datasets/gpt-oss-milestone-f-confirmation/manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    original = _rows(ORIGINAL)
    pool = _rows(POOL)
    original_ids = {str(row["sample_id"]) for row in original}
    original_prompts = {
        hashlib.sha256(str(row["prompt"]).encode()).hexdigest() for row in original
    }
    selected = [
        row
        for row in pool
        if str(row["sample_id"]) not in original_ids
        and hashlib.sha256(str(row["prompt"]).encode()).hexdigest()
        not in original_prompts
    ]
    counts = Counter(str(row["domain"]) for row in selected)
    if len(selected) != 64 or set(counts.values()) != {16}:
        raise RuntimeError(f"expected exactly 16 fresh prompts per domain: {counts}")
    selected_ids = [str(row["sample_id"]) for row in selected]
    selected_hashes = [
        hashlib.sha256(str(row["prompt"]).encode()).hexdigest() for row in selected
    ]
    if len(set(selected_ids)) != 64 or len(set(selected_hashes)) != 64:
        raise RuntimeError("fresh confirmation prompts are not unique")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, OUTPUT)
    write_json(
        MANIFEST,
        {
            "name": "gpt-oss-milestone-f-confirmation",
            "selection_rule": (
                "materialize the first 48 pinned shuffled examples per domain with "
                "the original seed, then exclude every original sample ID and prompt hash"
            ),
            "original_file": str(ORIGINAL),
            "original_sha256": _sha256(ORIGINAL),
            "pool_file": str(POOL),
            "pool_sha256": _sha256(POOL),
            "output_file": str(OUTPUT),
            "output_sha256": _sha256(OUTPUT),
            "records": len(selected),
            "domains": dict(sorted(counts.items())),
            "sample_id_overlap_with_original": 0,
            "prompt_hash_overlap_with_original": 0,
            "sample_ids": selected_ids,
            "prompt_sha256": selected_hashes,
        },
    )
    print(json.dumps(json.loads(MANIFEST.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
