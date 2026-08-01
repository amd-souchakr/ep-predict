from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ep_predict.tracing.schema import TraceRecord


class RequestTraceStore:
    """One atomic gzip member per request makes collection cheaply resumable."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.trace_dir = self.run_dir / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_sample_id(sample_id: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in sample_id
        )

    def path_for(self, request_id: int, sample_id: str) -> Path:
        safe_id = self._safe_sample_id(sample_id)
        return self.trace_dir / f"request-{request_id:05d}-{safe_id}.jsonl.gz"

    def completed(self, request_id: int, sample_id: str) -> bool:
        return self.path_for(request_id, sample_id).is_file()

    def write_request(
        self,
        request_id: int,
        sample_id: str,
        records: Iterable[TraceRecord],
    ) -> Path:
        destination = self.path_for(request_id, sample_id)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True)
                )
                handle.write("\n")
        os.replace(temporary, destination)
        return destination


def iter_trace_records(run_dir: str | Path) -> Iterator[dict[str, Any]]:
    paths = sorted((Path(run_dir) / "trace").glob("request-*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no request traces found under {Path(run_dir) / 'trace'}")
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
