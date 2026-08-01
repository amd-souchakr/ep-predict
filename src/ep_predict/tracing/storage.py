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


class RequestFeatureStore:
    """Atomic numeric NPZ feature shard aligned one-to-one with trace records."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.feature_dir = self.run_dir / "features"
        self.feature_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, request_id: int, sample_id: str) -> Path:
        safe_id = RequestTraceStore._safe_sample_id(sample_id)
        return self.feature_dir / f"request-{request_id:05d}-{safe_id}.npz"

    def completed(self, request_id: int, sample_id: str) -> bool:
        return self.path_for(request_id, sample_id).is_file()

    def write_request(
        self,
        request_id: int,
        sample_id: str,
        records: list[TraceRecord],
        hidden_features: Any,
    ) -> Path:
        import numpy as np

        features = hidden_features.detach().cpu().numpy()
        if features.ndim != 2 or len(features) != len(records):
            raise ValueError("feature matrix must align one-to-one with records")
        phase_codes = {"prefill": 0, "decode": 1}
        try:
            phases = np.asarray(
                [phase_codes[record.phase] for record in records],
                dtype=np.uint8,
            )
        except KeyError as error:
            raise ValueError(f"unsupported trace phase {error.args[0]!r}") from error

        destination = self.path_for(request_id, sample_id)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                hidden_feature=features.astype(np.float16, copy=False),
                request_id=np.asarray(request_id, dtype=np.int64),
                sample_id=np.asarray(sample_id),
                phase=phases,
                token_position=np.asarray(
                    [record.token_position for record in records],
                    dtype=np.int32,
                ),
                input_token_id=np.asarray(
                    [record.input_token_id for record in records],
                    dtype=np.int32,
                ),
                layer_id=np.asarray(
                    [record.layer_id for record in records],
                    dtype=np.int16,
                ),
                moe_layer_index=np.asarray(
                    [record.moe_layer_index for record in records],
                    dtype=np.int16,
                ),
            )
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
