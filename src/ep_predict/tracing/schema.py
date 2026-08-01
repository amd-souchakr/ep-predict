from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RequestContext:
    run_id: str
    request_id: int
    sample_id: str
    dataset_name: str
    domain: str


@dataclass(frozen=True)
class TraceRecord:
    run_id: str
    request_id: int
    sample_id: str
    phase: str
    token_position: int
    input_token_id: int
    layer_id: int
    moe_layer_index: int
    selected_expert_ids: list[int]
    selected_expert_weights: list[float]
    batch_id: int
    batch_size: int
    dataset_name: str
    domain: str
    metadata_version: int = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
