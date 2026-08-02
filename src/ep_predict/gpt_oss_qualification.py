from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class DispatchObservation:
    expert_ids: torch.Tensor
    weights: torch.Tensor
    gather_src: torch.Tensor
    gather_dst: torch.Tensor
    histogram: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_nbytes(tensor: Any) -> int | None:
    """Return physical bytes for torch or Triton-wrapped tensors when exposed."""
    if isinstance(tensor, torch.Tensor):
        return tensor.untyped_storage().nbytes()
    storage = getattr(tensor, "storage", None)
    data = getattr(storage, "data", None)
    if isinstance(data, torch.Tensor):
        return data.untyped_storage().nbytes()
    return None


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    raise AttributeError(f"none of {names!r} found on {type(value).__name__}")


def decode_dispatch_inputs(
    routing_data: Any,
    gather_idx: Any,
    *,
    num_tokens: int,
    top_k: int,
) -> DispatchObservation:
    """Decode the exact expert-major inputs consumed by the MXFP4 GEMMs.

    `routing_data.hist` defines the expert ID for each expert-major dispatch
    slot. `gather_idx.src_indx` maps those slots back to flattened token/gate
    positions. This uses dispatch inputs themselves, not router reconstruction.
    """
    # The pinned kernel names this `expt_hist`; the Transformers distributed
    # fallback uses the shorter `hist`. Both are the histogram consumed by the
    # grouped GEMM, not a histogram recomputed by the observer.
    histogram = _field(routing_data, "expt_hist", "hist").detach().to("cpu", torch.int64).reshape(-1)
    gate_weights = _field(routing_data, "gate_scal").detach().to("cpu").reshape(-1)
    gather_src = _field(gather_idx, "src_indx", "src_idx").detach().to("cpu", torch.int64).reshape(-1)
    gather_dst = _field(gather_idx, "dst_indx", "dst_idx").detach().to("cpu", torch.int64).reshape(-1)

    expected_slots = num_tokens * top_k
    if histogram.sum().item() != expected_slots:
        raise ValueError(f"dispatch histogram covers {histogram.sum().item()} slots, expected {expected_slots}")
    if gate_weights.numel() != expected_slots or gather_src.numel() != expected_slots:
        raise ValueError("dispatch tensor lengths do not match token_count * top_k")
    if not torch.equal(torch.sort(gather_src).values, torch.arange(expected_slots)):
        raise ValueError("gather source indices are not a complete permutation")
    if not torch.equal(torch.sort(gather_dst).values, torch.arange(expected_slots)):
        raise ValueError("gather destination indices are not a complete permutation")

    dispatch_ids = torch.repeat_interleave(torch.arange(histogram.numel()), histogram)
    flat_ids = torch.empty(expected_slots, dtype=torch.int64)
    flat_weights = torch.empty(expected_slots, dtype=gate_weights.dtype)
    flat_ids[gather_src] = dispatch_ids
    flat_weights[gather_src] = gate_weights
    return DispatchObservation(
        expert_ids=flat_ids.reshape(num_tokens, top_k),
        weights=flat_weights.reshape(num_tokens, top_k),
        gather_src=gather_src,
        gather_dst=gather_dst,
        histogram=histogram,
    )


def canonicalize_pairs(expert_ids: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort each token's ID/weight pairs by expert ID for order-independent parity."""
    order = torch.argsort(expert_ids.to(torch.int64), dim=-1, stable=True)
    return torch.gather(expert_ids.to(torch.int64), -1, order), torch.gather(weights, -1, order)


def compare_routes(
    expected_ids: torch.Tensor,
    expected_weights: torch.Tensor,
    consumed_ids: torch.Tensor,
    consumed_weights: torch.Tensor,
) -> dict[str, int | float]:
    expected_ids, expected_weights = canonicalize_pairs(expected_ids.detach().cpu(), expected_weights.detach().cpu())
    consumed_ids, consumed_weights = canonicalize_pairs(consumed_ids.detach().cpu(), consumed_weights.detach().cpu())
    if expected_ids.shape != consumed_ids.shape or expected_weights.shape != consumed_weights.shape:
        return {
            "id_mismatches": max(expected_ids.numel(), consumed_ids.numel()),
            "weight_mismatches": max(expected_weights.numel(), consumed_weights.numel()),
            "max_abs_weight_error": float("inf"),
        }
    id_bad = expected_ids.ne(consumed_ids)
    weight_error = (expected_weights.float() - consumed_weights.float()).abs()
    return {
        "id_mismatches": int(id_bad.sum().item()),
        "weight_mismatches": int(weight_error.gt(1e-6).sum().item()),
        "max_abs_weight_error": float(weight_error.max().item()) if weight_error.numel() else 0.0,
    }


def expected_from_logits(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    values, ids = torch.topk(logits, top_k, dim=-1)
    return ids, torch.softmax(values, dim=-1, dtype=values.dtype)


def dataclass_json(value: Any) -> dict[str, Any]:
    return asdict(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
