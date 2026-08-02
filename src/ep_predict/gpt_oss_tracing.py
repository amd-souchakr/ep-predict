from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import torch

from ep_predict.tracing.schema import TraceRecord


def canonical_pairs(
    expert_ids: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(expert_ids.to(torch.int64), dim=-1, stable=True)
    return (
        torch.gather(expert_ids.to(torch.int64), -1, order),
        torch.gather(weights, -1, order),
    )


def validate_request_coverage(
    records: list[TraceRecord],
    *,
    prompt_tokens: int,
    generated_tokens: int,
    expected_layers: int,
    expected_top_k: int,
) -> dict[str, Any]:
    expected_tokens = prompt_tokens + generated_tokens
    keys = [(record.token_position, record.moe_layer_index) for record in records]
    key_counts = Counter(keys)
    expected_keys = {
        (position, layer)
        for position in range(expected_tokens)
        for layer in range(expected_layers)
    }
    observed_keys = set(keys)
    bad_width = sum(
        len(record.selected_expert_ids) != expected_top_k
        or len(record.selected_expert_weights) != expected_top_k
        for record in records
    )
    bad_phase = sum(
        record.phase
        != ("prefill" if record.token_position < prompt_tokens else "decode")
        for record in records
    )
    return {
        "expected_tokens": expected_tokens,
        "observed_token_positions": len({record.token_position for record in records}),
        "expected_records": expected_tokens * expected_layers,
        "observed_records": len(records),
        "missing_token_layer_keys": len(expected_keys - observed_keys),
        "unexpected_token_layer_keys": len(observed_keys - expected_keys),
        "duplicate_token_layer_keys": sum(
            count - 1 for count in key_counts.values() if count > 1
        ),
        "bad_top_k_width_records": bad_width,
        "bad_phase_records": bad_phase,
        "complete": (
            observed_keys == expected_keys
            and len(records) == len(expected_keys)
            and bad_width == 0
            and bad_phase == 0
        ),
    }


def compare_repeat_records(
    reference: list[TraceRecord],
    repeated: list[TraceRecord],
    *,
    weight_atol: float,
) -> dict[str, Any]:
    def keyed(records: list[TraceRecord]) -> dict[tuple[str, int, int], TraceRecord]:
        return {
            (record.phase, record.token_position, record.moe_layer_index): record
            for record in records
        }

    left = keyed(reference)
    right = keyed(repeated)
    reference_duplicate_keys = len(reference) - len(left)
    repeated_duplicate_keys = len(repeated) - len(right)
    common = sorted(left.keys() & right.keys())
    id_mismatches = 0
    weight_mismatches = 0
    max_abs_weight_error = 0.0
    token_id_mismatches = 0
    for key in common:
        a, b = left[key], right[key]
        token_id_mismatches += int(a.input_token_id != b.input_token_id)
        id_mismatches += sum(
            x != y for x, y in zip(a.selected_expert_ids, b.selected_expert_ids)
        )
        if len(a.selected_expert_ids) != len(b.selected_expert_ids):
            id_mismatches += abs(
                len(a.selected_expert_ids) - len(b.selected_expert_ids)
            )
        errors = [
            abs(x - y)
            for x, y in zip(a.selected_expert_weights, b.selected_expert_weights)
        ]
        weight_mismatches += sum(error > weight_atol for error in errors)
        if errors:
            max_abs_weight_error = max(max_abs_weight_error, max(errors))
    return {
        "reference_keys": len(left),
        "repeated_keys": len(right),
        "reference_records": len(reference),
        "repeated_records": len(repeated),
        "reference_duplicate_keys": reference_duplicate_keys,
        "repeated_duplicate_keys": repeated_duplicate_keys,
        "missing_keys": len(left.keys() - right.keys()),
        "unexpected_keys": len(right.keys() - left.keys()),
        "token_id_mismatches": token_id_mismatches,
        "expert_id_mismatches": id_mismatches,
        "selected_weight_mismatches": weight_mismatches,
        "max_abs_weight_error": max_abs_weight_error,
        "identical": (
            left.keys() == right.keys()
            and len(reference) == len(repeated)
            and reference_duplicate_keys == 0
            and repeated_duplicate_keys == 0
            and token_id_mismatches == 0
            and id_mismatches == 0
            and weight_mismatches == 0
        ),
    }


def summarize_routing(records: Iterable[TraceRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[TraceRecord]] = defaultdict(list)
    for record in records:
        groups[(record.phase, record.moe_layer_index)].append(record)

    rows: list[dict[str, Any]] = []
    for (phase, layer), group in sorted(groups.items()):
        counts: Counter[int] = Counter()
        entropies: list[float] = []
        max_weights: list[float] = []
        for record in group:
            counts.update(record.selected_expert_ids)
            weights = torch.tensor(record.selected_expert_weights, dtype=torch.float64)
            entropies.append(
                float(-(weights * weights.clamp_min(1e-30).log()).sum().item())
            )
            max_weights.append(float(weights.max().item()))
        top_expert, top_count = counts.most_common(1)[0]
        selections = sum(counts.values())
        rows.append(
            {
                "phase": phase,
                "layer": layer,
                "tokens": len(group),
                "selections": selections,
                "unique_experts": len(counts),
                "top_expert": top_expert,
                "top_expert_count": top_count,
                "top_expert_share": top_count / selections,
                "mean_selected_entropy": sum(entropies) / len(entropies),
                "mean_max_selected_weight": sum(max_weights) / len(max_weights),
            }
        )
    return rows
