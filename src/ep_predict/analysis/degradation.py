from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ep_predict.analysis.architecture import (
    GIB,
    MIB,
    _correlated_complete_mask,
    _percentile,
    _transfer_ms,
)
from ep_predict.tracing.storage import iter_trace_records, write_json


@dataclass(frozen=True)
class WeightedRoutes:
    token_count: int
    layers: int
    top_k: int
    request_ids: np.ndarray
    domains: np.ndarray
    token_indices: np.ndarray
    layer_ids: np.ndarray
    expert_ids: np.ndarray
    raw_weights: np.ndarray
    normalized_weights: np.ndarray


@dataclass(frozen=True)
class CandidateProfile:
    predicted_mask: np.ndarray
    service_order: np.ndarray
    useful_counts: np.ndarray
    resident_normalized_mass: np.ndarray
    resident_raw_mass: np.ndarray
    ordered_normalized_weights: np.ndarray
    ordered_raw_weights: np.ndarray
    complete_mask: np.ndarray


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_weighted_decode(run: Path, expected_layers: int) -> WeightedRoutes:
    grouped: dict[
        tuple[int, str, str, int],
        dict[int, tuple[list[int], list[float]]],
    ] = defaultdict(dict)
    for record in iter_trace_records(run):
        if str(record["phase"]) != "decode":
            continue
        key = (
            int(record["request_id"]),
            str(record["sample_id"]),
            str(record["domain"]),
            int(record["token_position"]),
        )
        layer = int(record["layer_id"])
        if layer in grouped[key]:
            raise ValueError(f"duplicate weighted route for {key}, layer {layer}")
        grouped[key][layer] = (
            [int(value) for value in record["selected_expert_ids"]],
            [float(value) for value in record["selected_expert_weights"]],
        )
    keys = sorted(grouped, key=lambda key: (key[0], key[3]))
    if not keys:
        raise ValueError("trace has no decode routes")
    top_ks = {
        len(values[0])
        for layers in grouped.values()
        for values in layers.values()
    }
    if len(top_ks) != 1:
        raise ValueError("AX4 requires uniform routing top-k")
    top_k = top_ks.pop()

    request_ids: list[int] = []
    domains: list[str] = []
    token_indices: list[int] = []
    layer_ids: list[int] = []
    expert_ids: list[list[int]] = []
    raw_weights: list[list[float]] = []
    for token_index, key in enumerate(keys):
        routes = grouped[key]
        if sorted(routes) != list(range(expected_layers)):
            raise ValueError(f"incomplete decode token {key}: {sorted(routes)}")
        for layer in range(expected_layers):
            ids, weights = routes[layer]
            if len(ids) != top_k or len(weights) != top_k:
                raise ValueError("selected IDs and weights do not align")
            if len(set(ids)) != top_k:
                raise ValueError("duplicate expert in selected route")
            if any(weight < 0 for weight in weights):
                raise ValueError("negative router weight")
            request_ids.append(key[0])
            domains.append(key[2])
            token_indices.append(token_index)
            layer_ids.append(layer)
            expert_ids.append(ids)
            raw_weights.append(weights)

    raw = np.asarray(raw_weights, dtype=np.float64)
    sums = raw.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("selected route has zero weight sum")
    return WeightedRoutes(
        token_count=len(keys),
        layers=expected_layers,
        top_k=top_k,
        request_ids=np.asarray(request_ids, dtype=np.int32),
        domains=np.asarray(domains),
        token_indices=np.asarray(token_indices, dtype=np.int32),
        layer_ids=np.asarray(layer_ids, dtype=np.int16),
        expert_ids=np.asarray(expert_ids, dtype=np.int16),
        raw_weights=raw,
        normalized_weights=raw / sums,
    )


def _lru_cold_mask(routes: WeightedRoutes, capacity: int) -> np.ndarray:
    if capacity < routes.top_k:
        raise ValueError("resident capacity is smaller than the selected route")
    cold = np.zeros(routes.expert_ids.shape, dtype=bool)
    caches: dict[int, OrderedDict[int, None]] = {}
    for wave_index, layer in enumerate(routes.layer_ids):
        cache = caches.setdefault(int(layer), OrderedDict())
        ids = [int(value) for value in routes.expert_ids[wave_index]]
        for position, expert in enumerate(ids):
            cold[wave_index, position] = expert not in cache
        protected = set(ids)
        for expert in ids:
            if expert in cache:
                cache.move_to_end(expert)
                continue
            while len(cache) >= capacity:
                victim = next(
                    (candidate for candidate in cache if candidate not in protected),
                    None,
                )
                if victim is None:
                    break
                del cache[victim]
            if len(cache) < capacity:
                cache[expert] = None
    return cold


def _weight_integrity_rows(routes: WeightedRoutes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, np.ndarray]] = [
        ("all_decode", np.ones(len(routes.layer_ids), dtype=bool))
    ]
    scopes.extend(
        (f"domain:{domain}", routes.domains == domain)
        for domain in sorted(set(routes.domains.tolist()))
    )
    scopes.extend(
        (f"layer:{layer}", routes.layer_ids == layer)
        for layer in range(routes.layers)
    )
    for scope, mask in scopes:
        sums = routes.raw_weights[mask].sum(axis=1)
        rows.append(
            {
                "scope": scope,
                "waves": int(mask.sum()),
                "raw_selected_weight_sum_mean": float(sums.mean()),
                "raw_selected_weight_sum_p01": _percentile(sums, 0.01),
                "raw_selected_weight_sum_p50": _percentile(sums, 0.50),
                "raw_selected_weight_sum_p99": _percentile(sums, 0.99),
                "raw_selected_weight_sum_min": float(sums.min()),
                "raw_selected_weight_sum_max": float(sums.max()),
                "normalized_selected_weight_sum_max_abs_error": float(
                    np.max(
                        np.abs(
                            routes.normalized_weights[mask].sum(axis=1) - 1.0
                        )
                    )
                ),
                "checkpoint_norm_topk_prob": False,
                "actual_execution_weight_semantics": (
                    "softmax_over_64_then_top8_without_renormalization"
                ),
                "primary_architecture_mass_semantics": (
                    "normalize_within_selected_topk"
                ),
            }
        )
    return rows


def _false_counts(
    useful_counts: np.ndarray,
    amplification: float,
    eligible: np.ndarray,
) -> np.ndarray:
    false = np.zeros_like(useful_counts, dtype=np.int16)
    indices = np.flatnonzero(eligible & (useful_counts > 0))
    if indices.size:
        cumulative = np.rint(
            (amplification - 1.0)
            * np.cumsum(useful_counts[indices], dtype=np.int64)
        ).astype(np.int64)
        false[indices] = np.diff(
            np.concatenate((np.asarray([0], dtype=np.int64), cumulative))
        )
    return false


def _candidate_profile(
    routes: WeightedRoutes,
    cold: np.ndarray,
    *,
    lookahead: int,
    complete: np.ndarray,
    importance_order: str,
    seed: int,
) -> CandidateProfile:
    eligible = routes.layer_ids >= lookahead
    predicted = cold.copy()
    incomplete = eligible & (cold.sum(axis=1) > 0) & ~complete
    random_priority = np.random.default_rng(seed).random(cold.shape)
    for wave in np.flatnonzero(incomplete):
        cold_positions = np.flatnonzero(cold[wave])
        weights = routes.normalized_weights[wave, cold_positions]
        if importance_order == "mass_priority_oracle":
            omitted = cold_positions[int(np.argmin(weights))]
        elif importance_order == "mass_adversarial":
            omitted = cold_positions[int(np.argmax(weights))]
        elif importance_order == "random_within_route":
            omitted = cold_positions[
                int(np.argmin(random_priority[wave, cold_positions]))
            ]
        else:
            raise ValueError(f"unknown importance order {importance_order!r}")
        predicted[wave, omitted] = False

    # The first lookahead layers have no prior source head. They may issue exact
    # reactive demand at the target and use only the configured commit slack.
    predicted[~eligible] = cold[~eligible]
    if importance_order == "mass_priority_oracle":
        priority = np.where(predicted, -routes.normalized_weights, np.inf)
    elif importance_order == "mass_adversarial":
        priority = np.where(predicted, routes.normalized_weights, np.inf)
    else:
        priority = np.where(predicted, random_priority, np.inf)
    service_order = np.argsort(priority, axis=1, kind="stable")
    ordered_valid = np.take_along_axis(predicted, service_order, axis=1)
    ordered_norm = np.where(
        ordered_valid,
        np.take_along_axis(routes.normalized_weights, service_order, axis=1),
        0.0,
    )
    ordered_raw = np.where(
        ordered_valid,
        np.take_along_axis(routes.raw_weights, service_order, axis=1),
        0.0,
    )
    resident = ~cold
    return CandidateProfile(
        predicted_mask=predicted,
        service_order=service_order,
        useful_counts=predicted.sum(axis=1).astype(np.int16),
        resident_normalized_mass=np.where(
            resident, routes.normalized_weights, 0.0
        ).sum(axis=1),
        resident_raw_mass=np.where(resident, routes.raw_weights, 0.0).sum(axis=1),
        ordered_normalized_weights=ordered_norm,
        ordered_raw_weights=ordered_raw,
        complete_mask=complete,
    )


def _served_useful_counts(
    *,
    useful: np.ndarray,
    false: np.ndarray,
    completed: np.ndarray,
    order: str,
    seed: int,
) -> np.ndarray:
    if order == "mass_priority_oracle":
        return np.minimum(useful, completed).astype(np.int16)
    if order == "mass_adversarial":
        return np.minimum(useful, np.maximum(0, completed - false)).astype(
            np.int16
        )
    total = useful + false
    expected = np.divide(
        completed * useful,
        total,
        out=np.zeros_like(completed, dtype=np.float64),
        where=total > 0,
    )
    fractional = expected - np.floor(expected)
    draws = np.random.default_rng(seed).random(len(useful))
    sampled = np.floor(expected).astype(np.int16) + (draws < fractional)
    return np.minimum(useful, sampled).astype(np.int16)


def _take_prefix_sum(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(values, axis=1)
    result = np.zeros(len(values), dtype=np.float64)
    nonzero = counts > 0
    result[nonzero] = cumulative[
        np.flatnonzero(nonzero), counts[nonzero].astype(np.int64) - 1
    ]
    return result


def _longest_run_per_token(values: np.ndarray) -> int:
    longest = 0
    for row in values:
        run = 0
        for value in row:
            run = run + 1 if value else 0
            longest = max(longest, run)
    return longest


def _layer_band(layer: int, layers: int) -> str:
    if layer < layers // 3:
        return "early"
    if layer < 2 * layers // 3:
        return "middle"
    return "late"


def _distribution_metrics(
    missing: np.ndarray,
    raw_missing: np.ndarray,
    *,
    routes: WeightedRoutes,
    renormalization_floor: float,
) -> dict[str, Any]:
    delivered = 1.0 - missing
    token_missing = missing.reshape(routes.token_count, routes.layers)
    token_mean = token_missing.mean(axis=1)
    token_max = token_missing.max(axis=1)
    return {
        "mean_missing_routed_mass": float(missing.mean()),
        "p50_missing_routed_mass": _percentile(missing, 0.50),
        "p95_missing_routed_mass": _percentile(missing, 0.95),
        "p99_missing_routed_mass": _percentile(missing, 0.99),
        "worst_missing_routed_mass": float(missing.max()),
        "mean_raw_missing_router_probability": float(raw_missing.mean()),
        "p99_raw_missing_router_probability": _percentile(raw_missing, 0.99),
        "full_fallback_wave_fraction": float((delivered <= 1e-12).mean()),
        "renormalization_fallback_wave_fraction": float(
            (delivered < renormalization_floor).mean()
        ),
        "degraded_wave_fraction": float((missing > 1e-12).mean()),
        "p50_token_mean_missing_mass": _percentile(token_mean, 0.50),
        "p95_token_mean_missing_mass": _percentile(token_mean, 0.95),
        "p99_token_mean_missing_mass": _percentile(token_mean, 0.99),
        "p99_token_max_layer_missing_mass": _percentile(token_max, 0.99),
        "maximum_consecutive_degraded_layers": _longest_run_per_token(
            token_missing > 1e-12
        ),
    }


def _scope_metrics(
    missing: np.ndarray,
    *,
    routes: WeightedRoutes,
    tau: float,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    passing_domains = 0
    passing_bands = 0
    for domain in sorted(set(routes.domains.tolist())):
        mask = routes.domains == domain
        p99 = _percentile(missing[mask], 0.99)
        fallback = float((missing[mask] >= 1.0 - 1e-12).mean())
        passed = p99 <= tau and fallback <= 0.01
        passing_domains += int(passed)
        rows.append(
            {
                "scope_type": "domain",
                "scope": domain,
                "waves": int(mask.sum()),
                "p99_missing_routed_mass": p99,
                "full_fallback_wave_fraction": fallback,
                "mass_contract_pass": passed,
            }
        )
    bands = np.asarray(
        [_layer_band(int(layer), routes.layers) for layer in routes.layer_ids]
    )
    for band in ("early", "middle", "late"):
        mask = bands == band
        p99 = _percentile(missing[mask], 0.99)
        fallback = float((missing[mask] >= 1.0 - 1e-12).mean())
        passed = p99 <= tau and fallback <= 0.01
        passing_bands += int(passed)
        rows.append(
            {
                "scope_type": "layer_band",
                "scope": band,
                "waves": int(mask.sum()),
                "p99_missing_routed_mass": p99,
                "full_fallback_wave_fraction": fallback,
                "mass_contract_pass": passed,
            }
        )
    return rows, passing_domains, passing_bands


def _simulate_local_deadline(
    *,
    routes: WeightedRoutes,
    profile: CandidateProfile,
    amplification: float,
    importance_order: str,
    lookahead: int,
    slack_intervals: float,
    layer_ms: float,
    transfer_ms: float,
    concurrency: int,
    expert_bytes: int,
    seed: int,
    renormalization_floor: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    eligible = routes.layer_ids >= lookahead
    false = _false_counts(profile.useful_counts, amplification, eligible)
    # Early layers issue exact demand only at the target; amplification does not
    # apply because no future prediction exists.
    false[~eligible] = 0
    total = profile.useful_counts + false
    lead = np.where(
        eligible,
        (lookahead + slack_intervals) * layer_ms,
        slack_intervals * layer_ms,
    )
    complete_slots = (
        np.floor(lead / transfer_ms + 1e-12).astype(np.int32) * concurrency
    )
    started_slots = np.where(
        lead > 0,
        np.ceil(lead / transfer_ms - 1e-12).astype(np.int32) * concurrency,
        0,
    )
    completed = np.minimum(total, complete_slots).astype(np.int16)
    started = np.minimum(total, started_slots).astype(np.int16)
    served_useful = _served_useful_counts(
        useful=profile.useful_counts,
        false=false,
        completed=completed,
        order=importance_order,
        seed=seed,
    )
    served_norm = _take_prefix_sum(
        profile.ordered_normalized_weights, served_useful
    )
    served_raw = _take_prefix_sum(profile.ordered_raw_weights, served_useful)
    missing = np.clip(
        1.0 - profile.resident_normalized_mass - served_norm, 0.0, 1.0
    )
    selected_raw_sum = routes.raw_weights.sum(axis=1)
    raw_missing = np.clip(
        selected_raw_sum - profile.resident_raw_mass - served_raw,
        0.0,
        selected_raw_sum,
    )
    completed_false = completed - served_useful
    metrics = _distribution_metrics(
        missing,
        raw_missing,
        routes=routes,
        renormalization_floor=renormalization_floor,
    )
    useful_total = int(profile.useful_counts.sum())
    candidate_total = int(total.sum())
    metrics.update(
        {
            "predicted_useful_experts": useful_total,
            "predicted_false_experts": int(false.sum()),
            "candidate_experts": candidate_total,
            "realized_candidate_amplification": (
                candidate_total / useful_total if useful_total else 1.0
            ),
            "useful_movement_bytes": int(served_useful.sum()) * expert_bytes,
            "wasted_false_movement_bytes": int(completed_false.sum())
            * expert_bytes,
            "late_inflight_bytes": int((started - completed).sum())
            * expert_bytes,
            "cancelled_before_start_bytes": int((total - started).sum())
            * expert_bytes,
            "candidate_movement_bytes": candidate_total * expert_bytes,
            "zero_post_commit_transfer_wait": True,
            "transfer_induced_post_commit_stall_ms": 0.0,
            "complete_candidate_fraction": float(
                (served_useful == cold_count_from_mask(profile.predicted_mask)).mean()
            ),
        }
    )
    return metrics, missing, raw_missing


def cold_count_from_mask(mask: np.ndarray) -> np.ndarray:
    return mask.sum(axis=1).astype(np.int16)


def _reactive_lookup(
    rows: list[dict[str, str]], capacity: int, bandwidth: float
) -> float:
    candidates = [
        float(row["reactive_p99_tpot_ms"])
        for row in rows
        if int(row["capacity_experts_per_layer"]) == capacity
        and math.isclose(float(row["bandwidth_gbps"]), bandwidth, abs_tol=1e-6)
    ]
    if not candidates:
        raise ValueError(f"missing AX1 reactive anchor K={capacity}, BW={bandwidth}")
    return candidates[0]


def _exact_wait_lookup(
    rows: list[dict[str, str]],
    *,
    capacity: int,
    lookahead: int,
    coverage: float,
    amplification: float,
    bandwidth: float,
) -> float:
    candidates = [
        float(row["modeled_p99_tpot_ms"])
        for row in rows
        if int(row["capacity_experts_per_layer"]) == capacity
        and int(row["lookahead_layers"]) == lookahead
        and math.isclose(
            float(row["requested_complete_cold_set_coverage"]), coverage
        )
        and math.isclose(float(row["requested_amplification"]), amplification)
        and math.isclose(float(row["bandwidth_gbps"]), bandwidth, abs_tol=1e-6)
    ]
    if not candidates:
        raise ValueError("missing AX1 exact-wait anchor")
    return candidates[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _gate_fields(
    *,
    config: dict[str, Any],
    capacity: int,
    num_experts: int,
    bounded_tpot_ms: float,
    base_tpot_ms: float,
    reactive_tpot_ms: float,
    metrics: dict[str, Any],
    passing_domains: int,
    passing_bands: int,
) -> dict[str, Any]:
    gate = config["gate"]
    improvement = 1.0 - bounded_tpot_ms / reactive_tpot_ms
    throughput_gain = reactive_tpot_ms / bounded_tpot_ms
    checks = {
        "gate_resident_capacity": (
            capacity / num_experts <= float(gate["maximum_resident_fraction"])
        ),
        "gate_latency_or_throughput": (
            improvement
            >= float(gate["minimum_p99_tpot_improvement_vs_reactive"])
            or throughput_gain
            >= float(gate["minimum_throughput_improvement_vs_reactive"])
        ),
        "gate_local_tpot_ratio": (
            bounded_tpot_ms / base_tpot_ms
            <= float(gate["maximum_tpot_ratio_vs_all_local_with_fallback"])
        ),
        "gate_p99_missing_mass": (
            metrics["p99_missing_routed_mass"]
            <= float(gate["maximum_p99_missing_routed_mass"])
        ),
        "gate_full_fallback": (
            metrics["full_fallback_wave_fraction"]
            <= float(gate["maximum_full_fallback_wave_fraction"])
        ),
        "gate_domains": passing_domains >= int(gate["minimum_domains"]),
        "gate_layer_bands": passing_bands >= int(gate["minimum_layer_bands"]),
    }
    return {
        "p99_tpot_improvement_vs_reactive": improvement,
        "throughput_improvement_vs_reactive": throughput_gain,
        "passing_domains": passing_domains,
        "passing_layer_bands": passing_bands,
        **checks,
        "gate_pass": all(checks.values()),
    }


def _physical_bounds(
    config: dict[str, Any],
    *,
    layer_ms: float,
    expert_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    startup = float(config["cold_tier"]["startup_latency_us"])
    for capacity in config["residency"]["experts_per_layer"]:
        for lookahead in config["residency"]["lookaheads"]:
            for bandwidth in config["cold_tier"]["bandwidth_gbps"]:
                service = _transfer_ms(expert_bytes, float(bandwidth), startup)
                for amplification in config["predictor"]["amplifications"]:
                    for slack in config["deadline"][
                        "commit_slack_layer_intervals"
                    ]:
                        lead = (float(lookahead) + float(slack)) * layer_ms
                        for concurrency in config["cold_tier"][
                            "transfer_concurrency"
                        ]:
                            completed = (
                                math.floor(lead / service + 1e-12)
                                * int(concurrency)
                            )
                            rows.append(
                                {
                                    "capacity_experts_per_layer": int(capacity),
                                    "lookahead_layers": int(lookahead),
                                    "bandwidth_gbps": float(bandwidth),
                                    "requested_amplification": float(amplification),
                                    "commit_slack_layer_intervals": float(slack),
                                    "transfer_concurrency": int(concurrency),
                                    "expert_transfer_ms": service,
                                    "lead_time_ms": lead,
                                    "complete_expert_objects_before_deadline": completed,
                                    "complete_payload_mib_before_deadline": (
                                        completed * expert_bytes / MIB
                                    ),
                                    "zero_post_commit_wait_required": True,
                                    "evidence": "analytical_hypothetical_hardware",
                                }
                            )
    return rows


def _policy_bounds(
    envelope: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    distances = config["fallback"][
        "normalized_shared_distance_over_expert_bound"
    ]
    for row in envelope:
        common = {
            "scenario_id": row["scenario_id"],
            "capacity_experts_per_layer": row["capacity_experts_per_layer"],
            "p99_missing_routed_mass": row["p99_missing_routed_mass"],
            "bounded_p99_tpot_ms": row["bounded_p99_tpot_ms"],
        }
        rows.append(
            {
                **common,
                "policy": "null_residual",
                "shared_distance_over_expert_bound": "",
                "p99_normalized_perturbation_bound": row[
                    "p99_missing_routed_mass"
                ],
                "bound_semantics": "m_times_B",
            }
        )
        rows.append(
            {
                **common,
                "policy": "present_renormalization",
                "shared_distance_over_expert_bound": "",
                "p99_normalized_perturbation_bound": min(
                    2.0, 2.0 * row["p99_missing_routed_mass"]
                ),
                "bound_semantics": "2m_times_B",
            }
        )
        for distance in distances:
            rows.append(
                {
                    **common,
                    "policy": "shared_residual",
                    "shared_distance_over_expert_bound": float(distance),
                    "p99_normalized_perturbation_bound": (
                        row["p99_missing_routed_mass"] * float(distance)
                    ),
                    "bound_semantics": "m_times_D_over_B",
                }
            )
    return rows


def _choose_fcfs_points(
    envelope: list[dict[str, Any]], measured_bandwidth: float
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    bandwidths = sorted({float(row["bandwidth_gbps"]) for row in envelope})
    for capacity in (8, 16, 32):
        # Compact 3 x 4 FCFS hardware boundary for the mass-priority policy.
        for bandwidth in bandwidths:
            candidates = [
                row
                for row in envelope
                if row["capacity_experts_per_layer"] == capacity
                and row["importance_order"] == "mass_priority_oracle"
                and row["requested_complete_coverage"] >= 0.99
                and math.isclose(
                    row["bandwidth_gbps"], bandwidth, abs_tol=1e-6
                )
            ]
            candidates.sort(
                key=lambda row: (
                    not row["gate_pass"],
                    round(float(row["p99_missing_routed_mass"]), 6),
                    row["requested_amplification"],
                    row["lookahead_layers"],
                    row["commit_slack_layer_intervals"],
                )
            )
            if candidates:
                selected.append(candidates[0])
        # Measured-PCIe importance-order sensitivities.
        for order in ("random_within_route", "mass_adversarial"):
            candidates = [
                row
                for row in envelope
                if row["capacity_experts_per_layer"] == capacity
                and row["importance_order"] == order
                and row["requested_complete_coverage"] >= 0.99
                and math.isclose(
                    row["bandwidth_gbps"], measured_bandwidth, abs_tol=1e-6
                )
            ]
            candidates.sort(
                key=lambda row: (
                    not row["gate_pass"],
                    round(float(row["p99_missing_routed_mass"]), 6),
                    row["requested_amplification"],
                    row["lookahead_layers"],
                    row["commit_slack_layer_intervals"],
                )
            )
            if candidates:
                selected.append(candidates[0])
    anchor = min(
        envelope,
        key=lambda row: (
            abs(row["capacity_experts_per_layer"] - 16),
            abs(row["lookahead_layers"] - 9),
            abs(row["requested_complete_coverage"] - 0.99),
            abs(row["requested_amplification"] - 1.5),
            abs(row["bandwidth_gbps"] - measured_bandwidth),
            abs(row["commit_slack_layer_intervals"] - 0.5),
            row["importance_order"] != "mass_priority_oracle",
        ),
    )
    selected.append(anchor)
    return list(
        {
            row["scenario_id"]: row
            for row in selected
        }.values()
    )


def _fcfs_replay(
    *,
    routes: WeightedRoutes,
    cold: np.ndarray,
    profile: CandidateProfile,
    amplification: float,
    importance_order: str,
    lookahead: int,
    slack_intervals: float,
    layer_ms: float,
    transfer_ms: float,
    concurrency: int,
    expert_bytes: int,
    seed: int,
    renormalization_floor: float,
) -> tuple[dict[str, Any], np.ndarray]:
    eligible = routes.layer_ids >= lookahead
    false = _false_counts(profile.useful_counts, amplification, eligible)
    false[~eligible] = 0
    available = np.zeros_like(cold, dtype=bool)
    useful_completed = 0
    false_completed = 0
    late = 0
    cancelled = 0
    rng = np.random.default_rng(seed)

    for token in range(routes.token_count):
        lanes = np.zeros(concurrency, dtype=np.float64)
        base = token * routes.layers
        for source in range(routes.layers):
            targets: list[int] = []
            if source < lookahead:
                targets.append(source)
            target = source + lookahead
            if target < routes.layers:
                targets.append(target)
            for target in targets:
                wave = base + target
                deadline = (target + slack_intervals) * layer_ms
                useful_positions = [
                    int(position)
                    for position in profile.service_order[wave]
                    if profile.predicted_mask[wave, position]
                ]
                false_jobs = int(false[wave])
                jobs: list[int | None]
                if importance_order == "mass_priority_oracle":
                    jobs = useful_positions + [None] * false_jobs
                elif importance_order == "mass_adversarial":
                    jobs = [None] * false_jobs + useful_positions
                else:
                    jobs = useful_positions + [None] * false_jobs
                    rng.shuffle(jobs)
                issue = source * layer_ms
                for position in jobs:
                    lane = int(np.argmin(lanes))
                    start = max(issue, float(lanes[lane]))
                    if start >= deadline - 1e-12:
                        cancelled += 1
                        continue
                    finish = start + transfer_ms
                    lanes[lane] = finish
                    if finish <= deadline + 1e-12:
                        if position is None:
                            false_completed += 1
                        else:
                            available[wave, position] = True
                            useful_completed += 1
                    else:
                        late += 1

    resident = ~cold
    delivered_mask = resident | available
    delivered_norm = np.where(
        delivered_mask, routes.normalized_weights, 0.0
    ).sum(axis=1)
    delivered_raw = np.where(delivered_mask, routes.raw_weights, 0.0).sum(axis=1)
    missing = np.clip(1.0 - delivered_norm, 0.0, 1.0)
    raw_missing = np.clip(
        routes.raw_weights.sum(axis=1) - delivered_raw,
        0.0,
        routes.raw_weights.sum(axis=1),
    )
    metrics = _distribution_metrics(
        missing,
        raw_missing,
        routes=routes,
        renormalization_floor=renormalization_floor,
    )
    candidate_total = int(profile.useful_counts.sum() + false.sum())
    metrics.update(
        {
            "predicted_useful_experts": int(profile.useful_counts.sum()),
            "predicted_false_experts": int(false.sum()),
            "candidate_experts": candidate_total,
            "realized_candidate_amplification": (
                candidate_total / int(profile.useful_counts.sum())
                if profile.useful_counts.sum()
                else 1.0
            ),
            "useful_movement_bytes": useful_completed * expert_bytes,
            "wasted_false_movement_bytes": false_completed * expert_bytes,
            "late_inflight_bytes": late * expert_bytes,
            "cancelled_before_start_bytes": cancelled * expert_bytes,
            "candidate_movement_bytes": candidate_total * expert_bytes,
            "zero_post_commit_transfer_wait": True,
            "transfer_induced_post_commit_stall_ms": 0.0,
        }
    )
    return metrics, missing


def _large_model_projection(
    config: dict[str, Any],
    *,
    selected: dict[str, Any],
    expert_bytes_current: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_missing = float(selected["p99_missing_routed_mass"])
    reference_fallback = float(selected["full_fallback_wave_fraction"])
    batches = [
        int(config["low_batch"]["primary_batch_size"]),
        *[int(value) for value in config["low_batch"]["sensitivity_batch_sizes"]],
        int(config["low_batch"]["separate_batch_size"]),
    ]
    for experts in config["large_model_sensitivity"]["experts_per_layer"]:
        for resident_fraction in config["large_model_sensitivity"][
            "resident_fractions"
        ]:
            for route_width in config["large_model_sensitivity"]["route_widths"]:
                # Rank truncation changes erasure severity. Without a trace from
                # the target model, use the top-8 result as a labeled workload
                # shape rather than inventing new routing statistics.
                mass = min(
                    1.0,
                    reference_missing
                    * math.sqrt(8.0 / float(route_width)),
                )
                fallback = min(
                    1.0,
                    reference_fallback * 8.0 / float(route_width),
                )
                for layers in config["large_model_sensitivity"]["moe_layers"]:
                    for expert_mib in config["large_model_sensitivity"][
                        "expert_size_mib"
                    ]:
                        total = (
                            int(experts)
                            * int(layers)
                            * float(expert_mib)
                            * MIB
                        )
                        resident_experts = math.ceil(
                            int(experts) * float(resident_fraction)
                        )
                        fallback_bytes = int(layers) * float(expert_mib) * MIB
                        resident = (
                            int(layers)
                            * resident_experts
                            * float(expert_mib)
                            * MIB
                            + fallback_bytes
                        )
                        for batch in batches:
                            for target in config["low_batch"]["tpot_targets_ms"]:
                                rows.append(
                                    {
                                        "experts_per_layer": int(experts),
                                        "resident_fraction": float(
                                            resident_fraction
                                        ),
                                        "resident_experts_per_layer": resident_experts,
                                        "route_width": int(route_width),
                                        "moe_layers": int(layers),
                                        "expert_size_mib": float(expert_mib),
                                        "batch_size": batch,
                                        "bounded_tpot_target_ms": float(target),
                                        "projected_tokens_per_second": (
                                            1000.0 * batch / float(target)
                                        ),
                                        "total_expert_capacity_gib": total / GIB,
                                        "resident_expert_plus_fallback_gib": (
                                            resident / GIB
                                        ),
                                        "offloaded_expert_capacity_gib": (
                                            total - resident + fallback_bytes
                                        )
                                        / GIB,
                                        "ideal_capacity_expansion_before_fallback": (
                                            1.0 / float(resident_fraction)
                                        ),
                                        "projected_p99_missing_mass": mass,
                                        "projected_full_fallback_wave_fraction": fallback,
                                        "olmoe_top8_rank_sensitivity_only": True,
                                        "batch_compute_scaling_measured": False,
                                        "source_expert_size_mib": (
                                            expert_bytes_current / MIB
                                        ),
                                        "evidence": (
                                            "normalized_hypothetical_geometry_"
                                            "using_olmoe_workload_shape"
                                        ),
                                    }
                                )
    return rows


def analyze_deadline_degradation(
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    output = Path(experiment_config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    run = Path(experiment_config["h1_run"])
    architecture = Path(experiment_config["ax_analysis"])
    measurement = json.loads(
        Path(experiment_config["h4_measurement"]).read_text(encoding="utf-8")
    )
    model_report = json.loads((run / "model_report.json").read_text(encoding="utf-8"))
    layers = int(model_report["router_count"])
    num_experts = {int(row["num_experts"]) for row in model_report["routers"]}
    top_ks = {int(row["top_k"]) for row in model_report["routers"]}
    expert_sizes = {
        int(row["expert_bytes_each"]) for row in model_report["routers"]
    }
    if len(num_experts) != 1 or len(top_ks) != 1 or len(expert_sizes) != 1:
        raise ValueError("AX4 requires uniform router geometry")
    experts = num_experts.pop()
    top_k = top_ks.pop()
    expert_bytes = expert_sizes.pop()
    routes = _load_weighted_decode(run, layers)
    if routes.top_k != top_k:
        raise ValueError("trace top-k does not match model report")
    layer_ms = float(measurement["decode"]["effective_inter_moe_layer_ms"])
    base_tpot = float(measurement["decode"]["median_forward_ms"])
    startup_us = float(experiment_config["cold_tier"]["startup_latency_us"])
    overhead = 0.10
    if overhead not in [
        float(value)
        for value in experiment_config["fallback"][
            "local_tpot_overhead_fractions"
        ]
    ]:
        raise ValueError("frozen 10% fallback overhead is missing")
    bounded_tpot = base_tpot * (1.0 + overhead)
    renorm_floor = float(
        experiment_config["weights"][
            "minimum_delivered_mass_for_renormalization"
        ]
    )
    ax1 = _read_csv(architecture / "ax1_envelope.csv")

    integrity = _weight_integrity_rows(routes)
    _write_csv(output / "weight_semantics_integrity.csv", integrity)
    physical = _physical_bounds(
        experiment_config, layer_ms=layer_ms, expert_bytes=expert_bytes
    )
    _write_csv(output / "deadline_physical_bounds.csv", physical)

    cold_by_capacity = {
        int(capacity): _lru_cold_mask(routes, int(capacity))
        for capacity in experiment_config["residency"]["experts_per_layer"]
    }
    measured_bandwidth_value = float(
        measurement["transfer"]["fit"]["effective_bandwidth_gbps"]
    )
    measured_bandwidth = min(
        (float(value) for value in experiment_config["cold_tier"]["bandwidth_gbps"]),
        key=lambda value: abs(value - measured_bandwidth_value),
    )
    frozen_slacks = [
        float(value)
        for value in experiment_config["deadline"][
            "commit_slack_layer_intervals"
        ]
    ]
    predictor_panel_slack = min(frozen_slacks, key=lambda value: abs(value - 0.5))
    # The protocol explicitly asks for a factorized analysis. The predictor
    # panel spans C/A/order at measured PCIe and fixed 0.5-layer slack. The
    # hardware panel spans bandwidth/slack at frozen C=99%, A=1.5, and the
    # mass-priority scheduler. Concurrency is swept independently below.
    scenario_specs: set[tuple[int, int, float, str, float, float, float]] = set()
    for capacity in experiment_config["residency"]["experts_per_layer"]:
        for lookahead in experiment_config["residency"]["lookaheads"]:
            for coverage in experiment_config["predictor"]["complete_coverages"]:
                for amplification in experiment_config["predictor"][
                    "amplifications"
                ]:
                    for order in experiment_config["predictor"][
                        "importance_orders"
                    ]:
                        scenario_specs.add(
                            (
                                int(capacity),
                                int(lookahead),
                                float(coverage),
                                str(order),
                                float(amplification),
                                measured_bandwidth,
                                predictor_panel_slack,
                            )
                        )
            for bandwidth in experiment_config["cold_tier"]["bandwidth_gbps"]:
                for slack in frozen_slacks:
                    scenario_specs.add(
                        (
                            int(capacity),
                            int(lookahead),
                            0.99,
                            "mass_priority_oracle",
                            1.5,
                            float(bandwidth),
                            float(slack),
                        )
                    )

    profiles: dict[tuple[int, int, float, str], CandidateProfile] = {}
    coverage_cache: dict[tuple[int, int, float], tuple[np.ndarray, dict[str, Any]]] = {}
    envelope: list[dict[str, Any]] = []
    scope_rows: list[dict[str, Any]] = []
    for scenario, spec in enumerate(sorted(scenario_specs)):
        (
            capacity,
            lookahead,
            coverage,
            order,
            amplification,
            bandwidth,
            slack,
        ) = spec
        cold = cold_by_capacity[capacity]
        coverage_key = (capacity, lookahead, coverage)
        if coverage_key not in coverage_cache:
            cold_matrix = cold.sum(axis=1).reshape(routes.token_count, layers)
            eligible_cold = cold_matrix.copy()
            eligible_cold[:, :lookahead] = 0
            complete_matrix, coverage_info = _correlated_complete_mask(
                eligible_cold,
                coverage,
                seed=(
                    int(experiment_config["seed"])
                    + 1009 * capacity
                    + 97 * lookahead
                ),
            )
            coverage_cache[coverage_key] = (
                complete_matrix.reshape(-1),
                coverage_info,
            )
        complete, coverage_info = coverage_cache[coverage_key]
        profile_key = (capacity, lookahead, coverage, order)
        if profile_key not in profiles:
            profiles[profile_key] = _candidate_profile(
                routes,
                cold,
                lookahead=lookahead,
                complete=complete,
                importance_order=order,
                seed=(
                    int(experiment_config["seed"])
                    + 7919 * capacity
                    + 313 * lookahead
                    + 17 * len(order)
                ),
            )
        profile = profiles[profile_key]
        transfer_ms = _transfer_ms(expert_bytes, bandwidth, startup_us)
        reactive = _reactive_lookup(ax1, capacity, bandwidth)
        exact_wait = _exact_wait_lookup(
            ax1,
            capacity=capacity,
            lookahead=lookahead,
            coverage=coverage,
            amplification=amplification,
            bandwidth=bandwidth,
        )
        metrics, missing, _raw_missing = _simulate_local_deadline(
            routes=routes,
            profile=profile,
            amplification=amplification,
            importance_order=order,
            lookahead=lookahead,
            slack_intervals=slack,
            layer_ms=layer_ms,
            transfer_ms=transfer_ms,
            concurrency=1,
            expert_bytes=expert_bytes,
            seed=int(experiment_config["seed"]) + scenario,
            renormalization_floor=renorm_floor,
        )
        scopes, domain_count, band_count = _scope_metrics(
            missing,
            routes=routes,
            tau=float(
                experiment_config["gate"]["maximum_p99_missing_routed_mass"]
            ),
        )
        gate = _gate_fields(
            config=experiment_config,
            capacity=capacity,
            num_experts=experts,
            bounded_tpot_ms=bounded_tpot,
            base_tpot_ms=base_tpot,
            reactive_tpot_ms=reactive,
            metrics=metrics,
            passing_domains=domain_count,
            passing_bands=band_count,
        )
        scenario_id = f"ax4-local-{scenario:05d}"
        factor_panel = (
            "predictor"
            if math.isclose(bandwidth, measured_bandwidth, abs_tol=1e-6)
            and math.isclose(slack, predictor_panel_slack)
            else "hardware"
        )
        row = {
            "scenario_id": scenario_id,
            "factor_panel": factor_panel,
            "tail_model": "trace_wave_local_deadline_factorized",
            "capacity_experts_per_layer": capacity,
            "resident_fraction": capacity / experts,
            "fast_tier_expert_gib": capacity * layers * expert_bytes / GIB,
            "offloaded_expert_gib": (
                (experts - capacity) * layers * expert_bytes / GIB
            ),
            "lookahead_layers": lookahead,
            "requested_complete_coverage": coverage,
            "realized_complete_coverage": coverage_info[
                "realized_complete_cold_set_coverage"
            ],
            "requested_amplification": amplification,
            "importance_order": order,
            "bandwidth_gbps": bandwidth,
            "transfer_concurrency": 1,
            "commit_slack_layer_intervals": slack,
            "fallback_overhead_fraction": overhead,
            "bounded_p99_tpot_ms": bounded_tpot,
            "bounded_batch1_tokens_per_second": 1000.0 / bounded_tpot,
            "reactive_p99_tpot_ms": reactive,
            "reactive_batch1_tokens_per_second": 1000.0 / reactive,
            "exact_wait_wave_local_p99_tpot_ms": exact_wait,
            **metrics,
            **gate,
            "evidence_weights_and_demand": "trace_derived",
            "evidence_timing": "measured_anchor",
            "evidence_predictor": "assumed",
            "evidence_robustness": "assumed_not_validated",
            "evidence_hardware": (
                "measured_bandwidth_anchor"
                if math.isclose(
                    bandwidth,
                    measured_bandwidth,
                    abs_tol=1e-3,
                )
                else "hypothetical"
            ),
        }
        envelope.append(row)
        scope_rows.extend(
            {"scenario_id": scenario_id, **scope} for scope in scopes
        )

    _write_csv(output / "deadline_envelope.csv", envelope)
    _write_csv(output / "deadline_scope_metrics.csv", scope_rows)
    policy = _policy_bounds(envelope, experiment_config)
    _write_csv(output / "degradation_policy_bounds.csv", policy)

    selected_local = _choose_fcfs_points(envelope, measured_bandwidth)
    fcfs_rows: list[dict[str, Any]] = []
    fcfs_scope_rows: list[dict[str, Any]] = []
    for local in selected_local:
        capacity = int(local["capacity_experts_per_layer"])
        lookahead = int(local["lookahead_layers"])
        coverage = float(local["requested_complete_coverage"])
        order = str(local["importance_order"])
        profile = profiles[(capacity, lookahead, coverage, order)]
        metrics, missing = _fcfs_replay(
            routes=routes,
            cold=cold_by_capacity[capacity],
            profile=profile,
            amplification=float(local["requested_amplification"]),
            importance_order=order,
            lookahead=lookahead,
            slack_intervals=float(local["commit_slack_layer_intervals"]),
            layer_ms=layer_ms,
            transfer_ms=_transfer_ms(
                expert_bytes, float(local["bandwidth_gbps"]), startup_us
            ),
            concurrency=1,
            expert_bytes=expert_bytes,
            seed=int(experiment_config["seed"]) + 50000,
            renormalization_floor=renorm_floor,
        )
        scopes, domain_count, band_count = _scope_metrics(
            missing,
            routes=routes,
            tau=float(
                experiment_config["gate"]["maximum_p99_missing_routed_mass"]
            ),
        )
        gate = _gate_fields(
            config=experiment_config,
            capacity=capacity,
            num_experts=experts,
            bounded_tpot_ms=bounded_tpot,
            base_tpot_ms=base_tpot,
            reactive_tpot_ms=float(local["reactive_p99_tpot_ms"]),
            metrics=metrics,
            passing_domains=domain_count,
            passing_bands=band_count,
        )
        fcfs_id = local["scenario_id"].replace("local", "fcfs")
        fcfs_rows.append(
            {
                "scenario_id": fcfs_id,
                "source_local_scenario_id": local["scenario_id"],
                "tail_model": "trace_ordered_per_token_fcfs_hard_deadline",
                "capacity_experts_per_layer": capacity,
                "resident_fraction": local["resident_fraction"],
                "fast_tier_expert_gib": local["fast_tier_expert_gib"],
                "offloaded_expert_gib": local["offloaded_expert_gib"],
                "lookahead_layers": lookahead,
                "requested_complete_coverage": coverage,
                "requested_amplification": float(
                    local["requested_amplification"]
                ),
                "importance_order": order,
                "bandwidth_gbps": float(local["bandwidth_gbps"]),
                "transfer_concurrency": 1,
                "commit_slack_layer_intervals": float(
                    local["commit_slack_layer_intervals"]
                ),
                "bounded_p99_tpot_ms": bounded_tpot,
                "bounded_batch1_tokens_per_second": 1000.0 / bounded_tpot,
                "reactive_p99_tpot_ms": float(local["reactive_p99_tpot_ms"]),
                "exact_wait_wave_local_p99_tpot_ms": float(
                    local["exact_wait_wave_local_p99_tpot_ms"]
                ),
                **metrics,
                **gate,
                "evidence": (
                    "trace_ordered_fcfs_plus_assumed_predictor_and_robustness"
                ),
            }
        )
        fcfs_scope_rows.extend(
            {"scenario_id": fcfs_id, **scope} for scope in scopes
        )
    _write_csv(output / "deadline_fcfs_candidates.csv", fcfs_rows)
    _write_csv(output / "deadline_fcfs_scope_metrics.csv", fcfs_scope_rows)

    passing_fcfs = [row for row in fcfs_rows if row["gate_pass"]]
    selected_projection = min(
        passing_fcfs or fcfs_rows,
        key=lambda row: (
            not row["gate_pass"],
            row["p99_missing_routed_mass"],
            row["capacity_experts_per_layer"],
        ),
    )
    large = _large_model_projection(
        experiment_config,
        selected=selected_projection,
        expert_bytes_current=expert_bytes,
    )
    _write_csv(output / "large_sparse_model_projection.csv", large)

    thresholds = [
        float(value)
        for value in experiment_config["deadline"]["missing_mass_thresholds"]
    ]
    threshold_rows: list[dict[str, Any]] = []
    for row in fcfs_rows:
        scopes = [
            value
            for value in fcfs_scope_rows
            if value["scenario_id"] == row["scenario_id"]
        ]
        for threshold in thresholds:
            threshold_rows.append(
                {
                    "scenario_id": row["scenario_id"],
                    "threshold": threshold,
                    "p99_mass_within_threshold": (
                        row["p99_missing_routed_mass"] <= threshold
                    ),
                    "domains_within_threshold": sum(
                        value["p99_missing_routed_mass"] <= threshold
                        for value in scopes
                        if value["scope_type"] == "domain"
                    ),
                    "layer_bands_within_threshold": sum(
                        value["p99_missing_routed_mass"] <= threshold
                        for value in scopes
                        if value["scope_type"] == "layer_band"
                    ),
                }
            )
    _write_csv(output / "missing_mass_contracts.csv", threshold_rows)

    oracle_rows = [
        row
        for row in fcfs_rows
        if row["importance_order"] == "mass_priority_oracle"
    ]
    oracle_stop = all(
        float(row["p99_missing_routed_mass"])
        > float(experiment_config["gate"]["oracle_stop_missing_mass"])
        for row in oracle_rows
    )
    passed = bool(passing_fcfs)
    headline = min(
        fcfs_rows,
        key=lambda row: (
            not row["gate_pass"],
            row["p99_missing_routed_mass"],
            -row["p99_tpot_improvement_vs_reactive"],
        ),
    )
    summary = {
        "track": "AX4",
        "state": "complete_pending_human_review",
        "formal_gate_passed": passed,
        "oracle_stop_triggered": oracle_stop,
        "gate_interpretation": (
            "Pass establishes a plausible future erasure-robustness target, "
            "not acceptable model quality."
        ),
        "trace": {
            "decode_tokens": routes.token_count,
            "decode_waves": len(routes.layer_ids),
            "domains": sorted(set(routes.domains.tolist())),
            "layers": layers,
            "experts_per_layer": experts,
            "top_k": top_k,
        },
        "weight_semantics": {
            "checkpoint_norm_topk_prob": False,
            "actual_execution": (
                "softmax_over_all_64_then_top8_without_renormalization"
            ),
            "primary_architecture_metric": "normalize_within_selected_topk",
            "raw_selected_weight_sum_mean": integrity[0][
                "raw_selected_weight_sum_mean"
            ],
            "raw_selected_weight_sum_p99": integrity[0][
                "raw_selected_weight_sum_p99"
            ],
        },
        "frozen_latency_prediction": {
            "all_local_anchor_ms": base_tpot,
            "fallback_commit_overhead_fraction": overhead,
            "bounded_tpot_ms": bounded_tpot,
            "batch1_tokens_per_second": 1000.0 / bounded_tpot,
            "reactive_k16_anchor_ms": 66.83,
            "reactive_k16_tokens_per_second": 1000.0 / 66.83,
            "tpot_improvement_fraction": 1.0 - bounded_tpot / 66.83,
            "same_batch_throughput_gain": 66.83 / bounded_tpot,
            "zero_post_commit_wait": True,
        },
        "headline_fcfs_candidate": headline,
        "grid": {
            "wave_local_scenarios": len(envelope),
            "factorized_physical_cells": len(physical),
            "selected_fcfs_candidates": len(fcfs_rows),
            "large_model_projection_rows": len(large),
        },
        "evidence_boundary": (
            "Latency is trace-calibrated and nonblocking by construction. "
            "Missing mass is trace-derived under an assumed predictor. "
            "Robustness, quality preservation, larger-model routing, and "
            "non-measured hardware points remain assumptions."
        ),
        "outputs": {
            "report": str(output / "REPORT.md"),
            "envelope": str(output / "deadline_envelope.csv"),
            "fcfs": str(output / "deadline_fcfs_candidates.csv"),
            "scopes": str(output / "deadline_fcfs_scope_metrics.csv"),
            "large_model": str(output / "large_sparse_model_projection.csv"),
        },
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "evidence_ledger.json",
        {
            "measured": [
                "hook-free OLMoE decode timing",
                "12 MiB expert size",
                "pinned host-to-device transfer fit",
            ],
            "trace_derived": [
                "selected expert IDs and unnormalized selected weights",
                "normalized-within-top8 contribution shape",
                "LRU cold demand, domains, layers, and request order",
            ],
            "assumed_predictor": [
                "wave-complete coverage",
                "candidate amplification",
                "mass ordering and correlated incomplete waves",
            ],
            "assumed_robustness": [
                "missing routed mass is a useful future training contract",
                "null, renormalized, or shared residual fallback preserves quality",
            ],
            "hypothetical_hardware": [
                "nonblocking atomic commit",
                "fallback provisioning and speculative-traffic isolation",
                "non-measured bandwidth, concurrency, and large-model geometries",
            ],
        },
    )
    _write_report(output, summary, fcfs_rows)
    return summary


def _write_report(
    output: Path,
    summary: dict[str, Any],
    fcfs: list[dict[str, Any]],
) -> None:
    prediction = summary["frozen_latency_prediction"]
    headline = summary["headline_fcfs_candidate"]
    gate_text = "passes" if summary["formal_gate_passed"] else "does not pass"
    lines = [
        "# AX4 deadline-bounded graceful expert degradation",
        "",
        "**State:** complete; pending human figure review",
        "",
        "## Plain-language result",
        "",
        f"AX4 {gate_text} the preregistered architecture gate under the selected "
        "trace-ordered FCFS candidates. Hard commit makes transfer latency "
        "irrelevant after the deadline; the price is missing routed mass, whose "
        "language-quality meaning remains unvalidated.",
        "",
        "## Weight-semantics finding",
        "",
        "OLMoE does **not** renormalize its selected top-8 weights. The router "
        "softmaxes over all 64 experts, selects eight, and the expert block uses "
        "those probabilities directly. AX4 therefore uses normalized-within-top-8 "
        "mass only as the primary architecture contract and preserves absolute "
        "missing router probability as a secondary integrity metric.",
        "",
        f"Across decode waves, the mean raw selected-weight sum is "
        f"{summary['weight_semantics']['raw_selected_weight_sum_mean']:.3f}; "
        f"P99 is {summary['weight_semantics']['raw_selected_weight_sum_p99']:.3f}.",
        "",
        "## Frozen latency prediction",
        "",
        f"The measured all-local anchor is {prediction['all_local_anchor_ms']:.2f} ms. "
        f"A fixed 10% commit/fallback allowance gives "
        f"{prediction['bounded_tpot_ms']:.2f} ms and "
        f"{prediction['batch1_tokens_per_second']:.1f} token/s. Against the "
        f"66.83 ms K=16 reactive projection, this is "
        f"{100 * prediction['tpot_improvement_fraction']:.1f}% lower TPOT and "
        f"{prediction['same_batch_throughput_gain']:.2f}× throughput.",
        "",
        "This bound is conditional on zero waiting after commit, reserved fallback "
        "compute, bounded local work, and isolation of speculative traffic.",
        "",
        "## Selected FCFS candidates",
        "",
        "| K | Δ | C | A | order | slack | BW | P99 missing mass | fallback waves | gate |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        fcfs,
        key=lambda value: (
            int(value["capacity_experts_per_layer"]),
            str(value["importance_order"]),
        ),
    ):
        lines.append(
            f"| {row['capacity_experts_per_layer']} | {row['lookahead_layers']} | "
            f"{100 * float(row['requested_complete_coverage']):.1f}% | "
            f"{float(row['requested_amplification']):.1f}× | "
            f"{row['importance_order']} | "
            f"{float(row['commit_slack_layer_intervals']):.2f} | "
            f"{float(row['bandwidth_gbps']):.1f} GB/s | "
            f"{100 * float(row['p99_missing_routed_mass']):.1f}% | "
            f"{100 * float(row['full_fallback_wave_fraction']):.2f}% | "
            f"{'pass' if row['gate_pass'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Headline candidate",
            "",
            f"The best selected FCFS point is K={headline['capacity_experts_per_layer']}, "
            f"Δ={headline['lookahead_layers']}, "
            f"C={100 * float(headline['requested_complete_coverage']):.1f}%, "
            f"A={float(headline['requested_amplification']):.1f}×, "
            f"{headline['importance_order']}. Its P99 normalized missing mass is "
            f"{100 * float(headline['p99_missing_routed_mass']):.1f}% and its "
            f"full-fallback wave rate is "
            f"{100 * float(headline['full_fallback_wave_fraction']):.2f}%.",
            "",
            "Null, present-renormalized, and shared-residual execution have the "
            "same availability and TPOT in this model. They differ only in the "
            "assumed quality response and perturbation bound: mB, 2mB, and mD.",
            "",
            "## Interpretation",
            "",
            "A gate pass would justify a tightly scoped future training test of "
            "availability-conditioned expert erasure. It would not show that "
            "current OLMoE maintains perplexity or task quality. A gate failure "
            "would mean that even controlled approximation needs either more "
            "resident capacity, better mass ranking, more lookahead/bandwidth, "
            "or a looser missing-mass contract.",
            "",
            summary["evidence_boundary"],
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
