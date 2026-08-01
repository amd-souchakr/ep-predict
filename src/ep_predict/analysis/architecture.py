from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ep_predict.analysis.h4 import _cold_sets, _decode_waves
from ep_predict.tracing.storage import write_json


MIB = 1024**2
GIB = 1024**3


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.quantile(values, percentile, method="higher"))


def _histogram_percentile(
    values: np.ndarray,
    frequencies: np.ndarray,
    percentile: float,
) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_frequencies = frequencies[order]
    index = math.ceil(percentile * (int(ordered_frequencies.sum()) - 1))
    position = int(
        np.searchsorted(np.cumsum(ordered_frequencies), index, side="right")
    )
    return float(ordered_values[position])


def _transfer_ms(size_bytes: float, bandwidth_gbps: float, startup_us: float) -> float:
    return startup_us / 1000.0 + size_bytes / (bandwidth_gbps * 1e6)


def _correlated_complete_mask(
    cold_counts: np.ndarray,
    coverage: float,
    *,
    seed: int,
    block_waves: int = 4,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Create nested, block-correlated misses over nonempty demand waves."""
    flat = cold_counts.reshape(-1)
    cold_indices = np.flatnonzero(flat > 0)
    complete = np.zeros(flat.size, dtype=bool)
    complete[cold_indices] = True
    miss_target = int(round((1.0 - coverage) * cold_indices.size))
    if miss_target:
        blocks = [
            cold_indices[start : start + block_waves]
            for start in range(0, cold_indices.size, block_waves)
        ]
        order = np.random.default_rng(seed).permutation(len(blocks))
        remaining = miss_target
        for block_index in order:
            if remaining <= 0:
                break
            block = blocks[int(block_index)]
            chosen = block[:remaining]
            complete[chosen] = False
            remaining -= len(chosen)

    cold_complete = complete[cold_indices]
    miss_runs: list[int] = []
    run = 0
    for value in cold_complete:
        if value:
            if run:
                miss_runs.append(run)
                run = 0
        else:
            run += 1
    if run:
        miss_runs.append(run)
    realized = float(cold_complete.mean()) if cold_complete.size else 1.0
    return complete.reshape(cold_counts.shape), {
        "requested_complete_cold_set_coverage": coverage,
        "realized_complete_cold_set_coverage": realized,
        "eligible_cold_waves": int(cold_indices.size),
        "incomplete_cold_waves": int((~cold_complete).sum()),
        "correlation_block_waves": block_waves,
        "mean_incomplete_run_waves": (
            float(np.mean(miss_runs)) if miss_runs else 0.0
        ),
        "max_incomplete_run_waves": max(miss_runs, default=0),
    }


def _predicted_counts(
    cold_counts: np.ndarray,
    complete: np.ndarray,
    amplification: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    useful = np.where(complete, cold_counts, 0).astype(np.int64)
    false = np.zeros_like(useful)
    useful_indices = np.flatnonzero(useful.reshape(-1) > 0)
    if useful_indices.size:
        useful_values = useful.reshape(-1)[useful_indices]
        desired_cumulative = np.rint(
            (amplification - 1.0) * np.cumsum(useful_values)
        ).astype(np.int64)
        false_values = np.diff(
            np.concatenate((np.array([0], dtype=np.int64), desired_cumulative))
        )
        false.reshape(-1)[useful_indices] = false_values
    predicted = useful + false
    useful_total = int(useful.sum())
    predicted_total = int(predicted.sum())
    return predicted, {
        "requested_predicted_to_useful_amplification": amplification,
        "realized_predicted_to_useful_amplification": (
            predicted_total / useful_total if useful_total else 1.0
        ),
        "useful_predicted_objects": useful_total,
        "false_predicted_objects": int(false.sum()),
        "predicted_objects": predicted_total,
    }


def _cold_count_matrix(
    *,
    waves: list[Any],
    token_count: int,
    layers: int,
    capacity: int,
) -> np.ndarray:
    cold, _compulsory, _demanded, _misses, _first = _cold_sets(waves, capacity)
    return np.asarray(
        [
            [len(cold[(token_index, layer)]) for layer in range(layers)]
            for token_index in range(token_count)
        ],
        dtype=np.int16,
    )


def _local_stall_metrics(
    *,
    cold_counts: np.ndarray,
    complete: np.ndarray,
    predicted: np.ndarray,
    lookahead: int,
    layer_ms: float,
    transfer_ms: float,
    base_tpot_ms: float,
    concurrency: int = 1,
) -> dict[str, float]:
    """Wave-local tail model; preserves per-token bursts but not cross-wave queues."""
    layers = cold_counts.shape[1]
    eligible = np.zeros_like(complete)
    eligible[:, lookahead:] = True
    successfully_predicted = complete & eligible & (cold_counts > 0)
    missed = (cold_counts > 0) & ~successfully_predicted
    transfer_slots = np.ceil(predicted / concurrency)
    reactive_slots = np.ceil(cold_counts / concurrency)
    lead_ms = lookahead * layer_ms

    predicted_wave_stall = np.zeros(cold_counts.shape, dtype=np.float64)
    predicted_wave_stall[successfully_predicted] = np.maximum(
        0.0,
        transfer_slots[successfully_predicted] * transfer_ms - lead_ms,
    )
    predicted_wave_stall[missed] = (
        reactive_slots[missed] * transfer_ms
    )
    reactive_wave_stall = reactive_slots * transfer_ms

    oracle_wave_stall = reactive_wave_stall.copy()
    oracle_eligible = eligible & (cold_counts > 0)
    oracle_wave_stall[oracle_eligible] = np.maximum(
        0.0,
        reactive_slots[oracle_eligible] * transfer_ms - lead_ms,
    )

    predicted_token = predicted_wave_stall.sum(axis=1)
    reactive_token = reactive_wave_stall.sum(axis=1)
    oracle_token = oracle_wave_stall.sum(axis=1)
    p99_predicted = _percentile(predicted_token, 0.99)
    p99_reactive = _percentile(reactive_token, 0.99)
    p99_oracle = _percentile(oracle_token, 0.99)
    mean_predicted = float(predicted_token.mean())
    mean_reactive = float(reactive_token.mean())
    mean_oracle = float(oracle_token.mean())
    oracle_reduction = 1.0 - mean_oracle / mean_reactive if mean_reactive else 1.0
    predictor_reduction = (
        1.0 - mean_predicted / mean_reactive if mean_reactive else 1.0
    )
    return {
        "mean_reactive_stall_ms": mean_reactive,
        "p50_reactive_stall_ms": _percentile(reactive_token, 0.50),
        "p95_reactive_stall_ms": _percentile(reactive_token, 0.95),
        "p99_reactive_stall_ms": p99_reactive,
        "mean_predictive_stall_ms": mean_predicted,
        "p50_predictive_stall_ms": _percentile(predicted_token, 0.50),
        "p95_predictive_stall_ms": _percentile(predicted_token, 0.95),
        "p99_predictive_stall_ms": p99_predicted,
        "mean_oracle_stall_ms": mean_oracle,
        "p50_oracle_stall_ms": _percentile(oracle_token, 0.50),
        "p95_oracle_stall_ms": _percentile(oracle_token, 0.95),
        "p99_oracle_stall_ms": p99_oracle,
        "mean_stall_reduction_vs_reactive": predictor_reduction,
        "p99_stall_reduction_vs_reactive": (
            1.0 - p99_predicted / p99_reactive if p99_reactive else 1.0
        ),
        "mean_oracle_stall_reduction": oracle_reduction,
        "mean_oracle_recovery": (
            predictor_reduction / oracle_reduction if oracle_reduction > 0 else 1.0
        ),
        "modeled_p99_tpot_ms": base_tpot_ms + p99_predicted,
        "reactive_p99_tpot_ms": base_tpot_ms + p99_reactive,
        "oracle_p99_tpot_ms": base_tpot_ms + p99_oracle,
        "p99_tpot_ratio_vs_reactive": (
            (base_tpot_ms + p99_predicted) / (base_tpot_ms + p99_reactive)
        ),
        "p99_tpot_slowdown_vs_all_resident": (
            (base_tpot_ms + p99_predicted) / base_tpot_ms
        ),
        "deadline_feasible_predicted_wave_fraction": (
            float(
                (
                    predicted_wave_stall[successfully_predicted] == 0
                ).sum()
                / successfully_predicted.sum()
            )
            if successfully_predicted.any()
            else 1.0
        ),
        "deadline_feasible_useful_objects": int(
            cold_counts[
                successfully_predicted
                & (predicted_wave_stall == 0)
            ].sum()
        ),
        "late_useful_objects": int(
            cold_counts[
                successfully_predicted
                & (predicted_wave_stall > 0)
            ].sum()
        ),
        "missed_or_unpredictable_objects": int(cold_counts[missed].sum()),
    }


def _queue_replay(
    *,
    cold_counts: np.ndarray,
    complete: np.ndarray,
    predicted: np.ndarray,
    lookahead: int,
    layer_ms: float,
    transfer_ms: float,
    base_tpot_ms: float,
    concurrency: int,
) -> dict[str, float]:
    """Small selected-point FCFS replay with cross-wave queueing within a token."""
    token_stalls = np.zeros(cold_counts.shape[0], dtype=np.float64)
    for token_index in range(cold_counts.shape[0]):
        lanes = np.zeros(concurrency, dtype=np.float64)
        useful_finish: dict[int, float] = {}
        now = 0.0
        for layer in range(cold_counts.shape[1]):
            demand = int(cold_counts[token_index, layer])
            if demand:
                if layer >= lookahead and complete[token_index, layer]:
                    required_finish = useful_finish.get(layer, now)
                else:
                    finishes = []
                    for _ in range(demand):
                        lane = int(np.argmin(lanes))
                        start = max(now, float(lanes[lane]))
                        lanes[lane] = start + transfer_ms
                        finishes.append(float(lanes[lane]))
                    required_finish = max(finishes, default=now)
                stall = max(0.0, required_finish - now)
                token_stalls[token_index] += stall
                now += stall

            target = layer + lookahead
            if target < cold_counts.shape[1]:
                jobs = int(predicted[token_index, target])
                useful = (
                    int(cold_counts[token_index, target])
                    if complete[token_index, target]
                    else 0
                )
                useful_positions = set(
                    np.linspace(0, jobs - 1, useful, dtype=int).tolist()
                ) if useful and jobs else set()
                finishes = []
                for job in range(jobs):
                    lane = int(np.argmin(lanes))
                    start = max(now, float(lanes[lane]))
                    lanes[lane] = start + transfer_ms
                    if job in useful_positions:
                        finishes.append(float(lanes[lane]))
                if finishes:
                    useful_finish[target] = max(finishes)
            now += layer_ms
    return {
        "mean_queue_replay_stall_ms": float(token_stalls.mean()),
        "p95_queue_replay_stall_ms": _percentile(token_stalls, 0.95),
        "p99_queue_replay_stall_ms": _percentile(token_stalls, 0.99),
        "queue_replay_p99_tpot_ms": base_tpot_ms + _percentile(token_stalls, 0.99),
    }


def _ax1(
    *,
    config: dict[str, Any],
    cold_by_capacity: dict[int, np.ndarray],
    layer_ms: float,
    base_tpot_ms: float,
    expert_bytes: int,
    layers: int,
    measured_startup_us: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    seed = int(config["seed"])
    rows: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    profile_cache: dict[tuple[int, int, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for capacity in config["residency"]["experts_per_layer"]:
        capacity = int(capacity)
        cold_counts = cold_by_capacity[capacity]
        for lookahead in config["residency"]["lookaheads"]:
            lookahead = int(lookahead)
            eligible_counts = cold_counts.copy()
            eligible_counts[:, :lookahead] = 0
            for coverage in config["predictor"]["complete_coverages"]:
                coverage = float(coverage)
                complete, coverage_info = _correlated_complete_mask(
                    eligible_counts,
                    coverage,
                    seed=seed + 1009 * capacity + 97 * lookahead,
                )
                for amplification in config["predictor"]["amplifications"]:
                    amplification = float(amplification)
                    predicted, amplification_info = _predicted_counts(
                        eligible_counts, complete, amplification
                    )
                    integrity.append(
                        {
                            "capacity_experts_per_layer": capacity,
                            "lookahead_layers": lookahead,
                            **coverage_info,
                            **amplification_info,
                            "miss_generation": "deterministic_block_correlated_wave_misses",
                            "evidence_coverage": "assumed_predictor",
                            "evidence_demand": "trace_derived",
                        }
                    )
                    profile_cache[(capacity, lookahead, coverage, amplification)] = (
                        complete,
                        predicted,
                    )
                    for bandwidth in config["cold_tier"]["bandwidth_gbps"]:
                        bandwidth = float(bandwidth)
                        move_ms = _transfer_ms(
                            expert_bytes, bandwidth, measured_startup_us
                        )
                        metrics = _local_stall_metrics(
                            cold_counts=cold_counts,
                            complete=complete,
                            predicted=predicted,
                            lookahead=lookahead,
                            layer_ms=layer_ms,
                            transfer_ms=move_ms,
                            base_tpot_ms=base_tpot_ms,
                        )
                        cold_eligible = int(eligible_counts.sum())
                        useful = int(
                            np.where(complete, eligible_counts, 0).sum()
                        )
                        total_predicted = int(predicted.sum())
                        mean_cold = cold_eligible / max(1, eligible_counts.size)
                        headroom = (
                            bandwidth
                            * 1e6
                            * lookahead
                            * layer_ms
                            / (mean_cold * expert_bytes)
                            if mean_cold
                            else math.inf
                        )
                        rows.append(
                            {
                                "track": "AX1",
                                "capacity_experts_per_layer": capacity,
                                "lookahead_layers": lookahead,
                                "requested_complete_cold_set_coverage": coverage,
                                "realized_complete_cold_set_coverage": (
                                    coverage_info[
                                        "realized_complete_cold_set_coverage"
                                    ]
                                ),
                                "requested_amplification": amplification,
                                "realized_amplification": (
                                    amplification_info[
                                        "realized_predicted_to_useful_amplification"
                                    ]
                                ),
                                "bandwidth_gbps": bandwidth,
                                "startup_latency_us": measured_startup_us,
                                "transfer_concurrency": 1,
                                "expert_size_mib": expert_bytes / MIB,
                                "expert_transfer_ms": move_ms,
                                "fast_tier_expert_gib": capacity
                                * layers
                                * expert_bytes
                                / GIB,
                                "offloaded_expert_gib": (64 - capacity)
                                * layers
                                * expert_bytes
                                / GIB,
                                "eligible_cold_demand_objects": cold_eligible,
                                "useful_predicted_objects": useful,
                                "false_predicted_objects": total_predicted - useful,
                                "cold_demand_bytes": cold_eligible * expert_bytes,
                                "useful_predicted_bytes": useful * expert_bytes,
                                "false_predicted_bytes": (
                                    total_predicted - useful
                                )
                                * expert_bytes,
                                "candidate_movement_bytes": (
                                    total_predicted * expert_bytes
                                ),
                                "deadline_feasible_useful_bytes": (
                                    metrics[
                                        "deadline_feasible_useful_objects"
                                    ]
                                    * expert_bytes
                                ),
                                "late_useful_bytes": (
                                    metrics["late_useful_objects"]
                                    * expert_bytes
                                ),
                                "missed_or_unpredictable_bytes": (
                                    metrics[
                                        "missed_or_unpredictable_objects"
                                    ]
                                    * expert_bytes
                                ),
                                "mean_eligible_cold_objects_per_wave": mean_cold,
                                "cold_service_headroom": headroom,
                                "cold_service_pressure_with_amplification": (
                                    amplification / headroom
                                ),
                                **metrics,
                                "tail_model": "trace_ordered_per_token_wave_local",
                                "evidence_expert_and_timing": "measured",
                                "evidence_demand": "trace_derived",
                                "evidence_predictor": "assumed",
                                "evidence_hardware": (
                                    "measured_anchor"
                                    if math.isclose(
                                        bandwidth,
                                        float(
                                            config["cold_tier"]["bandwidth_gbps"][1]
                                        ),
                                    )
                                    else "hypothetical"
                                ),
                            }
                        )

    measured_bandwidth = float(config["cold_tier"]["bandwidth_gbps"][1])
    pareto: list[dict[str, Any]] = []
    quality_profiles = (
        ("predictive_C99_A1.5", 0.99, 1.5),
        ("predictive_C999_A1.25", 0.999, 1.25),
    )
    for capacity in (int(value) for value in config["residency"]["experts_per_layer"]):
        capacity_rows = [
            row
            for row in rows
            if row["capacity_experts_per_layer"] == capacity
            and math.isclose(row["bandwidth_gbps"], measured_bandwidth)
        ]
        reactive = min(
            capacity_rows,
            key=lambda row: row["reactive_p99_tpot_ms"],
        )
        pareto.append(
            {
                "policy": "reactive_offload",
                "quality_profile": "none",
                "capacity_experts_per_layer": capacity,
                "fast_tier_expert_gib": reactive["fast_tier_expert_gib"],
                "offloaded_expert_gib": reactive["offloaded_expert_gib"],
                "selected_lookahead": 0,
                "modeled_p99_tpot_ms": reactive["reactive_p99_tpot_ms"],
                "p99_slowdown_vs_all_resident": (
                    reactive["reactive_p99_tpot_ms"] / base_tpot_ms
                ),
                "bandwidth_gbps": measured_bandwidth,
                "evidence": "measured_anchor_plus_trace_derived_model",
            }
        )
        for name, coverage, amplification in quality_profiles:
            candidates = [
                row
                for row in capacity_rows
                if math.isclose(
                    row["requested_complete_cold_set_coverage"], coverage
                )
                and math.isclose(row["requested_amplification"], amplification)
            ]
            best = min(candidates, key=lambda row: row["modeled_p99_tpot_ms"])
            pareto.append(
                {
                    "policy": "predictive_offload",
                    "quality_profile": name,
                    "capacity_experts_per_layer": capacity,
                    "fast_tier_expert_gib": best["fast_tier_expert_gib"],
                    "offloaded_expert_gib": best["offloaded_expert_gib"],
                    "selected_lookahead": best["lookahead_layers"],
                    "modeled_p99_tpot_ms": best["modeled_p99_tpot_ms"],
                    "p99_slowdown_vs_all_resident": (
                        best["modeled_p99_tpot_ms"] / base_tpot_ms
                    ),
                    "bandwidth_gbps": measured_bandwidth,
                    "evidence": "assumed_predictor_trace_calibrated_projection",
                }
            )

        oracle_candidates = []
        cold_counts = cold_by_capacity[capacity]
        for lookahead in (int(value) for value in config["residency"]["lookaheads"]):
            eligible = cold_counts.copy()
            eligible[:, :lookahead] = 0
            complete = eligible > 0
            predicted = eligible.astype(np.int64)
            oracle_candidates.append(
                (
                    lookahead,
                    _local_stall_metrics(
                        cold_counts=cold_counts,
                        complete=complete,
                        predicted=predicted,
                        lookahead=lookahead,
                        layer_ms=layer_ms,
                        transfer_ms=_transfer_ms(
                            expert_bytes, measured_bandwidth, measured_startup_us
                        ),
                        base_tpot_ms=base_tpot_ms,
                    ),
                )
            )
        oracle_lookahead, oracle_metrics = min(
            oracle_candidates, key=lambda item: item[1]["modeled_p99_tpot_ms"]
        )
        pareto.append(
            {
                "policy": "oracle_offload",
                "quality_profile": "C1_A1",
                "capacity_experts_per_layer": capacity,
                "fast_tier_expert_gib": capacity * layers * expert_bytes / GIB,
                "offloaded_expert_gib": (64 - capacity)
                * layers
                * expert_bytes
                / GIB,
                "selected_lookahead": oracle_lookahead,
                "modeled_p99_tpot_ms": oracle_metrics["modeled_p99_tpot_ms"],
                "p99_slowdown_vs_all_resident": (
                    oracle_metrics["modeled_p99_tpot_ms"] / base_tpot_ms
                ),
                "bandwidth_gbps": measured_bandwidth,
                "evidence": "oracle_trace_calibrated_projection",
            }
        )
    pareto.append(
        {
            "policy": "all_resident_reference",
            "quality_profile": "not_applicable",
            "capacity_experts_per_layer": 64,
            "fast_tier_expert_gib": 64 * layers * expert_bytes / GIB,
            "offloaded_expert_gib": 0.0,
            "selected_lookahead": 0,
            "modeled_p99_tpot_ms": base_tpot_ms,
            "p99_slowdown_vs_all_resident": 1.0,
            "bandwidth_gbps": math.inf,
            "evidence": "measured_current_model_reference",
        }
    )

    queue_points = (
        (16, 3, 0.999, 1.0, measured_bandwidth),
        (16, 9, 0.99, 1.5, measured_bandwidth),
        (32, 3, 0.99, 1.5, measured_bandwidth),
        (16, 3, 0.99, 1.5, 64.0),
    )
    queue_rows: list[dict[str, Any]] = []
    for capacity, lookahead, coverage, amplification, bandwidth in queue_points:
        complete, predicted = profile_cache[
            (capacity, lookahead, coverage, amplification)
        ]
        local = next(
            row
            for row in rows
            if row["capacity_experts_per_layer"] == capacity
            and row["lookahead_layers"] == lookahead
            and math.isclose(
                row["requested_complete_cold_set_coverage"], coverage
            )
            and math.isclose(row["requested_amplification"], amplification)
            and math.isclose(row["bandwidth_gbps"], bandwidth)
        )
        queue_rows.append(
            {
                "capacity_experts_per_layer": capacity,
                "lookahead_layers": lookahead,
                "coverage": coverage,
                "amplification": amplification,
                "bandwidth_gbps": bandwidth,
                "wave_local_p99_stall_ms": local["p99_predictive_stall_ms"],
                **_queue_replay(
                    cold_counts=cold_by_capacity[capacity],
                    complete=complete,
                    predicted=predicted,
                    lookahead=lookahead,
                    layer_ms=layer_ms,
                    transfer_ms=float(local["expert_transfer_ms"]),
                    base_tpot_ms=base_tpot_ms,
                    concurrency=1,
                ),
                "evidence": "selected_trace_ordered_fcfs_sensitivity",
            }
        )
    return rows, integrity, pareto, queue_rows


def _ax2(
    *,
    config: dict[str, Any],
    cold_by_capacity: dict[int, np.ndarray],
    layer_ms: float,
    expert_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inverse: list[dict[str, Any]] = []
    sources: list[tuple[str, int | str, int, float]] = []
    for capacity, cold in cold_by_capacity.items():
        for lookahead in (int(value) for value in config["residency"]["lookaheads"]):
            eligible = cold[:, lookahead:]
            sources.append(
                (
                    "trace_derived_olmoe",
                    capacity,
                    lookahead,
                    float(eligible.mean()) if eligible.size else 0.0,
                )
            )
    for unique in config["granularity"]["normalized_unique_cold_objects"]:
        for lookahead in (int(value) for value in config["residency"]["lookaheads"]):
            sources.append(
                ("normalized_sensitivity", "normalized", lookahead, float(unique))
            )

    for source, capacity, lookahead, unique in sources:
        lead_ms = lookahead * layer_ms
        for object_mib in config["granularity"]["object_mib"]:
            object_mib = float(object_mib)
            object_bytes = object_mib * MIB
            for amplification in config["predictor"]["amplifications"]:
                amplification = float(amplification)
                transferred_objects = amplification * unique
                for startup_us in config["cold_tier"]["startup_latency_us"]:
                    startup_us = float(startup_us)
                    for concurrency in config["cold_tier"]["transfer_concurrency"]:
                        concurrency = int(concurrency)
                        startup_total_ms = (
                            math.ceil(transferred_objects / concurrency)
                            * startup_us
                            / 1000.0
                        )
                        payload_bytes = transferred_objects * object_bytes
                        remaining_ms = lead_ms - startup_total_ms
                        minimum_bandwidth = (
                            payload_bytes / (remaining_ms * 1e6)
                            if remaining_ms > 0
                            else math.inf
                        )
                        measured_bandwidth = float(
                            config["cold_tier"]["bandwidth_gbps"][1]
                        )
                        measured_service_ms = startup_total_ms + payload_bytes / (
                            measured_bandwidth * 1e6
                        )
                        maximum_amplification = lead_ms / (
                            unique
                            * (
                                object_bytes / (measured_bandwidth * 1e6)
                                + startup_us / (1000.0 * concurrency)
                            )
                        ) if unique else math.inf
                        maximum_object_mib = (
                            max(0.0, remaining_ms)
                            * measured_bandwidth
                            * 1e6
                            / max(transferred_objects, 1e-12)
                            / MIB
                        )
                        inverse.append(
                            {
                                "track": "AX2",
                                "demand_source": source,
                                "capacity_experts_per_layer": capacity,
                                "lookahead_layers": lookahead,
                                "unique_cold_objects_per_wave": unique,
                                "object_size_mib": object_mib,
                                "amplification": amplification,
                                "startup_latency_us": startup_us,
                                "transfer_concurrency": concurrency,
                                "lead_time_ms": lead_ms,
                                "payload_mib_per_wave": payload_bytes / MIB,
                                "startup_service_ms": startup_total_ms,
                                "minimum_bandwidth_gbps": minimum_bandwidth,
                                "maximum_amplification_at_measured_pcie": (
                                    maximum_amplification
                                ),
                                "maximum_object_mib_at_measured_pcie": (
                                    maximum_object_mib
                                ),
                                "measured_pcie_service_ms": measured_service_ms,
                                "measured_pcie_timing_feasible": (
                                    measured_service_ms <= lead_ms
                                ),
                                "isolated_object_timing_ratio_at_measured_pcie": (
                                    _transfer_ms(
                                        object_bytes,
                                        measured_bandwidth,
                                        startup_us,
                                    )
                                    / lead_ms
                                ),
                                "per_object_recall_for_C99_independent_only": (
                                    0.99 ** (1.0 / unique) if unique else 1.0
                                ),
                                "evidence_demand": (
                                    "trace_derived"
                                    if source == "trace_derived_olmoe"
                                    else "normalized_assumption"
                                ),
                                "evidence_hardware": "analytical_sweep",
                            }
                        )

    phase: list[dict[str, Any]] = []
    for capacity, cold in cold_by_capacity.items():
        for lookahead in (int(value) for value in config["residency"]["lookaheads"]):
            eligible = cold[:, lookahead:]
            mean_cold = float(eligible.mean()) if eligible.size else 0.0
            for bandwidth in config["cold_tier"]["bandwidth_gbps"]:
                bandwidth = float(bandwidth)
                raw_headroom = (
                    bandwidth
                    * 1e6
                    * lookahead
                    * layer_ms
                    / (mean_cold * expert_bytes)
                )
                for coverage in config["predictor"]["complete_coverages"]:
                    coverage = float(coverage)
                    for amplification in config["predictor"]["amplifications"]:
                        amplification = float(amplification)
                        effective = raw_headroom / amplification
                        oracle = min(1.0, raw_headroom)
                        benefit = coverage * min(1.0, effective)
                        recovery = benefit / oracle if oracle else 1.0
                        profitable = benefit >= 0.25 and recovery >= 0.50
                        slo_candidate = (
                            coverage >= 0.99 and effective >= 1.25
                        )
                        if slo_candidate:
                            category = "slo_candidate"
                        elif profitable:
                            category = "profitable_tail_risk"
                        elif effective < 1.0:
                            category = "service_limited"
                        else:
                            category = "reliability_limited"
                        phase.append(
                            {
                                "track": "AX2",
                                "capacity_experts_per_layer": capacity,
                                "lookahead_layers": lookahead,
                                "bandwidth_gbps": bandwidth,
                                "coverage": coverage,
                                "amplification": amplification,
                                "raw_cold_service_headroom": raw_headroom,
                                "effective_headroom_after_amplification": effective,
                                "first_order_stall_reduction": benefit,
                                "oracle_recovery": recovery,
                                "category": category,
                                "evidence_demand": "trace_derived",
                                "evidence_predictor": "assumed",
                                "evidence_hardware": (
                                    "measured_anchor"
                                    if math.isclose(
                                        bandwidth,
                                        float(
                                            config["cold_tier"]["bandwidth_gbps"][1]
                                        ),
                                    )
                                    else "hypothetical"
                                ),
                            }
                        )
    return inverse, phase


def _ax3(
    *,
    config: dict[str, Any],
    layer_ms: float,
    expert_bytes: int,
    top_k: int,
    token_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    short_lookaheads = [
        int(value)
        for value in config["residency"]["lookaheads"]
        if int(value) <= 3
    ]
    coverage_profiles: dict[
        tuple[int, float], tuple[np.ndarray, dict[str, float | int]]
    ] = {}
    for lookahead in short_lookaheads:
        eligible = np.ones((token_count, 16), dtype=np.int16)
        eligible[:, :lookahead] = 0
        for coverage in config["predictor"]["complete_coverages"]:
            coverage = float(coverage)
            complete, coverage_info = _correlated_complete_mask(
                eligible,
                coverage,
                seed=seed + 1009 * lookahead + int(1000 * coverage),
            )
            incomplete_per_token = (~complete).sum(axis=1)
            coverage_profiles[(lookahead, coverage)] = (
                np.bincount(incomplete_per_token, minlength=17),
                coverage_info,
            )
    route_payload_mib = top_k * expert_bytes / MIB
    for capacity_mib in config["sram"]["capacity_mib"]:
        capacity_mib = float(capacity_mib)
        for bandwidth in config["sram"]["bandwidth_gbps"]:
            bandwidth = float(bandwidth)
            for startup_us in config["sram"]["startup_latency_us"]:
                startup_us = float(startup_us)
                for amplification in config["predictor"]["amplifications"]:
                    amplification = float(amplification)
                    transferred_mib = route_payload_mib * amplification
                    double_buffer_mib = 2.0 * transferred_mib
                    for object_mib in config["granularity"]["object_mib"]:
                        object_mib = float(object_mib)
                        objects = math.ceil(transferred_mib / object_mib)
                        payload_bytes = transferred_mib * MIB
                        service_ms = (
                            objects * startup_us / 1000.0
                            + payload_bytes / (bandwidth * 1e6)
                        )
                        for lookahead in short_lookaheads:
                            lead_ms = lookahead * layer_ms
                            capacity_feasible = double_buffer_mib <= capacity_mib
                            timing_feasible = service_ms <= lead_ms
                            if not capacity_feasible:
                                bottleneck = "staging_capacity_pollution"
                            elif not timing_feasible:
                                bulk_ms = payload_bytes / (bandwidth * 1e6)
                                startup_ms = objects * startup_us / 1000.0
                                bottleneck = (
                                    "startup_latency"
                                    if startup_ms > bulk_ms
                                    else "bulk_bandwidth"
                                )
                            else:
                                bottleneck = "physically_feasible"
                            remaining_ms = lead_ms - objects * startup_us / 1000.0
                            minimum_bandwidth = (
                                payload_bytes / (remaining_ms * 1e6)
                                if remaining_ms > 0
                                else math.inf
                            )
                            reactive_objects = math.ceil(
                                route_payload_mib / object_mib
                            )
                            reactive_service_ms = (
                                reactive_objects * startup_us / 1000.0
                                + route_payload_mib
                                * MIB
                                / (bandwidth * 1e6)
                            )
                            for coverage in config["predictor"][
                                "complete_coverages"
                            ]:
                                coverage = float(coverage)
                                (
                                    incomplete_histogram,
                                    coverage_info,
                                ) = coverage_profiles[
                                    (lookahead, coverage)
                                ]
                                if capacity_feasible:
                                    covered_stall = max(
                                        0.0, service_ms - lead_ms
                                    )
                                    incomplete_values = np.arange(17)
                                    token_stall_values = (
                                        (16 - incomplete_values) * covered_stall
                                        + incomplete_values * reactive_service_ms
                                    )
                                else:
                                    token_stall_values = np.full(
                                        17, 16 * reactive_service_ms
                                    )
                                deadline_miss = (
                                    token_count * 16
                                    if not capacity_feasible
                                    else (
                                        int(
                                            np.dot(
                                                np.arange(17),
                                                incomplete_histogram,
                                            )
                                        )
                                        + (
                                            token_count * 16
                                            - int(
                                                np.dot(
                                                    np.arange(17),
                                                    incomplete_histogram,
                                                )
                                            )
                                            if service_ms > lead_ms
                                            else 0
                                        )
                                    )
                                )
                                rows.append({
                                    "track": "AX3",
                                    "sram_capacity_mib": capacity_mib,
                                    "sram_bandwidth_gbps": bandwidth,
                                    "startup_latency_us": startup_us,
                                    "lookahead_layers": lookahead,
                                    "amplification": amplification,
                                    "requested_complete_wave_coverage": coverage,
                                    "realized_complete_wave_coverage": (
                                        coverage_info[
                                            "realized_complete_cold_set_coverage"
                                        ]
                                    ),
                                    "transfer_object_mib": object_mib,
                                    "route_payload_mib": route_payload_mib,
                                    "transferred_payload_mib": transferred_mib,
                                    "transfer_objects": objects,
                                    "double_buffer_required_mib": double_buffer_mib,
                                    "lead_time_ms": lead_ms,
                                    "service_ms": service_ms,
                                    "service_headroom": lead_ms / service_ms,
                                    "minimum_bandwidth_gbps": minimum_bandwidth,
                                    "reactive_route_service_ms": (
                                        reactive_service_ms
                                    ),
                                    "p50_added_staging_stall_ms": _histogram_percentile(
                                        token_stall_values,
                                        incomplete_histogram,
                                        0.50,
                                    ),
                                    "p95_added_staging_stall_ms": _histogram_percentile(
                                        token_stall_values,
                                        incomplete_histogram,
                                        0.95,
                                    ),
                                    "p99_added_staging_stall_ms": _histogram_percentile(
                                        token_stall_values,
                                        incomplete_histogram,
                                        0.99,
                                    ),
                                    "deadline_miss_wave_fraction": (
                                        deadline_miss / (token_count * 16)
                                    ),
                                    "capacity_feasible": capacity_feasible,
                                    "timing_feasible": timing_feasible,
                                    "physical_feasible": (
                                        capacity_feasible and timing_feasible
                                    ),
                                    "bottleneck": bottleneck,
                                    "minimum_complete_wave_coverage_for_1pct_wave_miss": 0.99,
                                    "rolling_global_staging": True,
                                    "double_buffered": True,
                                    "evidence_route_width_and_size": "measured",
                                    "evidence_predictor": "assumed",
                                    "evidence_sram": "hypothetical",
                                    "tail_semantics": (
                                        "correlated_wave_misses_sram_staging_only"
                                    ),
                                })
    return rows


def _architecture_comparison(
    *,
    pareto: list[dict[str, Any]],
    ax3: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reactive = next(
        row
        for row in pareto
        if row["policy"] == "reactive_offload"
        and int(row["capacity_experts_per_layer"]) == 16
    )
    predictive = next(
        row
        for row in pareto
        if row["quality_profile"] == "predictive_C99_A1.5"
        and int(row["capacity_experts_per_layer"]) == 16
    )
    oracle = next(
        row
        for row in pareto
        if row["policy"] == "oracle_offload"
        and int(row["capacity_experts_per_layer"]) == 16
    )
    resident = next(
        row for row in pareto if row["policy"] == "all_resident_reference"
    )
    staging = next(
        row
        for row in ax3
        if math.isclose(float(row["sram_capacity_mib"]), 512.0)
        and math.isclose(float(row["sram_bandwidth_gbps"]), 1000.0)
        and math.isclose(float(row["startup_latency_us"]), 0.1)
        and math.isclose(float(row["amplification"]), 1.5)
        and math.isclose(float(row["requested_complete_wave_coverage"]), 0.99)
        and math.isclose(float(row["transfer_object_mib"]), 12.0)
        and int(row["lookahead_layers"]) == 1
    )
    common = {
        "comparison_semantics": (
            "P99 analytical comparison; three-tier conservatively sums "
            "separate upstream and staging P99 without crediting SRAM compute benefit"
        )
    }
    return [
        {
            "architecture": "two_tier_reactive_host_hbm",
            "hbm_expert_gib": reactive["fast_tier_expert_gib"],
            "sram_mib": 0,
            "predictor_profile": "none",
            "modeled_p99_tpot_ms": reactive["modeled_p99_tpot_ms"],
            "evidence": reactive["evidence"],
            **common,
        },
        {
            "architecture": "two_tier_predictive_host_hbm",
            "hbm_expert_gib": predictive["fast_tier_expert_gib"],
            "sram_mib": 0,
            "predictor_profile": "C99_A1.5",
            "modeled_p99_tpot_ms": predictive["modeled_p99_tpot_ms"],
            "evidence": predictive["evidence"],
            **common,
        },
        {
            "architecture": "three_tier_predictive_host_hbm_sram",
            "hbm_expert_gib": predictive["fast_tier_expert_gib"],
            "sram_mib": 512,
            "predictor_profile": "C99_A1.5_both_tiers",
            "modeled_p99_tpot_ms": float(
                predictive["modeled_p99_tpot_ms"]
            )
            + float(staging["p99_added_staging_stall_ms"]),
            "evidence": (
                "assumed_predictor_trace_calibrated_upstream_plus_"
                "hypothetical_sram_staging"
            ),
            **common,
        },
        {
            "architecture": "two_tier_oracle_host_hbm",
            "hbm_expert_gib": oracle["fast_tier_expert_gib"],
            "sram_mib": 0,
            "predictor_profile": "C1_A1",
            "modeled_p99_tpot_ms": oracle["modeled_p99_tpot_ms"],
            "evidence": oracle["evidence"],
            **common,
        },
        {
            "architecture": "all_resident_hbm_reference",
            "hbm_expert_gib": resident["fast_tier_expert_gib"],
            "sram_mib": 0,
            "predictor_profile": "not_applicable",
            "modeled_p99_tpot_ms": resident["modeled_p99_tpot_ms"],
            "evidence": resident["evidence"],
            **common,
        },
    ]


def _anchor_reproduction(
    *,
    config: dict[str, Any],
    measurement: dict[str, Any],
    cold_by_capacity: dict[int, np.ndarray],
    layer_ms: float,
    expert_bytes: int,
) -> list[dict[str, Any]]:
    bandwidth = float(measurement["transfer"]["fit"]["effective_bandwidth_gbps"])
    startup_us = float(measurement["transfer"]["fit"]["startup_ms"]) * 1000.0
    modeled_transfer = _transfer_ms(expert_bytes, bandwidth, startup_us)
    archived_transfer = float(measurement["transfer"]["exact_expert_median_ms"])
    rows = [
        {
            "anchor": "exact_12mib_transfer",
            "archived_value": archived_transfer,
            "reproduced_value": modeled_transfer,
            "absolute_difference": abs(modeled_transfer - archived_transfer),
            "tolerance": 1e-4,
            "passed": abs(modeled_transfer - archived_transfer) <= 1e-4,
        }
    ]
    h5_rows: list[dict[str, str]] = []
    h5_path = Path(config["h5_analysis"]) / "h5_physical_context.csv"
    with h5_path.open("r", encoding="utf-8", newline="") as handle:
        h5_rows = list(csv.DictReader(handle))
    for capacity, lookahead in ((16, 3), (32, 3), (16, 9)):
        cold = cold_by_capacity[capacity][:, lookahead:]
        nonempty = cold[cold > 0]
        mean_all_waves = float(cold.sum() / cold.size)
        reproduced = (
            lookahead
            * layer_ms
            / (mean_all_waves * archived_transfer)
        )
        archived = next(
            float(row["cold_service_headroom"])
            for row in h5_rows
            if int(row["capacity"]) == capacity
            and int(row["lookahead"]) == lookahead
            and math.isclose(float(row["bandwidth_scale"]), 1.0)
        )
        rows.append(
            {
                "anchor": f"h5_headroom_K{capacity}_D{lookahead}",
                "archived_value": archived,
                "reproduced_value": reproduced,
                "absolute_difference": abs(reproduced - archived),
                "tolerance": 1e-9,
                "passed": abs(reproduced - archived) <= 1e-9,
            }
        )
    return rows


def analyze_architecture(experiment_config: dict[str, Any]) -> dict[str, Any]:
    output = Path(experiment_config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    run = Path(experiment_config["h1_run"])
    measurement = json.loads(
        Path(experiment_config["h4_measurement"]).read_text(encoding="utf-8")
    )
    model_report = json.loads(
        (run / "model_report.json").read_text(encoding="utf-8")
    )
    layers = int(model_report["router_count"])
    expert_sizes = {
        int(router["expert_bytes_each"]) for router in model_report["routers"]
    }
    top_ks = {int(router["top_k"]) for router in model_report["routers"]}
    if len(expert_sizes) != 1 or len(top_ks) != 1:
        raise ValueError("AX requires uniform expert size and top-k in this testbed")
    expert_bytes = expert_sizes.pop()
    top_k = top_ks.pop()
    layer_ms = float(measurement["decode"]["effective_inter_moe_layer_ms"])
    base_tpot_ms = float(measurement["decode"]["median_forward_ms"])
    measured_startup_us = (
        float(measurement["transfer"]["fit"]["startup_ms"]) * 1000.0
    )
    waves, token_count = _decode_waves(run)
    cold_by_capacity = {
        int(capacity): _cold_count_matrix(
            waves=waves,
            token_count=token_count,
            layers=layers,
            capacity=int(capacity),
        )
        for capacity in experiment_config["residency"]["experts_per_layer"]
    }

    anchors = _anchor_reproduction(
        config=experiment_config,
        measurement=measurement,
        cold_by_capacity=cold_by_capacity,
        layer_ms=layer_ms,
        expert_bytes=expert_bytes,
    )
    if not all(bool(row["passed"]) for row in anchors):
        raise RuntimeError("AX anchor reproduction failed")

    ax1, integrity, pareto, queue = _ax1(
        config=experiment_config,
        cold_by_capacity=cold_by_capacity,
        layer_ms=layer_ms,
        base_tpot_ms=base_tpot_ms,
        expert_bytes=expert_bytes,
        layers=layers,
        measured_startup_us=measured_startup_us,
    )
    ax2_inverse, ax2_phase = _ax2(
        config=experiment_config,
        cold_by_capacity=cold_by_capacity,
        layer_ms=layer_ms,
        expert_bytes=expert_bytes,
    )
    ax3 = _ax3(
        config=experiment_config,
        layer_ms=layer_ms,
        expert_bytes=expert_bytes,
        top_k=top_k,
        token_count=token_count,
        seed=int(experiment_config["seed"]),
    )
    architecture_comparison = _architecture_comparison(
        pareto=pareto,
        ax3=ax3,
    )

    _write_csv(output / "anchor_reproduction.csv", anchors)
    _write_csv(output / "ax1_envelope.csv", ax1)
    _write_csv(output / "ax1_predictor_integrity.csv", integrity)
    _write_csv(output / "ax1_pareto.csv", pareto)
    _write_csv(output / "ax1_queue_sensitivity.csv", queue)
    _write_csv(output / "ax2_inverse_bounds.csv", ax2_inverse)
    _write_csv(output / "ax2_phase_points.csv", ax2_phase)
    _write_csv(output / "ax3_staging.csv", ax3)
    _write_csv(output / "architecture_comparison.csv", architecture_comparison)

    measured_bw = float(measurement["transfer"]["fit"]["effective_bandwidth_gbps"])
    predictive_pareto = [
        row
        for row in pareto
        if row["quality_profile"] == "predictive_C99_A1.5"
    ]
    reactive_pareto = {
        int(row["capacity_experts_per_layer"]): row
        for row in pareto
        if row["policy"] == "reactive_offload"
    }
    ax3_physical: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in ax3:
        key = (
            row["sram_capacity_mib"],
            row["sram_bandwidth_gbps"],
            row["startup_latency_us"],
            row["lookahead_layers"],
            row["amplification"],
            row["transfer_object_mib"],
        )
        ax3_physical[key] = row
    ax3_feasible = [
        row for row in ax3_physical.values() if row["physical_feasible"]
    ]
    summary = {
        "track": "AX",
        "state": "complete_pending_human_review",
        "evidence_grade": "trace_calibrated_assumption_driven_analytical_projection",
        "trace": {
            "run": str(run),
            "decode_tokens": token_count,
            "decode_waves": len(waves),
            "layers": layers,
            "top_k": top_k,
        },
        "measured_inputs": {
            "expert_size_mib": expert_bytes / MIB,
            "base_decode_tpot_ms": base_tpot_ms,
            "effective_inter_moe_layer_ms": layer_ms,
            "h2d_bandwidth_gbps": measured_bw,
            "h2d_startup_latency_us": measured_startup_us,
        },
        "grid": {
            "ax1_envelope_rows": len(ax1),
            "predictor_integrity_rows": len(integrity),
            "ax2_inverse_rows": len(ax2_inverse),
            "ax2_phase_rows": len(ax2_phase),
            "ax3_staging_rows": len(ax3),
            "ax3_factorized_physical_cells": len(ax3_physical),
        },
        "ax1_headline": [
            {
                "capacity_experts_per_layer": int(
                    row["capacity_experts_per_layer"]
                ),
                "fast_tier_expert_gib": row["fast_tier_expert_gib"],
                "selected_lookahead": int(row["selected_lookahead"]),
                "predictive_p99_tpot_ms": row["modeled_p99_tpot_ms"],
                "reactive_p99_tpot_ms": reactive_pareto[
                    int(row["capacity_experts_per_layer"])
                ]["modeled_p99_tpot_ms"],
                "p99_improvement_vs_reactive": 1.0
                - row["modeled_p99_tpot_ms"]
                / reactive_pareto[int(row["capacity_experts_per_layer"])][
                    "modeled_p99_tpot_ms"
                ],
            }
            for row in predictive_pareto
        ],
        "ax2_headline": {
            "whole_expert_k16_a1_minimum_bandwidth_gbps_by_lookahead": {
                str(delta): min(
                    float(row["minimum_bandwidth_gbps"])
                    for row in ax2_inverse
                    if row["demand_source"] == "trace_derived_olmoe"
                    and int(row["capacity_experts_per_layer"]) == 16
                    and int(row["lookahead_layers"]) == delta
                    and math.isclose(float(row["object_size_mib"]), 12.0)
                    and math.isclose(float(row["amplification"]), 1.0)
                    and math.isclose(
                        float(row["startup_latency_us"]), measured_startup_us,
                        abs_tol=0.1,
                    )
                    and int(row["transfer_concurrency"]) == 1
                )
                for delta in (1, 3, 6, 9)
            }
        },
        "ax3_headline": {
            "physically_feasible_cells": len(ax3_feasible),
            "physical_cells": len(ax3_physical),
            "coverage_conditioned_rows": len(ax3),
            "minimum_whole_expert_double_buffer_mib_A1": 2
            * top_k
            * expert_bytes
            / MIB,
            "minimum_whole_expert_double_buffer_mib_A2": 4
            * top_k
            * expert_bytes
            / MIB,
            "minimum_feasible_sram_capacity_mib": min(
                float(row["sram_capacity_mib"]) for row in ax3_feasible
            ),
        },
        "interpretation_boundary": (
            "Projected points combine measured current-testbed anchors, "
            "trace-derived demand, assumed future predictor quality, and "
            "hypothetical hardware. They are not measured speedups and do not "
            "show that current OLMoE attains the assumed quality."
        ),
        "outputs": {
            "ax1": str(output / "ax1_envelope.csv"),
            "pareto": str(output / "ax1_pareto.csv"),
            "ax2": str(output / "ax2_inverse_bounds.csv"),
            "ax3": str(output / "ax3_staging.csv"),
            "architecture_comparison": str(
                output / "architecture_comparison.csv"
            ),
        },
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "evidence_ledger.json",
        {
            "measured": [
                "12 MiB OLMoE expert",
                "0.639 ms effective inter-MoE-layer interval",
                "hook-free median decode forward",
                "pinned host-to-device transfer fit",
            ],
            "trace_derived": [
                "per-token per-layer LRU cold demand at K=8,16,32",
                "cold-demand burst and tail distribution",
            ],
            "assumed_predictor": [
                "wave-level complete cold-set coverage",
                "predicted/useful byte amplification",
                "block-correlated false-negative waves",
            ],
            "hypothetical_hardware": [
                "non-measured tier bandwidth and latency points",
                "transfer concurrency",
                "sub-expert transfer objects",
                "rolling software-managed SRAM capacities and bandwidths",
            ],
        },
    )
    _write_report(
        output,
        summary,
        pareto,
        queue,
        ax3,
        architecture_comparison,
    )
    return summary


def _write_report(
    output: Path,
    summary: dict[str, Any],
    pareto: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    ax3: list[dict[str, Any]],
    architecture_comparison: list[dict[str, Any]],
) -> None:
    lines = [
        "# AX assumption-driven architecture exploration",
        "",
        "**State:** complete; pending human figure review",
        "",
        "## Plain-language result",
        "",
        "A predictive hierarchy has a real design window, but reliability and "
        "speculative traffic are first-class hardware parameters. More "
        "lookahead trades prediction difficulty for lower required bandwidth; "
        "it does not make false-negative tail stalls disappear.",
        "",
        "## AX1 — host or pooled memory to HBM",
        "",
        "At the measured PCIe anchor and the assumed C=99%, A=1.5× future "
        "router, the best modeled points are:",
        "",
        "| HBM residents/layer | HBM expert GiB | Best Δ | P99 TPOT | Reactive P99 | Improvement |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["ax1_headline"]:
        lines.append(
            f"| {row['capacity_experts_per_layer']} | "
            f"{row['fast_tier_expert_gib']:.1f} | "
            f"{row['selected_lookahead']} | "
            f"{row['predictive_p99_tpot_ms']:.2f} ms | "
            f"{row['reactive_p99_tpot_ms']:.2f} ms | "
            f"{100 * row['p99_improvement_vs_reactive']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "These are capacity-enabling comparisons against reactive offload "
            "on the same hierarchy, not speedups over all-HBM execution.",
            "",
            "The selected FCFS queue replay is intentionally more pessimistic "
            "than the wave-local envelope:",
            "",
            "| K | Δ | C | A | BW | Wave-local P99 stall | Queue P99 stall |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in queue:
        lines.append(
            f"| {row['capacity_experts_per_layer']} | "
            f"{row['lookahead_layers']} | {row['coverage']:.3f} | "
            f"{row['amplification']:.2f}× | {row['bandwidth_gbps']:.1f} GB/s | "
            f"{row['wave_local_p99_stall_ms']:.2f} ms | "
            f"{row['p99_queue_replay_stall_ms']:.2f} ms |"
        )
    inverse = summary["ax2_headline"][
        "whole_expert_k16_a1_minimum_bandwidth_gbps_by_lookahead"
    ]
    lines.extend(
        [
            "",
            "## AX2 — inverse requirements",
            "",
            "For trace-derived K=16 cold demand, 12 MiB objects, A=1×, and one "
            "transfer lane, the minimum first-order bandwidth falls with "
            "lookahead:",
            "",
            "| Δ | Minimum bandwidth |",
            "|---:|---:|",
        ]
    )
    for delta, bandwidth in inverse.items():
        lines.append(f"| {delta} | {bandwidth:.1f} GB/s |")
    lines.extend(
        [
            "",
            "This inverse law is the central co-design lever: required bandwidth "
            "scales approximately as A/Δ. Coverage is orthogonal: it controls "
            "how often the synchronous tail still takes the cold path.",
            "",
            "## AX3 — rolling SRAM staging",
            "",
            f"OLMoE's top-8 route carries 96 MiB of whole-expert weights per "
            f"layer. A rolling double buffer therefore needs at least "
            f"{summary['ax3_headline']['minimum_whole_expert_double_buffer_mib_A1']:.0f} "
            f"MiB at A=1× and "
            f"{summary['ax3_headline']['minimum_whole_expert_double_buffer_mib_A2']:.0f} "
            "MiB at A=2×. This makes capacity pollution, not raw SRAM bandwidth, "
            "the first constraint for the frozen 32–512 MiB range.",
            "",
            f"{summary['ax3_headline']['physically_feasible_cells']} of "
            f"{summary['ax3_headline']['physical_cells']} factorized staging cells pass "
            "both timing and double-buffer capacity. A passing cell only shows "
            "that warming can finish; no SRAM compute-time or energy benefit is "
            "claimed without an execution model.",
            "",
            "A compact cross-hierarchy comparison conservatively adds separate "
            "upstream and SRAM-staging P99 values and credits no SRAM compute "
            "benefit:",
            "",
            "| Architecture | HBM expert GiB | SRAM MiB | Predictor | P99 TPOT |",
            "|---|---:|---:|---|---:|",
            "## Most important insights",
            "",
            "1. Lookahead buys bandwidth almost linearly, while amplification "
            "spends it almost linearly.",
            "2. P99 is reliability-limited before the mean link is saturated: "
            "a 99% complete-wave predictor still exposes the one-percent tail.",
            "3. Whole-expert SRAM staging is plausible only with hundreds of "
            "MiB for top-8 routing; top-1/top-2 or selective sub-expert staging "
            "changes this capacity bound much more than another small bandwidth gain.",
            "4. Queue replay can be materially worse than mean or wave-local "
            "headroom. Architecture claims should use the phase map for bounds "
            "and the queue points as a tail-risk warning.",
            "",
            "## Interpretation boundary",
            "",
            summary["interpretation_boundary"],
            "",
        ]
    )
    comparison_lines = [
        f"| {row['architecture']} | {float(row['hbm_expert_gib']):.1f} | "
        f"{float(row['sram_mib']):.0f} | {row['predictor_profile']} | "
        f"{float(row['modeled_p99_tpot_ms']):.2f} ms |"
        for row in architecture_comparison
    ]
    insertion = lines.index("## Most important insights")
    lines[insertion:insertion] = comparison_lines + [""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
