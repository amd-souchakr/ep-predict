from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ep_predict.analysis.h2 import (
    TokenRoute,
    _load_token_routes,
    _transition_candidates,
)
from ep_predict.analysis.h3 import _load_feature_map
from ep_predict.analysis.h4 import DemandWave, _cold_sets, _decode_waves
from ep_predict.tracing.storage import write_json


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _physical_context(
    *,
    waves: list[DemandWave],
    token_count: int,
    layers: int,
    capacities: list[int],
    lookaheads: list[int],
    bandwidth_scales: list[float],
    layer_ms: float,
    transfer_ms: float,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, float], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int, float], dict[str, Any]] = {}
    for capacity in capacities:
        cold, _compulsory, _demanded, _misses, _first = _cold_sets(
            waves, capacity
        )
        for delta in lookaheads:
            eligible_waves = token_count * (layers - delta)
            cold_experts = sum(
                len(cold[(token_index, layer)])
                for token_index in range(token_count)
                for layer in range(delta, layers)
            )
            mean_cold = cold_experts / eligible_waves
            for bandwidth_scale in bandwidth_scales:
                headroom = (
                    bandwidth_scale * delta * layer_ms
                    / (mean_cold * transfer_ms)
                )
                row = {
                    "capacity": capacity,
                    "lookahead": delta,
                    "bandwidth_scale": bandwidth_scale,
                    "eligible_waves": eligible_waves,
                    "cold_demand_experts": cold_experts,
                    "mean_cold_experts_per_wave": mean_cold,
                    "available_lead_time_ms": delta * layer_ms,
                    "measured_expert_transfer_ms": transfer_ms,
                    "cold_service_headroom": headroom,
                    "cold_service_pressure": 1.0 / headroom,
                    "first_order_oracle_stall_reduction": min(1.0, headroom),
                }
                rows.append(row)
                lookup[(capacity, delta, bandwidth_scale)] = row
    return rows, lookup


def _category(
    *,
    coverage: float,
    amplification: float,
    headroom: float,
    required_benefit: float,
    max_amplification: float,
    passed: bool,
) -> str:
    if passed:
        return "analytically_profitable"
    traffic_limited = amplification > max_amplification
    physics_limited = headroom / amplification < required_benefit
    prediction_limited = coverage < required_benefit
    limits = [
        name
        for name, active in (
            ("traffic", traffic_limited),
            ("physics", physics_limited),
            ("prediction", prediction_limited),
        )
        if active
    ]
    return "_and_".join(limits) + "_limited" if limits else "threshold_limited"


def _controlled_sweep(
    *,
    physical_rows: list[dict[str, Any]],
    coverage_step: float,
    amplifications: list[float],
    gate: dict[str, Any],
    expert_bytes: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    min_stall = float(gate["min_modeled_stall_reduction"])
    min_recovery = float(gate["min_oracle_recovery"])
    max_amplification = float(gate["max_predicted_to_useful_bytes"])
    coverage_values = [
        min(1.0, index * coverage_step)
        for index in range(round(1.0 / coverage_step) + 1)
    ]
    design_rows: list[dict[str, Any]] = []
    inverse_rows: list[dict[str, Any]] = []
    for physical in physical_rows:
        headroom = float(physical["cold_service_headroom"])
        oracle = float(physical["first_order_oracle_stall_reduction"])
        required_benefit = max(min_stall, min_recovery * oracle)
        physics_max_amplification = headroom / required_benefit
        allowed_max_amplification = min(
            max_amplification, physics_max_amplification
        )
        for amplification in amplifications:
            inverse_feasible = (
                amplification <= max_amplification
                and headroom / amplification + 1e-12 >= required_benefit
                and required_benefit <= 1.0
            )
            inverse_rows.append(
                {
                    **physical,
                    "candidate_transfer_amplification": amplification,
                    "required_benefit": required_benefit,
                    "minimum_complete_cold_set_coverage": (
                        required_benefit if inverse_feasible else ""
                    ),
                    "maximum_profitable_amplification": (
                        allowed_max_amplification
                        if allowed_max_amplification >= 1.0
                        else ""
                    ),
                    "inverse_window_exists": inverse_feasible,
                    "empty_window_reason": (
                        ""
                        if inverse_feasible
                        else (
                            "speculative_traffic_gate"
                            if amplification > max_amplification
                            else "insufficient_cold_service_headroom"
                        )
                    ),
                }
            )
            for coverage in coverage_values:
                useful_experts = coverage * physical["cold_demand_experts"]
                candidate_experts = amplification * useful_experts
                service_fraction = (
                    min(
                        1.0,
                        headroom / (amplification * coverage),
                    )
                    if coverage > 0
                    else 0.0
                )
                modeled_stall = coverage * service_fraction
                oracle_recovery = (
                    min(1.0, modeled_stall / oracle) if oracle > 0 else 1.0
                )
                passed = (
                    modeled_stall + 1e-12 >= min_stall
                    and oracle_recovery + 1e-12 >= min_recovery
                    and amplification <= max_amplification
                )
                design_rows.append(
                    {
                        **physical,
                        "complete_cold_set_coverage": coverage,
                        "candidate_transfer_amplification": amplification,
                        "predicted_useful_experts": useful_experts,
                        "predicted_candidate_experts": candidate_experts,
                        "predicted_false_experts": (
                            candidate_experts - useful_experts
                        ),
                        "predicted_useful_bytes": useful_experts * expert_bytes,
                        "predicted_candidate_bytes": (
                            candidate_experts * expert_bytes
                        ),
                        "predicted_false_bytes": (
                            (candidate_experts - useful_experts) * expert_bytes
                        ),
                        "proportional_deadline_service_fraction": service_fraction,
                        "modeled_stall_reduction": modeled_stall,
                        "oracle_recovery": oracle_recovery,
                        "required_benefit": required_benefit,
                        "profitable": passed,
                        "category": _category(
                            coverage=coverage,
                            amplification=amplification,
                            headroom=headroom,
                            required_benefit=required_benefit,
                            max_amplification=max_amplification,
                            passed=passed,
                        ),
                    }
                )

    grouped: dict[tuple[int, float, float, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in design_rows:
        grouped[
            (
                int(row["capacity"]),
                float(row["bandwidth_scale"]),
                float(row["candidate_transfer_amplification"]),
                float(row["complete_cold_set_coverage"]),
            )
        ].append(row)
    window_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        capacity, bandwidth, amplification, coverage = key
        passing = sorted(
            int(row["lookahead"]) for row in rows if row["profitable"]
        )
        window_rows.append(
            {
                "capacity": capacity,
                "bandwidth_scale": bandwidth,
                "candidate_transfer_amplification": amplification,
                "complete_cold_set_coverage": coverage,
                "profitable_lookahead_count": len(passing),
                "minimum_profitable_lookahead": (
                    min(passing) if passing else ""
                ),
                "maximum_profitable_lookahead": (
                    max(passing) if passing else ""
                ),
                "profitable_lookaheads": ",".join(map(str, passing)),
                "window_exists": bool(passing),
            }
        )
    return design_rows, window_rows, inverse_rows


def _load_linear_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _linear_candidates(
    arrays: dict[str, np.ndarray],
    *,
    source_layer: int,
    delta: int,
    features: np.ndarray,
    capacity: int,
) -> tuple[int, ...]:
    prefix = f"decode_l{source_layer:02d}_d{delta}"
    weight = arrays[f"{prefix}_weight"]
    bias = arrays[f"{prefix}_bias"]
    mean = arrays[f"{prefix}_mean"]
    scale = arrays[f"{prefix}_scale"]
    scores = ((features.astype(np.float32) - mean) / scale) @ weight.T + bias
    ranking = np.argsort(-scores, kind="stable")[:capacity]
    return tuple(int(value) for value in ranking)


def _test_residency(
    waves: list[DemandWave],
    *,
    capacity: int,
    test_ids: set[int],
) -> dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]]:
    caches: dict[int, OrderedDict[int, None]] = {}
    result: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]] = {}
    for wave in waves:
        cache = caches.setdefault(wave.layer, OrderedDict())
        resident = frozenset(cache)
        cold = frozenset(expert for expert in wave.experts if expert not in cache)
        if wave.request_id in test_ids:
            result[(wave.token_index, wave.layer)] = (resident, cold)
        protected = set(wave.experts)
        for expert in wave.experts:
            if expert in cache:
                cache.move_to_end(expert)
                continue
            while len(cache) >= capacity:
                victim = next(
                    (
                        candidate
                        for candidate in cache
                        if candidate not in protected
                    ),
                    None,
                )
                if victim is None:
                    break
                del cache[victim]
            if len(cache) < capacity:
                cache[expert] = None
    return result


def _transition_tables(
    train_tokens: list[TokenRoute],
    lookaheads: set[int],
) -> tuple[
    dict[tuple[str, int], Counter[int]],
    dict[tuple[str, int, int], dict[int, Counter[int]]],
]:
    marginals: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    transitions: dict[
        tuple[str, int, int], dict[int, Counter[int]]
    ] = defaultdict(lambda: defaultdict(Counter))
    for token in train_tokens:
        for target_layer, target in token.routes.items():
            marginals[(token.phase, target_layer)].update(target)
        for source_layer, source in token.routes.items():
            for delta in lookaheads:
                target = token.routes.get(source_layer + delta)
                if target is None:
                    continue
                table = transitions[(token.phase, source_layer, delta)]
                for source_expert in source:
                    table[source_expert].update(target)
    return marginals, transitions


def _policy_replay(
    *,
    config: dict[str, Any],
    waves: list[DemandWave],
    physical: dict[tuple[int, int, float], dict[str, Any]],
    layers: int,
    expert_bytes: int,
) -> list[dict[str, Any]]:
    h3_run = Path(config["h3_run"])
    h3_analysis = Path(config["h3_analysis"])
    tokens, _requests = _load_token_routes(h3_run)
    split = json.loads((h3_analysis / "split.json").read_text(encoding="utf-8"))
    train_ids = {int(value) for value in split["train_request_ids"]}
    test_ids = {int(value) for value in split["test_request_ids"]}
    train_tokens = [token for token in tokens if token.request_id in train_ids]
    test_tokens = [
        token
        for token in tokens
        if token.request_id in test_ids and token.phase == "decode"
    ]
    h1_tokens, _ = _load_token_routes(Path(config["h1_run"]))
    h1_decode = [token for token in h1_tokens if token.phase == "decode"]
    h3_decode = [token for token in tokens if token.phase == "decode"]
    if len(h1_decode) != len(h3_decode):
        raise ValueError("H1/H3 decode token count differs")
    token_index: dict[tuple[int, int], int] = {}
    for index, (h1, h3) in enumerate(zip(h1_decode, h3_decode, strict=True)):
        identity_h1 = (h1.request_id, h1.token_position)
        identity_h3 = (h3.request_id, h3.token_position)
        if identity_h1 != identity_h3 or h1.routes != h3.routes:
            raise ValueError(f"H1/H3 route mismatch at decode token {index}")
        token_index[identity_h3] = index

    requested_deltas = {int(cell["lookahead"]) for cell in config["policy_cells"]}
    marginals, transitions = _transition_tables(train_tokens, requested_deltas)
    features, _integrity = _load_feature_map(h3_run)
    linear = _load_linear_arrays(h3_analysis / "linear_predictors.npz")
    num_experts = 64
    gate = config["decision_gate"]
    max_gate_amplification = float(gate["max_predicted_to_useful_bytes"])
    rows: list[dict[str, Any]] = []
    residency_by_capacity: dict[
        int, dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]]
    ] = {}

    for cell in config["policy_cells"]:
        capacity = int(cell["capacity"])
        delta = int(cell["lookahead"])
        bandwidth = float(cell["bandwidth_scale"])
        if capacity not in residency_by_capacity:
            residency_by_capacity[capacity] = _test_residency(
                waves, capacity=capacity, test_ids=test_ids
            )
        residency = residency_by_capacity[capacity]
        context = physical[(capacity, delta, bandwidth)]
        headroom = float(context["cold_service_headroom"])
        oracle = float(context["first_order_oracle_stall_reduction"])
        for policy in ("transition", "linear"):
            cold_experts = 0
            cold_waves = 0
            useful = 0
            false = 0
            complete = 0
            candidate_churn: list[float] = []
            previous: dict[tuple[int, str], frozenset[int]] = {}
            for token in test_tokens:
                index = token_index[(token.request_id, token.token_position)]
                for source_layer in range(layers - delta):
                    target_layer = source_layer + delta
                    resident, cold = residency[(index, target_layer)]
                    if not cold:
                        continue
                    if policy == "transition":
                        candidates = _transition_candidates(
                            token.routes[source_layer],
                            rows=transitions[("decode", source_layer, delta)],
                            marginal=marginals[("decode", target_layer)],
                            capacity=capacity,
                            num_experts=num_experts,
                        )
                    else:
                        candidates = _linear_candidates(
                            linear,
                            source_layer=source_layer,
                            delta=delta,
                            features=features[
                                (
                                    token.request_id,
                                    token.phase,
                                    token.token_position,
                                    source_layer,
                                )
                            ],
                            capacity=capacity,
                        )
                    candidate_set = frozenset(candidates)
                    transfers = candidate_set - resident
                    wave_useful = len(transfers & cold)
                    wave_false = len(transfers - cold)
                    cold_experts += len(cold)
                    cold_waves += 1
                    useful += wave_useful
                    false += wave_false
                    complete += cold <= candidate_set
                    churn_key = (source_layer, token.domain)
                    prior = previous.get(churn_key)
                    if prior is not None:
                        candidate_churn.append(
                            len(prior - candidate_set) / capacity
                        )
                    previous[churn_key] = candidate_set

            candidate_transfers = useful + false
            cold_coverage = useful / cold_experts
            complete_coverage = complete / cold_waves
            amplification = (
                candidate_transfers / useful if useful else math.inf
            )
            allowed_false = (max_gate_amplification - 1.0) * useful
            required_false_rejection = (
                max(0.0, false - allowed_false) / false if false else 0.0
            )
            transfer_load_per_cold = candidate_transfers / cold_experts
            service_fraction = (
                min(1.0, headroom / transfer_load_per_cold)
                if transfer_load_per_cold
                else 0.0
            )
            modeled_stall = complete_coverage * service_fraction
            recovery = min(1.0, modeled_stall / oracle) if oracle else 1.0
            passed = (
                modeled_stall
                >= float(gate["min_modeled_stall_reduction"])
                and recovery >= float(gate["min_oracle_recovery"])
                and amplification <= max_gate_amplification
            )
            rows.append(
                {
                    "cell": str(cell["name"]),
                    "policy": policy,
                    "capacity": capacity,
                    "lookahead": delta,
                    "bandwidth_scale": bandwidth,
                    "test_requests": len(test_ids),
                    "eligible_cold_waves": cold_waves,
                    "cold_demand_experts": cold_experts,
                    "useful_candidate_experts": useful,
                    "false_candidate_experts": false,
                    "candidate_transfer_experts": candidate_transfers,
                    "cold_expert_coverage": cold_coverage,
                    "complete_cold_set_coverage": complete_coverage,
                    "candidate_transfer_amplification": amplification,
                    "traffic_gate_excess_factor": (
                        amplification / max_gate_amplification
                    ),
                    "required_false_candidate_rejection_for_traffic_gate": (
                        required_false_rejection
                    ),
                    "candidate_transfer_load_per_cold_expert": (
                        transfer_load_per_cold
                    ),
                    "cold_service_headroom": headroom,
                    "proportional_deadline_service_fraction": service_fraction,
                    "first_order_oracle_stall_reduction": oracle,
                    "modeled_stall_reduction": modeled_stall,
                    "oracle_recovery": recovery,
                    "useful_candidate_bytes": useful * expert_bytes,
                    "false_candidate_bytes": false * expert_bytes,
                    "candidate_transfer_bytes": candidate_transfers * expert_bytes,
                    "deadline_feasible_useful_bytes": (
                        useful * expert_bytes * service_fraction
                    ),
                    "late_useful_bytes": (
                        useful * expert_bytes * (1.0 - service_fraction)
                    ),
                    "late_candidate_bytes": (
                        candidate_transfers
                        * expert_bytes
                        * (1.0 - service_fraction)
                    ),
                    "mean_candidate_replacement_fraction": (
                        statistics.fmean(candidate_churn)
                        if candidate_churn
                        else 0.0
                    ),
                    "profitable": passed,
                    "category": _category(
                        coverage=complete_coverage,
                        amplification=amplification,
                        headroom=headroom,
                        required_benefit=max(
                            float(gate["min_modeled_stall_reduction"]),
                            float(gate["min_oracle_recovery"]) * oracle,
                        ),
                        max_amplification=max_gate_amplification,
                        passed=passed,
                    ),
                }
            )
    return rows


def analyze_h5(experiment_config: dict[str, Any]) -> dict[str, Any]:
    output = Path(experiment_config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    h1_run = Path(experiment_config["h1_run"])
    model_report = json.loads(
        (h1_run / "model_report.json").read_text(encoding="utf-8")
    )
    expert_sizes = {
        int(router["expert_bytes_each"]) for router in model_report["routers"]
    }
    if len(expert_sizes) != 1:
        raise ValueError("H5 requires one exact expert size")
    expert_bytes = expert_sizes.pop()
    layers = int(model_report["router_count"])
    measurement = json.loads(
        Path(experiment_config["h4_measurement"]).read_text(encoding="utf-8")
    )
    layer_ms = float(measurement["decode"]["effective_inter_moe_layer_ms"])
    transfer_ms = float(measurement["transfer"]["exact_expert_median_ms"])
    waves, token_count = _decode_waves(h1_run)
    if len(waves) != token_count * layers:
        raise ValueError("incomplete H1 decode trace")

    sweep = experiment_config["sweep"]
    capacities = [int(value) for value in sweep["capacities"]]
    lookaheads = [int(value) for value in sweep["lookaheads"]]
    bandwidth_scales = [float(value) for value in sweep["bandwidth_scales"]]
    amplifications = [
        float(value) for value in sweep["candidate_amplifications"]
    ]
    physical_rows, physical_lookup = _physical_context(
        waves=waves,
        token_count=token_count,
        layers=layers,
        capacities=capacities,
        lookaheads=lookaheads,
        bandwidth_scales=bandwidth_scales,
        layer_ms=layer_ms,
        transfer_ms=transfer_ms,
    )
    design, windows, inverse = _controlled_sweep(
        physical_rows=physical_rows,
        coverage_step=float(sweep["complete_coverage_step"]),
        amplifications=amplifications,
        gate=experiment_config["decision_gate"],
        expert_bytes=expert_bytes,
    )
    _write_csv(output / "h5_physical_context.csv", physical_rows)
    _write_csv(output / "h5_design_points.csv", design)
    _write_csv(output / "h5_windows.csv", windows)
    _write_csv(output / "h5_inverse_requirements.csv", inverse)

    policy_rows = _policy_replay(
        config=experiment_config,
        waves=waves,
        physical=physical_lookup,
        layers=layers,
        expert_bytes=expert_bytes,
    )
    _write_csv(output / "h5_policy_placement.csv", policy_rows)

    profitable_design = sum(bool(row["profitable"]) for row in design)
    profitable_policies = [row for row in policy_rows if row["profitable"]]
    category_counts: dict[str, int] = defaultdict(int)
    for row in design:
        category_counts[str(row["category"])] += 1
    gate = {
        "hypothesis": "H5",
        "evidence_grade": "trace_driven_analytical_pilot",
        "thresholds": experiment_config["decision_gate"],
        "controlled_design_region_exists": profitable_design > 0,
        "profitable_controlled_cells": profitable_design,
        "controlled_cells": len(design),
        "existing_policy_passes": bool(profitable_policies),
        "passing_existing_policies": [
            {
                "cell": row["cell"],
                "policy": row["policy"],
            }
            for row in profitable_policies
        ],
        "decision": (
            "PILOT_SUPPORTS_EXISTING_POLICY"
            if profitable_policies
            else "PILOT_SUPPORTS_DESIGN_REGION_BUT_NOT_EXISTING_POLICY"
        ),
    }
    write_json(output / "gate.json", gate)
    summary = {
        "hypothesis": "H5",
        "decision": gate["decision"],
        "evidence_grade": gate["evidence_grade"],
        "trace": {
            "decode_tokens": token_count,
            "decode_waves": len(waves),
            "expert_bytes": expert_bytes,
            "effective_inter_moe_layer_ms": layer_ms,
            "measured_expert_transfer_ms": transfer_ms,
        },
        "grid": {
            "physical_cells": len(physical_rows),
            "controlled_design_cells": len(design),
            "inverse_cells": len(inverse),
            "window_rows": len(windows),
            "policy_rows": len(policy_rows),
        },
        "controlled_category_counts": dict(sorted(category_counts.items())),
        "gate": gate,
        "outputs": {
            "physical_context": str(output / "h5_physical_context.csv"),
            "design_points": str(output / "h5_design_points.csv"),
            "windows": str(output / "h5_windows.csv"),
            "inverse_requirements": str(
                output / "h5_inverse_requirements.csv"
            ),
            "policy_placement": str(output / "h5_policy_placement.csv"),
            "gate": str(output / "gate.json"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
