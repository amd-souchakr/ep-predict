from __future__ import annotations

import csv
import json
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ep_predict.analysis.h2 import _load_token_routes
from ep_predict.tracing.storage import write_json


@dataclass(frozen=True)
class DemandWave:
    token_index: int
    request_id: int
    domain: str
    layer: int
    experts: tuple[int, ...]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decode_waves(run: Path) -> tuple[list[DemandWave], int]:
    tokens, _ = _load_token_routes(run)
    decode = [token for token in tokens if token.phase == "decode"]
    waves: list[DemandWave] = []
    for token_index, token in enumerate(decode):
        for layer, selected in sorted(token.routes.items()):
            experts = tuple(dict.fromkeys(int(expert) for expert in selected))
            waves.append(
                DemandWave(
                    token_index=token_index,
                    request_id=token.request_id,
                    domain=token.domain,
                    layer=layer,
                    experts=experts,
                )
            )
    if not waves:
        raise ValueError("trace has no decode waves")
    return waves, len(decode)


def _cold_sets(
    waves: list[DemandWave],
    capacity: int,
) -> tuple[
    dict[tuple[int, int], tuple[int, ...]],
    dict[tuple[int, int], tuple[int, ...]],
    int,
    int,
    int,
]:
    caches: dict[int, OrderedDict[int, None]] = {}
    seen: dict[int, set[int]] = {}
    cold: dict[tuple[int, int], tuple[int, ...]] = {}
    compulsory: dict[tuple[int, int], tuple[int, ...]] = {}
    demanded = 0
    cold_misses = 0
    compulsory_misses = 0
    for wave in waves:
        cache = caches.setdefault(wave.layer, OrderedDict())
        layer_seen = seen.setdefault(wave.layer, set())
        demanded += len(wave.experts)
        missing = tuple(expert for expert in wave.experts if expert not in cache)
        first_use = tuple(expert for expert in missing if expert not in layer_seen)
        cold[(wave.token_index, wave.layer)] = missing
        compulsory[(wave.token_index, wave.layer)] = first_use
        cold_misses += len(missing)
        compulsory_misses += len(first_use)
        layer_seen.update(wave.experts)
        protected = set(wave.experts)
        for expert in wave.experts:
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
        if len(wave.experts) > capacity:
            raise ValueError("capacity is smaller than a complete demand wave")
    return (
        cold,
        compulsory,
        demanded,
        cold_misses,
        compulsory_misses,
    )


def _scaled_transfer_ms(measurement: dict[str, Any], scale: float) -> float:
    transfer = measurement["transfer"]
    exact = float(transfer["exact_expert_median_ms"])
    startup = min(float(transfer["fit"]["startup_ms"]), exact)
    return startup + (exact - startup) / scale


def _simulate(
    *,
    waves: list[DemandWave],
    token_count: int,
    cold: dict[tuple[int, int], tuple[int, ...]],
    compulsory: dict[tuple[int, int], tuple[int, ...]],
    all_cold_misses: int,
    all_compulsory_misses: int,
    demanded_experts: int,
    layers: int,
    delta: int,
    layer_ms: float,
    transfer_ms: float,
    expert_bytes: int,
) -> dict[str, Any]:
    by_key = {(wave.token_index, wave.layer): wave for wave in waves}
    completion: dict[tuple[int, int, int], float] = {}
    transfer_available_ms = 0.0
    now_ms = 0.0
    on_time = 0
    late = 0
    eligible_demanded = 0
    eligible_cold = 0
    eligible_compulsory = 0
    oracle_stall_ms = 0.0
    reactive_stall_ms = 0.0
    stalled_waves = 0
    eligible_waves = 0

    for token_index in range(token_count):
        for layer in range(layers):
            if layer >= delta:
                eligible_waves += 1
                wave = by_key[(token_index, layer)]
                target_cold = cold[(token_index, layer)]
                target_compulsory = compulsory[(token_index, layer)]
                eligible_demanded += len(wave.experts)
                eligible_cold += len(target_cold)
                eligible_compulsory += len(target_compulsory)
                deadline = now_ms
                completions = [
                    completion[(token_index, layer, expert)]
                    for expert in target_cold
                ]
                on_time += sum(value <= deadline for value in completions)
                late += sum(value > deadline for value in completions)
                stall = max(0.0, max(completions, default=deadline) - deadline)
                if stall > 0:
                    stalled_waves += 1
                    oracle_stall_ms += stall
                    now_ms += stall
                reactive_stall_ms += len(target_cold) * transfer_ms

            target = layer + delta
            if target < layers:
                for expert in cold[(token_index, target)]:
                    start = max(now_ms, transfer_available_ms)
                    finish = start + transfer_ms
                    completion[(token_index, target, expert)] = finish
                    transfer_available_ms = finish
            now_ms += layer_ms

    if on_time + late != eligible_cold:
        raise RuntimeError("oracle transfer accounting does not close")
    resident_hits = eligible_demanded - eligible_cold
    capacity_misses = eligible_cold - eligible_compulsory
    stall_reduction = (
        1.0 - oracle_stall_ms / reactive_stall_ms
        if reactive_stall_ms
        else 1.0
    )
    return {
        "eligible_waves": eligible_waves,
        "stalled_waves": stalled_waves,
        "total_demanded_experts": eligible_demanded,
        "resident_hit_experts": resident_hits,
        "cold_demand_experts": eligible_cold,
        "compulsory_miss_experts": eligible_compulsory,
        "capacity_miss_experts": capacity_misses,
        "deadline_feasible_cold_experts": on_time,
        "late_cold_experts": late,
        "total_demanded_bytes": eligible_demanded * expert_bytes,
        "resident_hit_bytes": resident_hits * expert_bytes,
        "cold_demand_bytes": eligible_cold * expert_bytes,
        "compulsory_miss_bytes": eligible_compulsory * expert_bytes,
        "capacity_miss_bytes": capacity_misses * expert_bytes,
        "deadline_feasible_demanded_bytes": on_time * expert_bytes,
        "late_bytes": late * expert_bytes,
        "resident_hit_fraction": resident_hits / eligible_demanded,
        "deadline_feasible_cold_fraction": (
            on_time / eligible_cold if eligible_cold else 1.0
        ),
        "late_cold_fraction": late / eligible_cold if eligible_cold else 0.0,
        "reactive_stall_ms": reactive_stall_ms,
        "oracle_stall_ms": oracle_stall_ms,
        "oracle_stall_reduction": stall_reduction,
        "oracle_wave_stall_fraction": stalled_waves / eligible_waves,
        "all_trace_cold_demand_experts": all_cold_misses,
        "all_trace_compulsory_miss_experts": all_compulsory_misses,
        "all_trace_capacity_miss_experts": (
            all_cold_misses - all_compulsory_misses
        ),
        "all_trace_demanded_experts": demanded_experts,
    }


def analyze_h4(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    run = Path(run_dir)
    output = Path(experiment_config["output_dir"])
    measurement_path = output / "measurement.json"
    if not measurement_path.is_file():
        raise FileNotFoundError(f"measure H4 before replay: {measurement_path}")
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    model_report = json.loads((run / "model_report.json").read_text(encoding="utf-8"))
    expert_sizes = {
        int(router["expert_bytes_each"]) for router in model_report["routers"]
    }
    if len(expert_sizes) != 1:
        raise ValueError("H4 requires one exact expert size")
    expert_bytes = expert_sizes.pop()
    if expert_bytes != int(measurement["transfer"]["exact_expert_bytes"]):
        raise ValueError("measured transfer size does not match inspected expert")
    layers = int(model_report["router_count"])
    waves, token_count = _decode_waves(run)
    expected = token_count * layers
    if len(waves) != expected:
        raise ValueError(f"incomplete decode trace: {len(waves)} != {expected}")

    simulation = experiment_config["simulation"]
    layer_ms = float(measurement["decode"]["effective_inter_moe_layer_ms"])
    rows: list[dict[str, Any]] = []
    for capacity in simulation["capacities"]:
        cold, compulsory, demanded, cold_misses, compulsory_misses = (
            _cold_sets(waves, int(capacity))
        )
        for scale in simulation["bandwidth_scales"]:
            transfer_ms = _scaled_transfer_ms(measurement, float(scale))
            for delta in simulation["lookaheads"]:
                metrics = _simulate(
                    waves=waves,
                    token_count=token_count,
                    cold=cold,
                    compulsory=compulsory,
                    all_cold_misses=cold_misses,
                    all_compulsory_misses=compulsory_misses,
                    demanded_experts=demanded,
                    layers=layers,
                    delta=int(delta),
                    layer_ms=layer_ms,
                    transfer_ms=transfer_ms,
                    expert_bytes=expert_bytes,
                )
                rows.append(
                    {
                        "phase": simulation["phase"],
                        "capacity": int(capacity),
                        "lookahead": int(delta),
                        "bandwidth_scale": float(scale),
                        "effective_inter_moe_layer_ms": layer_ms,
                        "expert_transfer_ms": transfer_ms,
                        **metrics,
                    }
                )
    _write_csv(output / "oracle_metrics.csv", rows)

    gate_config = experiment_config["decision_gate"]
    candidates = [
        row
        for row in rows
        if row["phase"] == gate_config["phase"]
        and row["capacity"] == int(gate_config["capacity_experts"])
        and row["lookahead"]
        in {int(value) for value in gate_config["eligible_lookaheads"]}
        and row["bandwidth_scale"] == float(gate_config["bandwidth_scale"])
    ]
    passing = [
        row
        for row in candidates
        if row["deadline_feasible_cold_fraction"]
        >= float(gate_config["min_deadline_feasible_cold_fraction"])
        and row["oracle_stall_reduction"]
        >= float(gate_config["min_oracle_stall_reduction"])
    ]
    best = max(
        candidates,
        key=lambda row: (
            row["oracle_stall_reduction"],
            row["deadline_feasible_cold_fraction"],
            -row["lookahead"],
        ),
    )
    decision = "PILOT_SUPPORTS" if passing else "PILOT_DOES_NOT_SUPPORT"
    gate = {
        "hypothesis": "H4",
        "decision": decision,
        "evidence_grade": "calibrated_single_gpu_pilot",
        "primary_scope": {
            "phase": gate_config["phase"],
            "capacity_experts": int(gate_config["capacity_experts"]),
            "lookaheads": list(gate_config["eligible_lookaheads"]),
            "bandwidth_scale": float(gate_config["bandwidth_scale"]),
        },
        "thresholds": {
            "deadline_feasible_cold_fraction": float(
                gate_config["min_deadline_feasible_cold_fraction"]
            ),
            "oracle_stall_reduction": float(
                gate_config["min_oracle_stall_reduction"]
            ),
        },
        "passing_lookaheads": [row["lookahead"] for row in passing],
        "best_primary_row": best,
    }
    write_json(output / "gate.json", gate)
    summary = {
        "hypothesis": "H4",
        "decision": decision,
        "trace_run": str(run),
        "decode_tokens": token_count,
        "decode_waves": len(waves),
        "expert_bytes": expert_bytes,
        "measurement": {
            "median_decode_forward_ms": measurement["decode"]["median_forward_ms"],
            "effective_inter_moe_layer_ms": layer_ms,
            "exact_expert_transfer_ms": measurement["transfer"][
                "exact_expert_median_ms"
            ],
            "effective_bandwidth_gbps": measurement["transfer"]["fit"][
                "effective_bandwidth_gbps"
            ],
        },
        "gate": gate,
    }
    write_json(output / "summary.json", summary)
    report_lines = [
        "# H4 oracle feasibility result",
        "",
        f"**Decision:** `{decision}`",
        "",
        "## Calibration",
        "",
        f"- Hook-free cached-token forward median: "
        f"{summary['measurement']['median_decode_forward_ms']:.3f} ms.",
        f"- Effective inter-MoE-layer budget: {layer_ms:.3f} ms.",
        f"- Exact 12 MiB pinned-host transfer median: "
        f"{summary['measurement']['exact_expert_transfer_ms']:.3f} ms.",
        f"- Fitted effective bandwidth: "
        f"{summary['measurement']['effective_bandwidth_gbps']:.2f} GB/s.",
        "",
        "## Frozen primary gate",
        "",
        "| Δ | Deadline-feasible cold bytes | Oracle stall reduction |",
        "|---:|---:|---:|",
    ]
    for row in sorted(candidates, key=lambda item: item["lookahead"]):
        report_lines.append(
            f"| {row['lookahead']} | "
            f"{100 * row['deadline_feasible_cold_fraction']:.1f}% | "
            f"{100 * row['oracle_stall_reduction']:.1f}% |"
        )
    descriptive_keys = (
        (8, 3, 1.0),
        (16, 3, 1.0),
        (32, 3, 1.0),
        (16, 9, 1.0),
        (16, 1, 2.0),
    )
    descriptive = [
        next(
            row
            for row in rows
            if (
                row["capacity"],
                row["lookahead"],
                row["bandwidth_scale"],
            )
            == key
        )
        for key in descriptive_keys
    ]
    report_lines.extend(
        [
            "",
            "The gate requires both metrics to reach 50% for at least one "
            "short horizon at measured bandwidth and K=16.",
            "",
            "## Descriptive feasibility boundary",
            "",
            "| K | Δ | Bandwidth | Resident hits | On-time cold bytes | "
            "Stall reduction |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in descriptive:
        report_lines.append(
            f"| {row['capacity']} | {row['lookahead']} | "
            f"{row['bandwidth_scale']:g}× | "
            f"{100 * row['resident_hit_fraction']:.1f}% | "
            f"{100 * row['deadline_feasible_cold_fraction']:.1f}% | "
            f"{100 * row['oracle_stall_reduction']:.1f}% |"
        )
    report_lines.extend(
        [
            "",
            "The frozen compact-tier target fails, but the broader scan is "
            "not a universal physical impossibility: K=32 at Δ=3, K=16 at "
            "Δ=9, and K=16 with 2× measured bandwidth expose feasible oracle "
            "regions. These cells are descriptive and do not change the "
            "formal gate or trigger predictor replay.",
            "",
            "## Interpretation boundary",
            "",
            "This is a trace-driven, single-copy-engine oracle calculation. "
            "It establishes a calibrated feasibility region, not end-to-end "
            "speedup or overlap correctness in the live model.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary
