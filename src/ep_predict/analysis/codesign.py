from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row(
    rows: list[dict[str, str]],
    *,
    policy: str,
    capacity: int,
    lookahead: int,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["phase"] == "decode"
        and row["domain"] == "__domain_balanced__"
        and row["baseline"] == policy
        and int(row["capacity"]) == capacity
        and int(row["delta"]) == lookahead
    ]
    if len(matches) != 1:
        raise ValueError(
            f"missing prediction row {policy}, K={capacity}, Δ={lookahead}"
        )
    return matches[0]


def analyze_codesign_map(
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    h4_dir = Path(experiment_config["output_dir"])
    h4_rows = _read_csv(h4_dir / "oracle_metrics.csv")
    settings = experiment_config["codesign_map"]
    prediction_path = Path(settings["prediction_summary"])
    prediction_rows = _read_csv(prediction_path)
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    simulation = experiment_config["simulation"]
    measured_scale = float(experiment_config["decision_gate"]["bandwidth_scale"])
    complete_threshold = float(settings["min_complete_route_coverage"])
    physical_threshold = float(settings["physical_headroom_ratio"])
    oracle_ontime_threshold = float(
        experiment_config["decision_gate"]["min_deadline_feasible_cold_fraction"]
    )
    oracle_stall_threshold = float(
        experiment_config["decision_gate"]["min_oracle_stall_reduction"]
    )
    rows: list[dict[str, Any]] = []
    for capacity in simulation["capacities"]:
        for lookahead in simulation["lookaheads"]:
            physical_matches = [
                row
                for row in h4_rows
                if int(row["capacity"]) == int(capacity)
                and int(row["lookahead"]) == int(lookahead)
                and float(row["bandwidth_scale"]) == measured_scale
            ]
            if len(physical_matches) != 1:
                raise ValueError(
                    f"missing H4 row K={capacity}, Δ={lookahead}"
                )
            physical = physical_matches[0]
            eligible_waves = int(physical["eligible_waves"])
            cold_experts = int(physical["cold_demand_experts"])
            mean_cold = cold_experts / eligible_waves
            slack_ms = int(lookahead) * float(
                physical["effective_inter_moe_layer_ms"]
            )
            cold_service_ms = mean_cold * float(physical["expert_transfer_ms"])
            headroom = slack_ms / cold_service_ms
            oracle_pass = (
                float(physical["deadline_feasible_cold_fraction"])
                >= oracle_ontime_threshold
                and float(physical["oracle_stall_reduction"])
                >= oracle_stall_threshold
            )
            for policy in settings["policies"]:
                prediction = _prediction_row(
                    prediction_rows,
                    policy=str(policy),
                    capacity=int(capacity),
                    lookahead=int(lookahead),
                )
                complete = float(prediction["mean_complete_token_coverage"])
                prediction_pass = complete >= complete_threshold
                if oracle_pass and prediction_pass:
                    category = "candidate_codesign_region"
                elif oracle_pass:
                    category = "prediction_limited"
                elif prediction_pass:
                    category = "physics_limited"
                else:
                    category = "physics_and_prediction_limited"
                if headroom >= physical_threshold and prediction_pass:
                    first_order_region = "nominal_joint_headroom"
                elif headroom >= physical_threshold:
                    first_order_region = "nominal_prediction_limited"
                elif prediction_pass:
                    first_order_region = "nominal_physics_limited"
                else:
                    first_order_region = "nominal_both_limited"
                rows.append(
                    {
                        "phase": "decode",
                        "policy": policy,
                        "capacity": int(capacity),
                        "lookahead": int(lookahead),
                        "bandwidth_scale": measured_scale,
                        "mean_cold_experts_per_wave": mean_cold,
                        "available_lead_time_ms": slack_ms,
                        "mean_cold_service_time_ms": cold_service_ms,
                        "cold_service_headroom_ratio": headroom,
                        "cold_service_pressure_ratio": 1.0 / headroom,
                        "selection_coverage": float(
                            prediction["mean_selection_coverage"]
                        ),
                        "complete_route_coverage": complete,
                        "deadline_feasible_cold_fraction": float(
                            physical["deadline_feasible_cold_fraction"]
                        ),
                        "oracle_stall_reduction": float(
                            physical["oracle_stall_reduction"]
                        ),
                        "oracle_pass": oracle_pass,
                        "prediction_pass": prediction_pass,
                        "category": category,
                        "first_order_region": first_order_region,
                    }
                )
    _write_csv(output / "codesign_points.csv", rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    summary = {
        "analysis": "h4_codesign_map",
        "status": "post_hoc_descriptive",
        "points": len(rows),
        "thresholds": {
            "cold_service_headroom_ratio": physical_threshold,
            "complete_route_coverage": complete_threshold,
            "deadline_feasible_cold_fraction": oracle_ontime_threshold,
            "oracle_stall_reduction": oracle_stall_threshold,
        },
        "category_counts": counts,
        "interpretation": (
            "The upper-right region is a candidate for policy replay, not "
            "evidence of profitability. Profit requires learned/oracle recovery "
            "and measured copy/compute overlap."
        ),
    }
    write_json(output / "summary.json", summary)
    return summary

