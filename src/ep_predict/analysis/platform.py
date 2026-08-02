from __future__ import annotations

import csv
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must have equal length >= 2")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if not left_scale or not right_scale:
        return 1.0 if left == right else 0.0
    return numerator / (left_scale * right_scale)


def _ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for position in ordered[index:end]:
            result[position] = average_rank
        index = end
    return result


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _geometry(report: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    return [
        (
            int(router["layer_id"]),
            int(router["num_experts"]),
            int(router["top_k"]),
            int(router["expert_bytes_each"]),
        )
        for router in report["routers"]
    ]


def _normalized_collection_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("run_id", None)
    result.pop("output_dir", None)
    result.pop("parity", None)
    return result


def _request_keys(manifest: dict[str, Any]) -> list[tuple[int, str]]:
    return [
        (int(request["request_id"]), str(request["sample_id"]))
        for request in manifest["requests"]
    ]


def _input_hash_integrity(
    manifest: dict[str, Any], reference_manifest: dict[str, Any]
) -> dict[str, int]:
    left = {
        (int(request["request_id"]), str(request["sample_id"])): str(value)
        for request in manifest["requests"]
        if (value := request.get("input_token_ids_sha256"))
    }
    right = {
        (int(request["request_id"]), str(request["sample_id"])): str(value)
        for request in reference_manifest["requests"]
        if (value := request.get("input_token_ids_sha256"))
    }
    common = sorted(set(left) & set(right))
    return {
        "comparable_requests": len(common),
        "mismatches": sum(left[key] != right[key] for key in common),
        "mi355x_hashes": len(left),
        "nvidia_hashes": len(right),
    }


def _h1_rows(run: Path) -> tuple[dict[int, dict[str, str]], dict[int, set[int]]]:
    analysis = run / "analysis" / "h1"
    popularity = {
        int(row["layer_id"]): row
        for row in _read_csv(analysis / "popularity.csv")
        if row["phase"] == "prefill" and row["domain"] == "__all__"
    }
    hotsets: dict[int, set[int]] = {}
    for row in _read_csv(analysis / "rank_frequency.csv"):
        if (
            row["phase"] == "prefill"
            and row["domain"] == "__all__"
            and int(row["rank"]) <= 8
        ):
            hotsets.setdefault(int(row["layer_id"]), set()).add(
                int(row["expert_id"])
            )
    return popularity, hotsets


def _h2_rows(run: Path) -> dict[int, dict[str, Any]]:
    summary = json.loads(
        (run / "analysis" / "h2" / "summary.json").read_text(encoding="utf-8")
    )
    return {
        int(row["delta"]): row
        for row in summary["gate"]["lookaheads"]
    }


def _series_summary(left: list[float], right: list[float]) -> dict[str, float]:
    differences = [a - b for a, b in zip(left, right, strict=True)]
    return {
        "mi355x_mean": statistics.fmean(left),
        "nvidia_mean": statistics.fmean(right),
        "mean_difference": statistics.fmean(differences),
        "mean_absolute_difference": statistics.fmean(map(abs, differences)),
        "max_absolute_difference": max(map(abs, differences)),
        "pearson": _pearson(left, right),
        "spearman": _spearman(left, right),
    }


def analyze_platform_comparison(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    run = Path(run_dir)
    parity = experiment_config["parity"]
    reference = Path(parity["reference_run"])
    output = run / "analysis" / "platform_comparison"
    output.mkdir(parents=True, exist_ok=True)

    model = json.loads((run / "model_report.json").read_text(encoding="utf-8"))
    reference_model = json.loads(
        (reference / "model_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    reference_manifest = json.loads(
        (reference / "run_manifest.json").read_text(encoding="utf-8")
    )
    trace_integrity = json.loads(
        (run / "analysis" / "h1" / "trace_integrity.json").read_text(
            encoding="utf-8"
        )
    )
    requests = manifest["requests"]
    reference_requests = reference_manifest["requests"]
    input_hashes = _input_hash_integrity(manifest, reference_manifest)
    mi_split = json.loads(
        (run / "analysis" / "h2" / "split.json").read_text(encoding="utf-8")
    )
    reference_split = json.loads(
        (reference / "analysis" / "h2" / "split.json").read_text(
            encoding="utf-8"
        )
    )
    router_mismatches = sum(
        int(request.get("router_validation_mismatches", 0)) for request in requests
    )
    bad_router_calls = sum(
        any(int(count) != int(model["router_count"]) for count in request.get("router_calls_per_forward", []))
        for request in requests
    )

    mi_h1, mi_hotsets = _h1_rows(run)
    nv_h1, nv_hotsets = _h1_rows(reference)
    if set(mi_h1) != set(nv_h1) or set(mi_hotsets) != set(nv_hotsets):
        raise ValueError("H1 layer sets differ between platform artifacts")
    layer_rows: list[dict[str, Any]] = []
    for layer in sorted(mi_h1):
        mi_hot = mi_hotsets[layer]
        nv_hot = nv_hotsets[layer]
        layer_rows.append(
            {
                "layer_id": layer,
                "mi355x_top8_coverage": float(mi_h1[layer]["top_8_coverage"]),
                "nvidia_top8_coverage": float(nv_h1[layer]["top_8_coverage"]),
                "top8_coverage_difference": (
                    float(mi_h1[layer]["top_8_coverage"])
                    - float(nv_h1[layer]["top_8_coverage"])
                ),
                "mi355x_gini": float(mi_h1[layer]["gini"]),
                "nvidia_gini": float(nv_h1[layer]["gini"]),
                "gini_difference": (
                    float(mi_h1[layer]["gini"])
                    - float(nv_h1[layer]["gini"])
                ),
                "top8_hotset_intersection": len(mi_hot & nv_hot),
                "top8_hotset_jaccard": len(mi_hot & nv_hot) / len(mi_hot | nv_hot),
            }
        )
    _write_csv(output / "h1_layer_trends.csv", layer_rows)

    mi_h2 = _h2_rows(run)
    nv_h2 = _h2_rows(reference)
    if set(mi_h2) != set(nv_h2):
        raise ValueError("H2 horizon sets differ between platform artifacts")
    horizon_rows: list[dict[str, Any]] = []
    for delta in sorted(mi_h2):
        mi_row = mi_h2[delta]
        nv_row = nv_h2[delta]
        horizon_rows.append(
            {
                "delta": delta,
                "mi355x_selection_gain": float(mi_row["mean_selection_coverage_gain"]),
                "nvidia_selection_gain": float(nv_row["mean_selection_coverage_gain"]),
                "selection_gain_difference": (
                    float(mi_row["mean_selection_coverage_gain"])
                    - float(nv_row["mean_selection_coverage_gain"])
                ),
                "mi355x_complete_gain": float(mi_row["mean_complete_token_coverage_gain"]),
                "nvidia_complete_gain": float(nv_row["mean_complete_token_coverage_gain"]),
                "complete_gain_difference": (
                    float(mi_row["mean_complete_token_coverage_gain"])
                    - float(nv_row["mean_complete_token_coverage_gain"])
                ),
                "mi355x_gate_pass": bool(mi_row["pass"]),
                "nvidia_gate_pass": bool(nv_row["pass"]),
            }
        )
    _write_csv(output / "h2_horizon_trends.csv", horizon_rows)

    h1_top8 = _series_summary(
        [row["mi355x_top8_coverage"] for row in layer_rows],
        [row["nvidia_top8_coverage"] for row in layer_rows],
    )
    h1_gini = _series_summary(
        [row["mi355x_gini"] for row in layer_rows],
        [row["nvidia_gini"] for row in layer_rows],
    )
    h2_selection = _series_summary(
        [row["mi355x_selection_gain"] for row in horizon_rows],
        [row["nvidia_selection_gain"] for row in horizon_rows],
    )
    h2_complete = _series_summary(
        [row["mi355x_complete_gain"] for row in horizon_rows],
        [row["nvidia_complete_gain"] for row in horizon_rows],
    )
    matched_workload = (
        len(requests) == len(reference_requests)
        and _request_keys(manifest) == _request_keys(reference_manifest)
        and manifest["prompt_file_sha256"] == reference_manifest["prompt_file_sha256"]
        and _normalized_collection_config(manifest["experiment_config"])
        == _normalized_collection_config(reference_manifest["experiment_config"])
        and mi_split["requests"] == reference_split["requests"]
    )
    evidence_grade = (
        "descriptive_matched_workload_platform_comparison"
        if matched_workload
        else "descriptive_nested_workload_platform_comparison"
    )
    limitations = [
        "NVIDIA request-level traces are unavailable, so selected-ID parity cannot be tested.",
        "The run contains prefill only; decode parity and timing are not evaluated.",
    ]
    if not matched_workload:
        limitations.extend(
            [
                "The platform workloads or analysis splits are not exactly matched; sampling and platform effects are confounded.",
                "The MI355X H2 estimate is descriptive rather than a matched confirmation.",
            ]
        )

    summary = {
        "analysis_id": "mi355x-olmoe-derived-platform-comparison",
        "state": "complete",
        "evidence_grade": evidence_grade,
        "interchangeability": "NOT_ESTABLISHED_NO_NVIDIA_RAW_TRACE",
        "scope": {
            "mi355x_run": str(run),
            "mi355x_requests": len(requests),
            "nvidia_reference_run": str(reference),
            "nvidia_reference_requests": len(reference_requests),
            "phase": "prefill",
            "routing_top_k": 8,
            "matched_workload_and_split": matched_workload,
        },
        "integrity": {
            "geometry_match": _geometry(model) == _geometry(reference_model),
            "mi355x_trace_integrity_passed": bool(trace_integrity["passed"]),
            "mi355x_router_validation_mismatches": router_mismatches,
            "mi355x_requests_with_bad_router_call_counts": bad_router_calls,
            "mi355x_input_hashes_recorded": sum(
                bool(request.get("input_token_ids_sha256")) for request in requests
            ),
            "nvidia_input_hashes_recorded": input_hashes["nvidia_hashes"],
            "comparable_input_hashes": input_hashes["comparable_requests"],
            "comparable_input_hash_mismatches": input_hashes["mismatches"],
            "request_keys_match": _request_keys(manifest)
            == _request_keys(reference_manifest),
            "prompt_file_sha256_match": manifest["prompt_file_sha256"]
            == reference_manifest["prompt_file_sha256"],
            "collection_settings_match": _normalized_collection_config(
                manifest["experiment_config"]
            )
            == _normalized_collection_config(reference_manifest["experiment_config"]),
            "h2_request_split_match": mi_split["requests"]
            == reference_split["requests"],
            "mi355x_environment": manifest["environment"],
        },
        "h1": {
            "top8_coverage": h1_top8,
            "gini": h1_gini,
            "mean_top8_hotset_jaccard": statistics.fmean(
                row["top8_hotset_jaccard"] for row in layer_rows
            ),
            "mean_top8_hotset_intersection": statistics.fmean(
                row["top8_hotset_intersection"] for row in layer_rows
            ),
        },
        "h2": {
            "selection_gain_over_static": h2_selection,
            "complete_gain_over_static": h2_complete,
            "mi355x_passing_horizons": [
                row["delta"] for row in horizon_rows if row["mi355x_gate_pass"]
            ],
            "nvidia_passing_horizons": [
                row["delta"] for row in horizon_rows if row["nvidia_gate_pass"]
            ],
        },
        "limitations": limitations,
    }
    write_json(output / "summary.json", summary)

    report = f"""# MI355X OLMoE derived-platform comparison

**Evidence grade:** {evidence_grade.replace('_', ' ')}  
**Raw-trace interchangeability:** `NOT_ESTABLISHED_NO_NVIDIA_RAW_TRACE`

## Integrity

- MI355X requests: {len(requests)}.
- MI355X routed records: {trace_integrity['record_count']:,}.
- Model geometry matches the preserved NVIDIA report: {summary['integrity']['geometry_match']}.
- Router validation mismatches: {router_mismatches}.
- Requests with bad router-call counts: {bad_router_calls}.
- Input-token hashes recorded: {summary['integrity']['mi355x_input_hashes_recorded']}/{len(requests)}.
- Matched request keys, prompt hash, collection settings, and H2 split: {matched_workload}.
- Comparable NVIDIA/MI355X input hashes: {input_hashes['comparable_requests']}; mismatches: {input_hashes['mismatches']}.

## H1 derived trends

- Mean top-8 coverage: {100 * h1_top8['mi355x_mean']:.2f}% MI355X versus {100 * h1_top8['nvidia_mean']:.2f}% NVIDIA.
- Layerwise top-8 coverage Pearson/Spearman: {h1_top8['pearson']:.6f}/{h1_top8['spearman']:.6f}.
- Mean absolute top-8 coverage difference: {100 * h1_top8['mean_absolute_difference']:.3f} points.
- Mean top-8 hot-set overlap: {summary['h1']['mean_top8_hotset_intersection']:.3f}/8 experts; Jaccard {summary['h1']['mean_top8_hotset_jaccard']:.4f}.
- Layerwise Gini Pearson/Spearman: {h1_gini['pearson']:.6f}/{h1_gini['spearman']:.6f}.

## H2 derived trends

- Horizon-wise transition selection-gain Pearson/Spearman: {h2_selection['pearson']:.6f}/{h2_selection['spearman']:.6f}.
- Mean absolute selection-gain difference: {100 * h2_selection['mean_absolute_difference']:.3f} points.
- Horizon-wise complete-route-gain Pearson/Spearman: {h2_complete['pearson']:.6f}/{h2_complete['spearman']:.6f}.
- MI355X descriptive passing horizons: {summary['h2']['mi355x_passing_horizons']}.
- NVIDIA descriptive passing horizons: {summary['h2']['nvidia_passing_horizons']}.

## Interpretation boundary

These artifacts compare matched requests and analysis splits when the scope
integrity above is true. They still cannot establish per-record route
agreement or authorize raw-trace interchangeability because the NVIDIA raw
records are unavailable. Small remaining aggregate differences are consistent
with platform-dependent numerical routing changes, but cannot localize them.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return summary
