from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ep_predict.tracing.schema import TRACE_SCHEMA_VERSION
from ep_predict.tracing.storage import iter_trace_records, write_json


Scope = tuple[str, str, int]


def entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total) for count in counts if count > 0
    )


def gini(counts: list[int]) -> float:
    values = sorted(counts)
    total = sum(values)
    size = len(values)
    if size == 0 or total == 0:
        return 0.0
    weighted = sum(
        (2 * index - size - 1) * value
        for index, value in enumerate(values, start=1)
    )
    return weighted / (size * total)


def demand_coverage(counts: list[int], top_n: int) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return sum(sorted(counts, reverse=True)[:top_n]) / total


def distribution_metrics(counts: list[int], top_n: list[int]) -> dict[str, Any]:
    total = sum(counts)
    distribution_entropy = entropy(counts)
    nonzero_median = statistics.median(counts) if counts else 0
    result: dict[str, Any] = {
        "n_selections": total,
        "entropy": distribution_entropy,
        "normalized_entropy": (
            distribution_entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
        ),
        "gini": gini(counts),
        "max_to_median": (
            max(counts) / nonzero_median
            if nonzero_median
            else (math.inf if counts and max(counts) else 0.0)
        ),
        "utilization_variance": (
            statistics.pvariance(counts) if len(counts) > 1 else 0.0
        ),
    }
    for count in top_n:
        result[f"top_{count}_coverage"] = demand_coverage(counts, count)
    return result


def _top_set(counter: Counter[int], size: int) -> set[int]:
    return {expert for expert, _ in counter.most_common(size)}


def _set_coverage(counter: Counter[int], experts: set[int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum(counter[expert] for expert in experts) / total


def window_stability(
    events: list[list[int]],
    *,
    window_size: int,
    hotset_size: int,
) -> dict[str, Any] | None:
    windows = [
        Counter(
            expert
            for selected in events[start : start + window_size]
            for expert in selected
        )
        for start in range(0, len(events) - window_size + 1, window_size)
    ]
    if len(windows) < 2:
        return None

    global_counter = sum(windows, Counter())
    global_hotset = _top_set(global_counter, hotset_size)
    jaccards: list[float] = []
    lagged_coverages: list[float] = []
    oracle_coverages: list[float] = []
    static_coverages: list[float] = []
    for previous, current in zip(windows, windows[1:]):
        previous_hotset = _top_set(previous, hotset_size)
        current_hotset = _top_set(current, hotset_size)
        union = previous_hotset | current_hotset
        jaccards.append(
            len(previous_hotset & current_hotset) / len(union) if union else 1.0
        )
        lagged_coverages.append(_set_coverage(current, previous_hotset))
        oracle_coverages.append(_set_coverage(current, current_hotset))
        static_coverages.append(_set_coverage(current, global_hotset))

    ratios = [
        lagged / oracle if oracle else 0.0
        for lagged, oracle in zip(lagged_coverages, oracle_coverages, strict=True)
    ]
    return {
        "window_size": window_size,
        "hotset_size": hotset_size,
        "n_windows": len(windows),
        "mean_jaccard": statistics.fmean(jaccards),
        "min_jaccard": min(jaccards),
        "mean_static_coverage": statistics.fmean(static_coverages),
        "mean_lagged_coverage": statistics.fmean(lagged_coverages),
        "mean_oracle_coverage": statistics.fmean(oracle_coverages),
        "mean_lagged_oracle_ratio": statistics.fmean(ratios),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _write_report(
    path: Path,
    *,
    run_id: str,
    gate: dict[str, Any],
    popularity_rows: list[dict[str, Any]],
    integrity: dict[str, Any],
) -> None:
    phase = gate["phase"]
    capacity = gate["capacity_experts"]
    headline = sorted(
        (
            row
            for row in popularity_rows
            if row["phase"] == phase and row["domain"] == "__all__"
        ),
        key=lambda row: row["layer_id"],
    )
    lines = [
        f"# H1 result: `{run_id}`",
        "",
        f"**Decision:** {gate['decision']}",
        "",
        (
            f"{gate['passing_layers']} of {gate['eligible_layers']} eligible "
            f"{phase} layers passed both the skew and stability thresholds "
            f"(required fraction: {gate['required_layer_fraction']:.2f})."
        ),
        "",
        "## Integrity",
        "",
        f"- Records: {integrity['record_count']}",
        f"- Requests: {integrity['request_count']}",
        f"- Observed layers: {integrity['layers']}",
        f"- Routing top-k values: {integrity['top_k_values']}",
        f"- Schema versions: {integrity['schema_versions']}",
        "",
        "## Headline per-layer skew",
        "",
        f"| Layer | Gini | Top-{capacity} coverage | Lift over uniform |",
        "|---:|---:|---:|---:|",
    ]
    for row in headline:
        coverage = row[f"top_{capacity}_coverage"]
        lines.append(
            f"| {row['layer_id']} | {row['gini']:.3f} | "
            f"{_format_percent(coverage)} | "
            f"{coverage / (capacity / row['num_experts']):.2f}× |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            gate["interpretation"],
            "",
            "This is a workload-characterization result only. Hooked inference is "
            "not a latency measurement, and H1 does not establish that experts can "
            "be prefetched in time.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_h1(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    directory = Path(run_dir)
    analysis_dir = directory / "analysis" / "h1"
    analysis_config = experiment_config.get("analysis", {})
    top_n = [int(value) for value in analysis_config.get("top_n", [1, 2, 4, 8])]
    window_sizes = [
        int(value) for value in analysis_config.get("window_sizes", [128, 256])
    ]
    gate_config = experiment_config["decision_gate"]
    capacity = int(gate_config["capacity_experts"])
    if capacity not in top_n:
        top_n.append(capacity)
        top_n.sort()

    records = list(iter_trace_records(directory))
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    model_report = json.loads(
        (directory / "model_report.json").read_text(encoding="utf-8")
    )
    experts_by_layer = {
        int(router["layer_id"]): int(router["num_experts"])
        for router in model_report["routers"]
    }

    counts: dict[Scope, Counter[int]] = defaultdict(Counter)
    token_counts: Counter[Scope] = Counter()
    events: dict[Scope, list[list[int]]] = defaultdict(list)
    request_ids: set[int] = set()
    schema_versions: set[int] = set()
    top_k_values: set[int] = set()
    errors: list[str] = []

    for record in records:
        request_ids.add(int(record["request_id"]))
        schema_versions.add(int(record["metadata_version"]))
        phase = record["phase"]
        domain = record["domain"]
        layer = int(record["layer_id"])
        selected = [int(expert) for expert in record["selected_expert_ids"]]
        top_k_values.add(len(selected))
        num_experts = experts_by_layer.get(layer)
        if num_experts is None:
            errors.append(f"record references unknown layer {layer}")
            continue
        if any(expert < 0 or expert >= num_experts for expert in selected):
            errors.append(f"out-of-range expert at layer {layer}")
            continue
        for scoped_domain in (domain, "__all__"):
            scope = (phase, scoped_domain, layer)
            counts[scope].update(selected)
            token_counts[scope] += 1
            events[scope].append(selected)

    if schema_versions != {TRACE_SCHEMA_VERSION}:
        errors.append(
            f"expected schema {TRACE_SCHEMA_VERSION}, observed {sorted(schema_versions)}"
        )
    integrity = {
        "record_count": len(records),
        "request_count": len(request_ids),
        "layers": sorted(experts_by_layer),
        "schema_versions": sorted(schema_versions),
        "top_k_values": sorted(top_k_values),
        "errors": errors,
        "passed": not errors,
    }
    write_json(analysis_dir / "trace_integrity.json", integrity)
    if errors:
        raise ValueError(f"trace integrity failed: {errors[:5]}")

    popularity_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for (phase, domain, layer), counter in sorted(counts.items()):
        num_experts = experts_by_layer[layer]
        dense_counts = [counter[index] for index in range(num_experts)]
        metrics = distribution_metrics(dense_counts, top_n)
        popularity_rows.append(
            {
                "phase": phase,
                "domain": domain,
                "layer_id": layer,
                "num_experts": num_experts,
                "n_tokens": token_counts[(phase, domain, layer)],
                **metrics,
            }
        )
        total = sum(dense_counts)
        for rank, (expert, count) in enumerate(counter.most_common(), start=1):
            rank_rows.append(
                {
                    "phase": phase,
                    "domain": domain,
                    "layer_id": layer,
                    "rank": rank,
                    "expert_id": expert,
                    "selections": count,
                    "probability": count / total if total else 0.0,
                }
            )

    window_rows: list[dict[str, Any]] = []
    for (phase, domain, layer), scoped_events in sorted(events.items()):
        for window_size in window_sizes:
            metrics = window_stability(
                scoped_events,
                window_size=window_size,
                hotset_size=capacity,
            )
            if metrics is not None:
                window_rows.append(
                    {
                        "phase": phase,
                        "domain": domain,
                        "layer_id": layer,
                        **metrics,
                    }
                )

    _write_csv(analysis_dir / "popularity.csv", popularity_rows)
    _write_csv(analysis_dir / "rank_frequency.csv", rank_rows)
    _write_csv(analysis_dir / "window_stability.csv", window_rows)

    gate_phase = str(gate_config.get("phase", "decode"))
    gate_window = int(gate_config.get("window_size", 256))
    min_lift = float(gate_config.get("min_coverage_lift_over_uniform", 2.0))
    min_jaccard = float(gate_config.get("min_hotset_jaccard", 0.5))
    min_ratio = float(gate_config.get("min_lagged_oracle_ratio", 0.8))
    required_fraction = float(gate_config.get("min_passing_layer_fraction", 0.5))
    stability_lookup = {
        (row["phase"], row["domain"], row["layer_id"], row["window_size"]): row
        for row in window_rows
    }
    layer_decisions: list[dict[str, Any]] = []
    for row in popularity_rows:
        if row["phase"] != gate_phase or row["domain"] != "__all__":
            continue
        stability = stability_lookup.get(
            (gate_phase, "__all__", row["layer_id"], gate_window)
        )
        if stability is None:
            continue
        uniform_coverage = capacity / row["num_experts"]
        actual_coverage = row[f"top_{capacity}_coverage"]
        lift = actual_coverage / uniform_coverage
        skew_pass = lift >= min_lift
        stability_pass = (
            stability["mean_jaccard"] >= min_jaccard
            and stability["mean_lagged_oracle_ratio"] >= min_ratio
        )
        layer_decisions.append(
            {
                "layer_id": row["layer_id"],
                "coverage": actual_coverage,
                "coverage_lift_over_uniform": lift,
                "mean_jaccard": stability["mean_jaccard"],
                "mean_lagged_oracle_ratio": stability[
                    "mean_lagged_oracle_ratio"
                ],
                "skew_pass": skew_pass,
                "stability_pass": stability_pass,
                "pass": skew_pass and stability_pass,
            }
        )

    passing = sum(decision["pass"] for decision in layer_decisions)
    eligible = len(layer_decisions)
    passing_fraction = passing / eligible if eligible else 0.0
    supported = eligible > 0 and passing_fraction >= required_fraction
    if not eligible:
        decision = "INCONCLUSIVE"
        interpretation = (
            f"No {gate_phase} layer had two complete {gate_window}-token windows. "
            "Collect a larger trace before deciding H1."
        )
    elif supported:
        decision = "PILOT_SUPPORT"
        interpretation = (
            "Operational hotness is strong and stable enough to justify a "
            "confirmation run and subsequent static/adaptive residency baselines."
        )
    else:
        decision = "PILOT_DOES_NOT_SUPPORT"
        interpretation = (
            "The configured hot tier does not pass the model-wide pilot gate. "
            "Inspect domain-specific rows before deciding whether H1 is locally "
            "mixed or should be rejected for this testbed."
        )
    gate = {
        "hypothesis": "H1",
        "phase": gate_phase,
        "capacity_experts": capacity,
        "window_size": gate_window,
        "passing_layers": passing,
        "eligible_layers": eligible,
        "passing_layer_fraction": passing_fraction,
        "required_layer_fraction": required_fraction,
        "decision": decision,
        "interpretation": interpretation,
        "thresholds": {
            "min_coverage_lift_over_uniform": min_lift,
            "min_hotset_jaccard": min_jaccard,
            "min_lagged_oracle_ratio": min_ratio,
        },
        "layers": layer_decisions,
    }
    write_json(analysis_dir / "gate.json", gate)
    summary = {
        "run_id": manifest["run_id"],
        "integrity": integrity,
        "gate": gate,
        "outputs": {
            "popularity": str(analysis_dir / "popularity.csv"),
            "rank_frequency": str(analysis_dir / "rank_frequency.csv"),
            "window_stability": str(analysis_dir / "window_stability.csv"),
        },
    }
    write_json(analysis_dir / "summary.json", summary)
    _write_report(
        analysis_dir / "REPORT.md",
        run_id=manifest["run_id"],
        gate=gate,
        popularity_rows=popularity_rows,
        integrity=integrity,
    )
    return summary
