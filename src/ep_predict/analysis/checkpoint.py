from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from ep_predict.analysis.h2 import (
    MetricAccumulator,
    TokenRoute,
    _load_token_routes,
    _transition_candidates,
)
from ep_predict.tracing.storage import iter_trace_records, write_json


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


def _config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _js_divergence(left: Counter[int], right: Counter[int], size: int) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return 0.0
    left_p = [left[index] / left_total for index in range(size)]
    right_p = [right[index] / right_total for index in range(size)]
    midpoint = [(a + b) / 2 for a, b in zip(left_p, right_p, strict=True)]

    def kl(values: list[float], reference: list[float]) -> float:
        return sum(
            value * math.log(value / other)
            for value, other in zip(values, reference, strict=True)
            if value and other
        )

    return 0.5 * kl(left_p, midpoint) + 0.5 * kl(right_p, midpoint)


def _ranked(counter: Counter[int], capacity: int, size: int) -> set[int]:
    return set(
        sorted(
            range(size),
            key=lambda expert: (-counter[expert], expert),
        )[:capacity]
    )


def _trace_map(
    run: Path,
    *,
    phase: str,
    request_ids: set[int],
) -> tuple[
    dict[tuple[int, str, int, int], tuple[int, tuple[int, ...], str]],
    int,
]:
    result: dict[tuple[int, str, int, int], tuple[int, tuple[int, ...], str]] = {}
    top_k_values: set[int] = set()
    for record in iter_trace_records(run):
        if str(record["phase"]) != phase:
            continue
        request_id = int(record["request_id"])
        if request_id not in request_ids:
            continue
        key = (
            request_id,
            str(record["sample_id"]),
            int(record["token_position"]),
            int(record["layer_id"]),
        )
        route = tuple(int(value) for value in record["selected_expert_ids"])
        top_k_values.add(len(route))
        value = (int(record["input_token_id"]), route, str(record["domain"]))
        if key in result:
            raise ValueError(f"duplicate trace key in {run}: {key}")
        result[key] = value
    if len(top_k_values) != 1:
        raise ValueError(f"{run} contains mixed top-k values: {top_k_values}")
    return result, top_k_values.pop()


def _matched_route_rows(
    base_run: Path,
    instruct_run: Path,
    *,
    phase: str,
    request_ids: set[int],
    num_experts: int,
    hot_capacity: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base, base_top_k = _trace_map(
        base_run, phase=phase, request_ids=request_ids
    )
    instruct, instruct_top_k = _trace_map(
        instruct_run, phase=phase, request_ids=request_ids
    )
    if base_top_k != instruct_top_k:
        raise ValueError("checkpoint routing top-k values differ")
    if set(base) != set(instruct):
        missing_base = sorted(set(instruct) - set(base))[:5]
        missing_instruct = sorted(set(base) - set(instruct))[:5]
        raise ValueError(
            "checkpoint traces do not have identical token-layer keys; "
            f"missing base={missing_base}, missing instruct={missing_instruct}"
        )

    accumulators: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "intersection": 0,
            "jaccard": 0.0,
            "exact": 0,
            "base_counts": Counter(),
            "instruct_counts": Counter(),
        }
    )
    input_mismatches: list[tuple[int, str, int, int, int, int]] = []
    for key in sorted(base):
        base_token, base_route, base_domain = base[key]
        instruct_token, instruct_route, instruct_domain = instruct[key]
        if base_domain != instruct_domain:
            raise ValueError(f"domain mismatch for {key}")
        if base_token != instruct_token:
            input_mismatches.append(
                (*key, base_token, instruct_token)
            )
            continue
        layer = key[3]
        accumulator = accumulators[(base_domain, layer)]
        left = set(base_route)
        right = set(instruct_route)
        intersection = len(left & right)
        accumulator["n"] += 1
        accumulator["intersection"] += intersection
        accumulator["jaccard"] += intersection / len(left | right)
        accumulator["exact"] += left == right
        accumulator["base_counts"].update(base_route)
        accumulator["instruct_counts"].update(instruct_route)

    if input_mismatches:
        raise ValueError(
            "matched-token invariant failed; first mismatches: "
            f"{input_mismatches[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for (domain, layer), value in sorted(accumulators.items()):
        base_hot = _ranked(value["base_counts"], hot_capacity, num_experts)
        instruct_hot = _ranked(
            value["instruct_counts"], hot_capacity, num_experts
        )
        rows.append(
            {
                "phase": phase,
                "domain": domain,
                "layer_id": layer,
                "n_tokens": value["n"],
                "selection_agreement": (
                    value["intersection"] / (value["n"] * base_top_k)
                ),
                "route_jaccard": value["jaccard"] / value["n"],
                "exact_route_match": value["exact"] / value["n"],
                "popularity_js_divergence_nats": _js_divergence(
                    value["base_counts"],
                    value["instruct_counts"],
                    num_experts,
                ),
                "hotset_jaccard": (
                    len(base_hot & instruct_hot) / len(base_hot | instruct_hot)
                ),
            }
        )

    domains = sorted({row["domain"] for row in rows})
    layers = sorted({int(row["layer_id"]) for row in rows})
    for layer in layers:
        scoped = [
            row
            for row in rows
            if int(row["layer_id"]) == layer and row["domain"] in domains
        ]
        rows.append(
            {
                "phase": phase,
                "domain": "__domain_balanced__",
                "layer_id": layer,
                "n_tokens": sum(int(row["n_tokens"]) for row in scoped),
                "selection_agreement": statistics.fmean(
                    float(row["selection_agreement"]) for row in scoped
                ),
                "route_jaccard": statistics.fmean(
                    float(row["route_jaccard"]) for row in scoped
                ),
                "exact_route_match": statistics.fmean(
                    float(row["exact_route_match"]) for row in scoped
                ),
                "popularity_js_divergence_nats": statistics.fmean(
                    float(row["popularity_js_divergence_nats"]) for row in scoped
                ),
                "hotset_jaccard": statistics.fmean(
                    float(row["hotset_jaccard"]) for row in scoped
                ),
            }
        )

    integrity = {
        "phase": phase,
        "held_out_requests": len(request_ids),
        "matched_token_layer_records": len(base),
        "input_token_id_mismatches": 0,
        "routing_top_k": base_top_k,
        "layers": layers,
        "domains": domains,
    }
    return rows, integrity


def _validate_all_input_tokens(
    base_run: Path,
    instruct_run: Path,
    *,
    phase: str,
    request_ids: set[int],
) -> dict[str, Any]:
    def selected(run: Path):
        return (
            record
            for record in iter_trace_records(run)
            if str(record["phase"]) == phase
            and int(record["request_id"]) in request_ids
        )

    token_keys: set[tuple[int, str, int]] = set()
    record_count = 0
    for base, instruct in zip_longest(
        selected(base_run),
        selected(instruct_run),
    ):
        if base is None or instruct is None:
            raise ValueError("full checkpoint trace lengths differ")
        key_fields = (
            "request_id",
            "sample_id",
            "phase",
            "token_position",
            "layer_id",
            "domain",
        )
        base_key = tuple(base[field] for field in key_fields)
        instruct_key = tuple(instruct[field] for field in key_fields)
        if base_key != instruct_key:
            raise ValueError(
                "full checkpoint trace keys differ: "
                f"{base_key} versus {instruct_key}"
            )
        if int(base["input_token_id"]) != int(instruct["input_token_id"]):
            raise ValueError(f"full matched-token invariant failed at {base_key}")
        token_keys.add(
            (
                int(base["request_id"]),
                str(base["sample_id"]),
                int(base["token_position"]),
            )
        )
        record_count += 1
    return {
        "all_requests_checked": len(request_ids),
        "all_matched_tokens": len(token_keys),
        "all_matched_token_layer_records": record_count,
        "all_input_token_id_mismatches": 0,
    }


def _predictability_rows(
    run: Path,
    checkpoint: str,
    *,
    phase: str,
    source_layer: int,
    capacities: list[int],
    lookaheads: list[int],
) -> list[dict[str, Any]]:
    metrics = _read_csv(run / "analysis" / "h2" / "metrics.csv")
    lookup = {
        (
            row["domain"],
            int(row["source_layer"]),
            int(row["delta"]),
            int(row["capacity"]),
            row["baseline"],
        ): row
        for row in metrics
        if row["phase"] == phase
    }
    domains = sorted({row["domain"] for row in metrics if row["phase"] == phase})
    rows: list[dict[str, Any]] = []
    for domain in domains:
        for delta in lookaheads:
            for capacity in capacities:
                static = lookup[(domain, source_layer, delta, capacity, "static")]
                transition = lookup[
                    (domain, source_layer, delta, capacity, "transition")
                ]
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "phase": phase,
                        "domain": domain,
                        "source_layer": source_layer,
                        "target_layer": source_layer + delta,
                        "delta": delta,
                        "capacity": capacity,
                        "transition_selection_coverage": float(
                            transition["selection_coverage"]
                        ),
                        "static_selection_coverage": float(
                            static["selection_coverage"]
                        ),
                        "selection_gain_over_static": float(
                            transition["selection_coverage"]
                        )
                        - float(static["selection_coverage"]),
                        "transition_complete_coverage": float(
                            transition["complete_token_coverage"]
                        ),
                        "static_complete_coverage": float(
                            static["complete_token_coverage"]
                        ),
                        "complete_gain_over_static": float(
                            transition["complete_token_coverage"]
                        )
                        - float(static["complete_token_coverage"]),
                    }
                )
    for checkpoint_delta_capacity in sorted(
        {
            (row["checkpoint"], row["delta"], row["capacity"])
            for row in rows
        }
    ):
        name, delta, capacity = checkpoint_delta_capacity
        scoped = [
            row
            for row in rows
            if row["checkpoint"] == name
            and row["delta"] == delta
            and row["capacity"] == capacity
            and row["domain"] in domains
        ]
        rows.append(
            {
                "checkpoint": name,
                "phase": phase,
                "domain": "__domain_balanced__",
                "source_layer": source_layer,
                "target_layer": source_layer + delta,
                "delta": delta,
                "capacity": capacity,
                **{
                    key: statistics.fmean(float(row[key]) for row in scoped)
                    for key in (
                        "transition_selection_coverage",
                        "static_selection_coverage",
                        "selection_gain_over_static",
                        "transition_complete_coverage",
                        "static_complete_coverage",
                        "complete_gain_over_static",
                    )
                },
            }
        )
    return rows


def _fit_transition_tables(
    tokens: list[TokenRoute],
    train_ids: set[int],
    *,
    phase: str,
    layers: list[int],
    lookaheads: list[int],
) -> tuple[
    dict[tuple[int, int], dict[int, Counter[int]]],
    dict[int, Counter[int]],
]:
    transitions: dict[
        tuple[int, int], dict[int, Counter[int]]
    ] = defaultdict(lambda: defaultdict(Counter))
    marginals: dict[int, Counter[int]] = defaultdict(Counter)
    for token in tokens:
        if token.request_id not in train_ids or token.phase != phase:
            continue
        for layer, route in token.routes.items():
            marginals[layer].update(route)
        for source_layer in layers:
            source = token.routes[source_layer]
            for delta in lookaheads:
                target = token.routes.get(source_layer + delta)
                if target is None:
                    continue
                table = transitions[(source_layer, delta)]
                for source_expert in source:
                    table[source_expert].update(target)
    return transitions, marginals


def _cross_transfer_rows(
    runs: dict[str, Path],
    *,
    phase: str,
    train_ids: set[int],
    test_ids: set[int],
    layers: list[int],
    lookaheads: list[int],
    capacities: list[int],
    num_experts: int,
    routing_top_k: int,
) -> list[dict[str, Any]]:
    token_sets = {name: _load_token_routes(run)[0] for name, run in runs.items()}
    fitted = {
        name: _fit_transition_tables(
            tokens,
            train_ids,
            phase=phase,
            layers=layers,
            lookaheads=lookaheads,
        )
        for name, tokens in token_sets.items()
    }
    rows: list[dict[str, Any]] = []
    for train_name, (transitions, marginals) in fitted.items():
        for eval_name, eval_tokens in token_sets.items():
            accumulators: dict[
                tuple[str, int, int, int], MetricAccumulator
            ] = {}
            for token in sorted(
                (
                    token
                    for token in eval_tokens
                    if token.request_id in test_ids and token.phase == phase
                ),
                key=lambda token: (
                    token.domain,
                    token.request_id,
                    token.token_position,
                ),
            ):
                for source_layer in layers:
                    source = token.routes[source_layer]
                    for delta in lookaheads:
                        target = token.routes.get(source_layer + delta)
                        if target is None:
                            continue
                        ranking = _transition_candidates(
                            source,
                            rows=transitions[(source_layer, delta)],
                            marginal=marginals[source_layer + delta],
                            capacity=max(capacities),
                            num_experts=num_experts,
                        )
                        for capacity in capacities:
                            key = (
                                token.domain,
                                source_layer,
                                delta,
                                capacity,
                            )
                            accumulator = accumulators.setdefault(
                                key,
                                MetricAccumulator(
                                    capacity=capacity,
                                    top_k=routing_top_k,
                                ),
                            )
                            accumulator.add(ranking[:capacity], target)
            for key, accumulator in sorted(accumulators.items()):
                rows.append(
                    {
                        "train_checkpoint": train_name,
                        "eval_checkpoint": eval_name,
                        "phase": phase,
                        "domain": key[0],
                        "source_layer": key[1],
                        "target_layer": key[1] + key[2],
                        "delta": key[2],
                        "capacity": key[3],
                        **accumulator.metrics(),
                    }
                )
    return rows


def _validate_self_transfer(
    rows: list[dict[str, Any]],
    runs: dict[str, Path],
) -> float:
    largest = 0.0
    for checkpoint, run in runs.items():
        h2 = _read_csv(run / "analysis" / "h2" / "metrics.csv")
        expected = {
            (
                row["domain"],
                int(row["source_layer"]),
                int(row["delta"]),
                int(row["capacity"]),
            ): row
            for row in h2
            if row["phase"] == "prefill" and row["baseline"] == "transition"
        }
        actual = {
            (
                row["domain"],
                int(row["source_layer"]),
                int(row["delta"]),
                int(row["capacity"]),
            ): row
            for row in rows
            if row["train_checkpoint"] == checkpoint
            and row["eval_checkpoint"] == checkpoint
        }
        if set(expected) != set(actual):
            raise ValueError(f"cross-transfer self scopes differ for {checkpoint}")
        for key in expected:
            largest = max(
                largest,
                abs(
                    float(expected[key]["selection_coverage"])
                    - float(actual[key]["selection_coverage"])
                ),
                abs(
                    float(expected[key]["complete_token_coverage"])
                    - float(actual[key]["complete_token_coverage"])
                ),
            )
    if largest > 1e-12:
        raise ValueError(f"self-transfer validation differs by {largest}")
    return largest


def _gate(
    rows: list[dict[str, Any]],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    source_layer = int(gate_config["source_layer"])
    delta = int(gate_config["lookahead"])
    capacity = int(gate_config["capacity_experts"])
    threshold = float(gate_config["min_abs_selection_gain_difference"])
    required_domains = int(gate_config["min_consistent_domains"])
    scoped = [
        row
        for row in rows
        if row["source_layer"] == source_layer
        and row["delta"] == delta
        and row["capacity"] == capacity
        and row["domain"] != "__domain_balanced__"
    ]
    by_checkpoint_domain = {
        (row["checkpoint"], row["domain"]): row for row in scoped
    }
    domains = sorted({row["domain"] for row in scoped})
    differences = {
        domain: float(
            by_checkpoint_domain[("instruct", domain)][
                "selection_gain_over_static"
            ]
        )
        - float(
            by_checkpoint_domain[("base", domain)]["selection_gain_over_static"]
        )
        for domain in domains
    }
    mean_difference = statistics.fmean(differences.values())
    direction = 1 if mean_difference > 0 else -1 if mean_difference < 0 else 0
    consistent = sum(
        1
        for value in differences.values()
        if direction and (1 if value > 0 else -1 if value < 0 else 0) == direction
    )
    passed = abs(mean_difference) >= threshold and consistent >= required_domains
    return {
        "hypothesis": "C0",
        "question": (
            "Does post-training materially change long-horizon conditional "
            "routing predictability under identical input tokens?"
        ),
        "phase": "prefill",
        "source_layer": source_layer,
        "target_layer": source_layer + delta,
        "lookahead": delta,
        "capacity_experts": capacity,
        "metric": "transition selection-coverage gain over static popularity",
        "instruct_minus_base_gain": mean_difference,
        "domain_differences": differences,
        "consistent_domains": consistent,
        "required_consistent_domains": required_domains,
        "min_abs_gain_difference": threshold,
        "decision": (
            "PILOT_SUPPORTS_POSTTRAINING_PREDICTABILITY_EFFECT"
            if passed
            else "PILOT_DOES_NOT_SUPPORT_POSTTRAINING_PREDICTABILITY_EFFECT"
        ),
        "pass": passed,
    }


def _write_report(
    path: Path,
    *,
    gate: dict[str, Any],
    route_rows: list[dict[str, Any]],
    predictability_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
    capacity: int,
) -> None:
    route_scoped = [
        row for row in route_rows if row["domain"] == "__domain_balanced__"
    ]
    mean_agreement = statistics.fmean(
        float(row["selection_agreement"]) for row in route_scoped
    )
    mean_exact = statistics.fmean(
        float(row["exact_route_match"]) for row in route_scoped
    )
    primary = {
        row["checkpoint"]: row
        for row in predictability_rows
        if row["domain"] == "__domain_balanced__"
        and row["source_layer"] == gate["source_layer"]
        and row["delta"] == gate["lookahead"]
        and row["capacity"] == capacity
    }
    penalties: list[float] = []
    for eval_name, other_name in (("base", "instruct"), ("instruct", "base")):
        self_values = [
            float(row["selection_coverage"])
            for row in cross_rows
            if row["eval_checkpoint"] == eval_name
            and row["train_checkpoint"] == eval_name
            and row["source_layer"] == gate["source_layer"]
            and row["delta"] == gate["lookahead"]
            and row["capacity"] == capacity
        ]
        cross_values = [
            float(row["selection_coverage"])
            for row in cross_rows
            if row["eval_checkpoint"] == eval_name
            and row["train_checkpoint"] == other_name
            and row["source_layer"] == gate["source_layer"]
            and row["delta"] == gate["lookahead"]
            and row["capacity"] == capacity
        ]
        penalties.append(
            statistics.fmean(self_values) - statistics.fmean(cross_values)
        )
    lines = [
        "# C0 Base–Instruct trajectory comparison",
        "",
        "## Formal decision",
        "",
        f"**{gate['decision']}**.",
        "",
        "At the preregistered prefill layer-0→15, K=16 gate, the change in "
        "transition-over-static selection gain is "
        f"{100 * gate['instruct_minus_base_gain']:+.1f} percentage points. "
        f"{gate['consistent_domains']}/4 domains have the aggregate direction.",
        "",
        "## Matched-token trajectory change",
        "",
        "Across held-out domains and layers, Base and Instruct select "
        f"{100 * mean_agreement:.1f}% of the same experts on average; the full "
        f"top-8 sets are identical for {100 * mean_exact:.1f}% of token-layer "
        "events.",
        "",
        "## Long-range prediction",
        "",
        "| Checkpoint | Transition selection | Gain over static | Complete top-8 |",
        "|---|---:|---:|---:|",
    ]
    for checkpoint in ("base", "instruct"):
        row = primary[checkpoint]
        lines.append(
            f"| {checkpoint.capitalize()} | "
            f"{100 * float(row['transition_selection_coverage']):.1f}% | "
            f"{100 * float(row['selection_gain_over_static']):+.1f} pp | "
            f"{100 * float(row['transition_complete_coverage']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "A transition table learned on the other checkpoint loses "
            f"{100 * statistics.fmean(penalties):.1f} selection-coverage points "
            "on average at the same primary layer pair. This transfer result "
            "tests policy portability, not language-model quality.",
            "",
            "## Scope",
            "",
            "The primary comparison uses identical externally supplied tokens, "
            "the same 96/32 held-out request split, and no hidden-state "
            "predictor. Free-running decode, SFT/DPO localization, hardware "
            "replay, and model-quality claims are outside this pilot.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_checkpoint_trajectories(
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    runs = {
        name: Path(value["run"])
        for name, value in experiment_config["checkpoints"].items()
    }
    if set(runs) != {"base", "instruct"}:
        raise ValueError("C0 requires exactly base and instruct checkpoints")
    output_dir = Path(experiment_config["output_dir"]) / "analysis" / "c0"
    phase = str(experiment_config.get("phase", "prefill"))
    if phase != "prefill":
        raise ValueError("the matched C0 pilot is preregistered for prefill")
    capacities = [int(value) for value in experiment_config["capacities"]]
    lookaheads = [int(value) for value in experiment_config["lookaheads"]]
    source_layer = int(experiment_config["primary_source_layer"])
    hot_capacity = int(experiment_config.get("hotset_capacity", 16))

    model_reports = {
        name: json.loads((run / "model_report.json").read_text(encoding="utf-8"))
        for name, run in runs.items()
    }
    geometry = {
        name: {
            (
                int(router["layer_id"]),
                int(router["num_experts"]),
                int(router["top_k"]),
            )
            for router in report["routers"]
        }
        for name, report in model_reports.items()
    }
    if geometry["base"] != geometry["instruct"]:
        raise ValueError("Base and Instruct routing geometries differ")
    layers = sorted(layer for layer, _experts, _top_k in geometry["base"])
    expert_counts = {experts for _layer, experts, _top_k in geometry["base"]}
    top_k_values = {top_k for _layer, _experts, top_k in geometry["base"]}
    if len(expert_counts) != 1 or len(top_k_values) != 1:
        raise ValueError("C0 requires uniform expert count and top-k across layers")
    num_experts = expert_counts.pop()
    routing_top_k = top_k_values.pop()

    splits = {
        name: json.loads(
            (run / "analysis" / "h2" / "split.json").read_text(encoding="utf-8")
        )
        for name, run in runs.items()
    }
    for name in ("instruct",):
        if splits[name]["requests"] != splits["base"]["requests"]:
            raise ValueError(f"held-out split differs for {name}")
    train_ids = set(int(value) for value in splits["base"]["train_request_ids"])
    test_ids = set(int(value) for value in splits["base"]["test_request_ids"])

    all_input_integrity = _validate_all_input_tokens(
        runs["base"],
        runs["instruct"],
        phase=phase,
        request_ids=train_ids | test_ids,
    )
    route_rows, integrity = _matched_route_rows(
        runs["base"],
        runs["instruct"],
        phase=phase,
        request_ids=test_ids,
        num_experts=num_experts,
        hot_capacity=hot_capacity,
    )
    predictability_rows: list[dict[str, Any]] = []
    for checkpoint, run in runs.items():
        predictability_rows.extend(
            _predictability_rows(
                run,
                checkpoint,
                phase=phase,
                source_layer=source_layer,
                capacities=capacities,
                lookaheads=lookaheads,
            )
        )
    cross_rows = _cross_transfer_rows(
        runs,
        phase=phase,
        train_ids=train_ids,
        test_ids=test_ids,
        layers=layers,
        lookaheads=lookaheads,
        capacities=capacities,
        num_experts=num_experts,
        routing_top_k=routing_top_k,
    )
    self_transfer_max_abs_difference = _validate_self_transfer(cross_rows, runs)
    gate = _gate(predictability_rows, experiment_config["decision_gate"])

    _write_csv(output_dir / "matched_route_overlap.csv", route_rows)
    _write_csv(output_dir / "predictability_by_horizon.csv", predictability_rows)
    _write_csv(output_dir / "cross_checkpoint_transfer.csv", cross_rows)
    write_json(output_dir / "gate.json", gate)
    _write_report(
        output_dir / "REPORT.md",
        gate=gate,
        route_rows=route_rows,
        predictability_rows=predictability_rows,
        cross_rows=cross_rows,
        capacity=int(experiment_config["decision_gate"]["capacity_experts"]),
    )
    summary = {
        "analysis_id": str(experiment_config["analysis_id"]),
        "hypothesis": "C0",
        "config_fingerprint": _config_sha256(experiment_config),
        "checkpoints": {
            name: {
                "run": str(run),
                "model_commit": model_reports[name].get("model_commit"),
            }
            for name, run in runs.items()
        },
        "integrity": {
            **integrity,
            **all_input_integrity,
            "train_requests": len(train_ids),
            "test_requests": len(test_ids),
            "self_transfer_max_abs_difference": self_transfer_max_abs_difference,
        },
        "gate": gate,
        "outputs": {
            "matched_route_overlap": str(
                output_dir / "matched_route_overlap.csv"
            ),
            "predictability_by_horizon": str(
                output_dir / "predictability_by_horizon.csv"
            ),
            "cross_checkpoint_transfer": str(
                output_dir / "cross_checkpoint_transfer.csv"
            ),
            "gate": str(output_dir / "gate.json"),
            "report": str(output_dir / "REPORT.md"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary
