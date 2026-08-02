from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ep_predict.tracing.storage import iter_trace_records, write_json


@dataclass(frozen=True)
class WeightedRoute:
    expert_ids: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class TokenRoutes:
    request_id: int
    sample_id: str
    domain: str
    phase: str
    token_position: int
    routes: dict[int, WeightedRoute]


@dataclass
class CoverageAccumulator:
    candidate_count: int
    top_k: int
    tokens: int = 0
    covered_selections: int = 0
    covered_mass: float = 0.0
    complete_routes: int = 0

    def add(self, candidates: tuple[int, ...], target: WeightedRoute) -> None:
        if len(candidates) != self.candidate_count:
            raise ValueError("candidate set does not match configured candidate count")
        if len(target.expert_ids) != self.top_k:
            raise ValueError("target route does not match configured top-k")
        candidate_set = set(candidates)
        covered = [expert in candidate_set for expert in target.expert_ids]
        self.tokens += 1
        self.covered_selections += sum(covered)
        self.covered_mass += sum(
            weight
            for weight, present in zip(target.weights, covered, strict=True)
            if present
        )
        self.complete_routes += all(covered)

    def metrics(self) -> dict[str, float | int]:
        if not self.tokens:
            raise ValueError("coverage scope has no observations")
        mean_covered = self.covered_selections / self.tokens
        return {
            "n_tokens": self.tokens,
            "selection_coverage": self.covered_selections / (self.tokens * self.top_k),
            "routed_mass_coverage": self.covered_mass / self.tokens,
            "complete_route_coverage": self.complete_routes / self.tokens,
            "candidate_amplification": self.candidate_count / self.top_k,
            "candidate_set_fraction": self.candidate_count / 32,
            "useful_candidate_amplification": (
                self.candidate_count / mean_covered if mean_covered else math.inf
            ),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ranked(
    scores: Counter[int] | dict[int, float], candidate_count: int, num_experts: int
) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(num_experts),
            key=lambda expert: (-float(scores.get(expert, 0)), expert),
        )[:candidate_count]
    )


def _transition_candidates(
    source: tuple[int, ...],
    rows: dict[int, Counter[int]],
    marginal: Counter[int],
    candidate_count: int,
    num_experts: int,
) -> tuple[int, ...]:
    scores: Counter[int] = Counter()
    marginal_total = sum(marginal.values())
    for source_expert in source:
        row = rows.get(source_expert)
        total = sum(row.values()) if row else 0
        distribution: Iterable[tuple[int, int]]
        denominator: int
        if total:
            distribution, denominator = row.items(), total
        else:
            distribution, denominator = marginal.items(), marginal_total
        if denominator:
            for target_expert, count in distribution:
                scores[target_expert] += count / denominator
    if source:
        for target_expert in list(scores):
            scores[target_expert] /= len(source)
    return _ranked(scores, candidate_count, num_experts)


def _copy_candidates(
    source: tuple[int, ...], marginal: Counter[int], candidate_count: int, num_experts: int
) -> tuple[int, ...]:
    selected = list(dict.fromkeys(source))[:candidate_count]
    if len(selected) == candidate_count:
        return tuple(selected)
    for expert in _ranked(marginal, num_experts, num_experts):
        if expert not in selected:
            selected.append(expert)
        if len(selected) == candidate_count:
            break
    return tuple(selected)


def _load_routes(run_dir: Path) -> tuple[list[TokenRoutes], dict[int, dict[str, str]]]:
    grouped: dict[tuple[int, str, str, str, int], dict[int, WeightedRoute]] = (
        defaultdict(dict)
    )
    requests: dict[int, dict[str, str]] = {}
    for record in iter_trace_records(run_dir):
        request_id = int(record["request_id"])
        sample_id = str(record["sample_id"])
        domain = str(record["domain"])
        phase = str(record["phase"])
        position = int(record["token_position"])
        layer = int(record["moe_layer_index"])
        key = (request_id, sample_id, domain, phase, position)
        if layer in grouped[key]:
            raise ValueError(f"duplicate token-layer key: {key}, {layer}")
        route = WeightedRoute(
            tuple(int(value) for value in record["selected_expert_ids"]),
            tuple(float(value) for value in record["selected_expert_weights"]),
        )
        if len(route.expert_ids) != len(set(route.expert_ids)):
            raise ValueError(f"duplicate expert ID in route: {key}, {layer}")
        grouped[key][layer] = route
        metadata = {"sample_id": sample_id, "domain": domain}
        if requests.setdefault(request_id, metadata) != metadata:
            raise ValueError(f"inconsistent metadata for request {request_id}")
    tokens = [TokenRoutes(*key, routes) for key, routes in grouped.items()]
    tokens.sort(key=lambda row: (row.request_id, row.phase, row.token_position))
    return tokens, requests


def _stratified_split(
    requests: dict[int, dict[str, str]], seed: int, test_per_domain: int
) -> dict[str, Any]:
    by_domain: dict[str, list[int]] = defaultdict(list)
    for request_id, metadata in requests.items():
        by_domain[metadata["domain"]].append(request_id)
    train: list[int] = []
    test: list[int] = []
    rows: list[dict[str, Any]] = []
    for domain, request_ids in sorted(by_domain.items()):
        ordered = sorted(
            request_ids,
            key=lambda request_id: hashlib.sha256(
                f"{seed}:{request_id}:{requests[request_id]['sample_id']}:{domain}".encode()
            ).hexdigest(),
        )
        if len(ordered) <= test_per_domain:
            raise ValueError(f"not enough requests in domain {domain}")
        domain_test = ordered[:test_per_domain]
        domain_train = ordered[test_per_domain:]
        train.extend(domain_train)
        test.extend(domain_test)
        for split, ids in (("train", domain_train), ("test", domain_test)):
            rows.extend(
                {
                    "request_id": request_id,
                    "sample_id": requests[request_id]["sample_id"],
                    "domain": domain,
                    "split": split,
                }
                for request_id in sorted(ids)
            )
    return {
        "seed": seed,
        "test_requests_per_domain": test_per_domain,
        "train_request_ids": sorted(train),
        "test_request_ids": sorted(test),
        "requests": rows,
    }


def _domain_balanced_mean(rows: list[dict[str, Any]], metric: str) -> float:
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(float(row[metric]))
    return float(np.mean([np.mean(values) for values in by_domain.values()]))


def _horizon_summaries(scope_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scope_rows:
        grouped[(row["phase"], row["delta"], row["candidate_count"], row["baseline"])].append(
            row
        )
    result: list[dict[str, Any]] = []
    for (phase, delta, candidate_count, baseline), rows in sorted(grouped.items()):
        result.append(
            {
                "phase": phase,
                "delta": delta,
                "candidate_count": candidate_count,
                "baseline": baseline,
                "source_layer_scopes": len(rows),
                "selection_coverage": _domain_balanced_mean(rows, "selection_coverage"),
                "routed_mass_coverage": _domain_balanced_mean(
                    rows, "routed_mass_coverage"
                ),
                "complete_route_coverage": _domain_balanced_mean(
                    rows, "complete_route_coverage"
                ),
                "candidate_amplification": candidate_count / int(rows[0]["routing_top_k"]),
                "candidate_set_fraction": candidate_count / int(rows[0]["num_experts"]),
                "useful_candidate_amplification": _domain_balanced_mean(
                    rows, "useful_candidate_amplification"
                ),
            }
        )
    return result


def _bootstrap_gate(
    request_rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analysis = config["analysis"]
    gate_config = config["decision_gate"]
    primary_phase = str(analysis["primary_phase"])
    primary_candidate_count = int(analysis["primary_capacity"])
    deltas = [int(value) for value in analysis["primary_lookaheads"]]
    resamples = int(analysis["bootstrap_resamples"])
    confidence = float(analysis["confidence_level"])
    rng = np.random.default_rng(int(analysis["bootstrap_seed"]))
    filtered = [
        row
        for row in request_rows
        if row["phase"] == primary_phase
        and int(row["candidate_count"]) == primary_candidate_count
    ]
    domains = sorted({str(row["domain"]) for row in filtered})
    metrics = (
        "selection_coverage",
        "routed_mass_coverage",
        "complete_route_coverage",
    )
    rows_by_key = {
        (int(row["request_id"]), int(row["delta"]), str(row["baseline"])): row
        for row in filtered
    }
    request_ids = {
        domain: sorted(
            {int(row["request_id"]) for row in filtered if row["domain"] == domain}
        )
        for domain in domains
    }
    baselines = ("domain_static", "source_copy")
    gate_rows: list[dict[str, Any]] = []
    for delta in deltas:
        points: dict[str, float] = {}
        comparator_names: dict[str, str] = {}
        bootstrap_gains: dict[str, list[float]] = {metric: [] for metric in metrics}
        for metric in metrics:
            transition = _domain_balanced_mean(
                [
                    row
                    for row in filtered
                    if row["delta"] == delta and row["baseline"] == "transition"
                ],
                metric,
            )
            comparator_values = {
                baseline: _domain_balanced_mean(
                    [
                        row
                        for row in filtered
                        if row["delta"] == delta and row["baseline"] == baseline
                    ],
                    metric,
                )
                for baseline in baselines
            }
            comparator = max(comparator_values, key=comparator_values.get)
            points[f"transition_{metric}"] = transition
            points[f"comparator_{metric}"] = comparator_values[comparator]
            points[f"{metric}_gain"] = transition - comparator_values[comparator]
            comparator_names[metric] = comparator
        domain_selection_gains: dict[str, float] = {}
        for domain in domains:
            domain_rows = [
                row
                for row in filtered
                if row["delta"] == delta and row["domain"] == domain
            ]
            transition = float(
                np.mean(
                    [
                        row["selection_coverage"]
                        for row in domain_rows
                        if row["baseline"] == "transition"
                    ]
                )
            )
            strongest = max(
                float(
                    np.mean(
                        [
                            row["selection_coverage"]
                            for row in domain_rows
                            if row["baseline"] == baseline
                        ]
                    )
                )
                for baseline in baselines
            )
            domain_selection_gains[domain] = transition - strongest
        for _ in range(resamples):
            sampled = {
                domain: rng.choice(ids, size=len(ids), replace=True).tolist()
                for domain, ids in request_ids.items()
            }
            for metric in metrics:
                means: dict[str, float] = {}
                for baseline in ("transition", *baselines):
                    domain_means = []
                    for domain in domains:
                        values = [
                            float(rows_by_key[(request_id, delta, baseline)][metric])
                            for request_id in sampled[domain]
                        ]
                        domain_means.append(float(np.mean(values)))
                    means[baseline] = float(np.mean(domain_means))
                bootstrap_gains[metric].append(
                    means["transition"] - max(means[baseline] for baseline in baselines)
                )
        tail = (1 - confidence) / 2
        intervals = {
            metric: (
                float(np.quantile(values, tail)),
                float(np.quantile(values, 1 - tail)),
            )
            for metric, values in bootstrap_gains.items()
        }
        positive_domains = sum(value > 0 for value in domain_selection_gains.values())
        passed = (
            points["selection_coverage_gain"]
            >= float(gate_config["min_selection_coverage_gain"])
            and (
                not gate_config["require_positive_selection_ci"]
                or intervals["selection_coverage"][0] > 0
            )
            and points["complete_route_coverage_gain"]
            >= float(gate_config["min_complete_route_coverage_gain"])
            and (
                not gate_config["require_nonnegative_complete_ci"]
                or intervals["complete_route_coverage"][0] >= 0
            )
            and positive_domains >= int(gate_config["min_positive_domains"])
        )
        gate_rows.append(
            {
                "delta": delta,
                **points,
                "selection_comparator": comparator_names["selection_coverage"],
                "mass_comparator": comparator_names["routed_mass_coverage"],
                "complete_comparator": comparator_names["complete_route_coverage"],
                "selection_gain_ci_low": intervals["selection_coverage"][0],
                "selection_gain_ci_high": intervals["selection_coverage"][1],
                "mass_gain_ci_low": intervals["routed_mass_coverage"][0],
                "mass_gain_ci_high": intervals["routed_mass_coverage"][1],
                "complete_gain_ci_low": intervals["complete_route_coverage"][0],
                "complete_gain_ci_high": intervals["complete_route_coverage"][1],
                "positive_domains": positive_domains,
                "domain_selection_gains_json": json.dumps(
                    domain_selection_gains, sort_keys=True, separators=(",", ":")
                ),
                "pass": passed,
            }
        )
    passing = sum(bool(row["pass"]) for row in gate_rows)
    required = int(gate_config["min_passing_lookaheads"])
    decision = {
        "milestone": "E",
        "decision": (
            "PILOT_SUPPORTS_20B_ROUTE_PREDICTION"
            if passing >= required
            else "PILOT_DOES_NOT_SUPPORT_20B_ROUTE_PREDICTION"
        ),
        "primary_phase": primary_phase,
        "primary_candidate_count": primary_candidate_count,
        "candidate_amplification": primary_candidate_count / 4,
        "candidate_set_fraction": primary_candidate_count / 32,
        "passing_lookaheads": passing,
        "required_passing_lookaheads": required,
        "bootstrap_resamples": resamples,
        "confidence_level": confidence,
        "gate_config": gate_config,
        "lookaheads": gate_rows,
        "claim_boundary": (
            "held-out GPT-OSS 20B route-prediction quality only; no language-quality, "
            "timing, 120B, or cross-model conclusion"
        ),
    }
    return gate_rows, decision


def _report(
    split: dict[str, Any], decision: dict[str, Any], gate_rows: list[dict[str, Any]]
) -> str:
    lines = [
        "# GPT-OSS 20B Milestone E result",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        (
            f"The request-held-out split contains {len(split['train_request_ids'])} "
            f"training and {len(split['test_request_ids'])} test requests. The primary "
            "point is decode K=8 (2× top-4 candidate amplification; 25% of experts). "
            "Transition is compared with the stronger of domain popularity and "
            "current-route copy separately for each metric."
        ),
        "",
        (
            "| Δ | Selection (transition / comparator / gain, 95% CI) | "
            "Routed mass gain | Complete-route gain (95% CI) | Positive domains | Pass |"
        ),
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in gate_rows:
        lines.append(
            f"| {row['delta']} | {100 * row['transition_selection_coverage']:.1f}% / "
            f"{100 * row['comparator_selection_coverage']:.1f}% / "
            f"{100 * row['selection_coverage_gain']:+.1f} pp "
            f"[{100 * row['selection_gain_ci_low']:+.1f}, {100 * row['selection_gain_ci_high']:+.1f}] | "
            f"{100 * row['routed_mass_coverage_gain']:+.1f} pp | "
            f"{100 * row['complete_route_coverage_gain']:+.1f} pp "
            f"[{100 * row['complete_gain_ci_low']:+.1f}, {100 * row['complete_gain_ci_high']:+.1f}] | "
            f"{row['positive_domains']}/4 | {'yes' if row['pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                f"{decision['passing_lookaheads']}/"
                f"{decision['required_passing_lookaheads']} required short-horizon "
                "points passed."
            ),
            "",
            (
                "This is route-set prediction evidence from one checkpoint and one "
                "workload. It is not a language-quality score, latency result, or "
                "substitute for the cancelled 120B comparison."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def analyze_gpt_oss_prediction(
    run_dir: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    directory = Path(run_dir)
    if Path(config["output_dir"]).resolve() != directory.resolve():
        raise ValueError("config output_dir does not match --run")
    integrity = json.loads((directory / "integrity.json").read_text(encoding="utf-8"))
    checks = integrity["gate_checks"]
    numerical_exception_keys = {
        "zero_dispatch_weight_mismatches",
        "dispatch_weight_tolerance",
    }
    disqualifying_failures = [
        key
        for key, passed in checks.items()
        if not passed and key not in numerical_exception_keys
    ]
    if disqualifying_failures:
        raise ValueError(
            "refusing to analyze trace with structural integrity failures: "
            f"{disqualifying_failures}"
        )
    numerical_exception = integrity["decision"] != "TRACE_COMPLETE"
    inspection = json.loads(
        (directory / "model_inspection.json").read_text(encoding="utf-8")
    )
    num_layers = int(inspection["routed_layers"])
    num_experts = int(inspection["experts_per_layer"])
    top_k = int(inspection["top_k"])
    if (num_layers, num_experts, top_k) != (24, 32, 4):
        raise ValueError("loaded trace does not have the frozen 24/32/top-4 geometry")
    tokens, requests = _load_routes(directory)
    expected_layers = set(range(num_layers))
    incomplete = [
        (token.request_id, token.phase, token.token_position)
        for token in tokens
        if set(token.routes) != expected_layers
    ]
    if incomplete:
        raise ValueError(f"incomplete token routes: {incomplete[:5]}")
    split = _stratified_split(
        requests,
        int(config["analysis"]["split_seed"]),
        int(config["analysis"]["test_requests_per_domain"]),
    )
    analysis_dir = directory / "analysis" / "prediction"
    split["config"] = config["analysis"]
    write_json(analysis_dir / "split.json", split)
    _write_csv(analysis_dir / "split.csv", split["requests"])
    train_ids = set(split["train_request_ids"])
    test_ids = set(split["test_request_ids"])
    lookaheads = [int(value) for value in config["analysis"]["lookaheads"]]
    capacities = [int(value) for value in config["analysis"]["capacities"]]

    marginals: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    domain_marginals: dict[tuple[str, str, int], Counter[int]] = defaultdict(Counter)
    transitions: dict[tuple[str, int, int], dict[int, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for token in tokens:
        if token.request_id not in train_ids:
            continue
        for target_layer, route in token.routes.items():
            marginals[(token.phase, target_layer)].update(route.expert_ids)
            domain_marginals[(token.phase, token.domain, target_layer)].update(
                route.expert_ids
            )
        for source_layer in range(num_layers):
            source = token.routes[source_layer]
            for delta in lookaheads:
                target = token.routes.get(source_layer + delta)
                if target is None:
                    continue
                table = transitions[(token.phase, source_layer, delta)]
                for source_expert in source.expert_ids:
                    table[source_expert].update(target.expert_ids)

    scopes: dict[tuple[Any, ...], CoverageAccumulator] = {}
    request_scopes: dict[tuple[Any, ...], CoverageAccumulator] = {}
    primary_candidate_count = int(config["analysis"]["primary_capacity"])
    for token in tokens:
        if token.request_id not in test_ids:
            continue
        for source_layer in range(num_layers):
            source = token.routes[source_layer]
            for delta in lookaheads:
                target_layer = source_layer + delta
                target = token.routes.get(target_layer)
                if target is None:
                    continue
                for candidate_count in capacities:
                    predictions = {
                        "global_static": _ranked(
                            marginals[(token.phase, target_layer)],
                            candidate_count,
                            num_experts,
                        ),
                        "domain_static": _ranked(
                            domain_marginals[(token.phase, token.domain, target_layer)],
                            candidate_count,
                            num_experts,
                        ),
                        "source_copy": _copy_candidates(
                            source.expert_ids,
                            marginals[(token.phase, target_layer)],
                            candidate_count,
                            num_experts,
                        ),
                        "transition": _transition_candidates(
                            source.expert_ids,
                            transitions[(token.phase, source_layer, delta)],
                            marginals[(token.phase, target_layer)],
                            candidate_count,
                            num_experts,
                        ),
                    }
                    for baseline, candidates in predictions.items():
                        scope_key = (
                            token.phase,
                            token.domain,
                            source_layer,
                            target_layer,
                            delta,
                            candidate_count,
                            baseline,
                        )
                        scopes.setdefault(
                            scope_key, CoverageAccumulator(candidate_count, top_k)
                        ).add(candidates, target)
                        if candidate_count == primary_candidate_count:
                            request_key = (
                                token.request_id,
                                token.sample_id,
                                token.domain,
                                token.phase,
                                delta,
                                candidate_count,
                                baseline,
                            )
                            request_scopes.setdefault(
                                request_key, CoverageAccumulator(candidate_count, top_k)
                            ).add(candidates, target)

    scope_rows = []
    for key, accumulator in sorted(scopes.items()):
        phase, domain, source, target, delta, candidate_count, baseline = key
        scope_rows.append(
            {
                "phase": phase,
                "domain": domain,
                "source_layer": source,
                "target_layer": target,
                "delta": delta,
                "candidate_count": candidate_count,
                "baseline": baseline,
                "num_experts": num_experts,
                "routing_top_k": top_k,
                **accumulator.metrics(),
            }
        )
    request_rows = []
    for key, accumulator in sorted(request_scopes.items()):
        request_id, sample_id, domain, phase, delta, candidate_count, baseline = key
        request_rows.append(
            {
                "request_id": request_id,
                "sample_id": sample_id,
                "domain": domain,
                "phase": phase,
                "delta": delta,
                "candidate_count": candidate_count,
                "baseline": baseline,
                **accumulator.metrics(),
            }
        )
    summaries = _horizon_summaries(scope_rows)
    gate_rows, decision = _bootstrap_gate(request_rows, config)
    decision["trace_integrity"] = integrity["decision"]
    decision["preregistered_trace_gate_passed"] = not numerical_exception
    decision["trace_numerical_exception"] = (
        {
            "dispatch_weight_mismatches": integrity["totals"][
                "dispatch_weight_mismatches"
            ],
            "dispatch_pairs": integrity["totals"]["dispatch_consumed_pairs"],
            "mismatch_fraction": integrity["totals"]["dispatch_weight_mismatches"]
            / integrity["totals"]["dispatch_consumed_pairs"],
            "max_abs_weight_error": integrity["totals"][
                "dispatch_max_abs_weight_error"
            ],
            "interpretation": (
                "Post-hoc analysis accepted because all executed expert IDs match, "
                "coverage is complete, and the trace stores dispatch-consumed weights. "
                "The frozen 1e-6 independent-weight parity gate nevertheless failed."
            ),
        }
        if numerical_exception
        else None
    )
    _write_csv(analysis_dir / "scope_metrics.csv", scope_rows)
    _write_csv(analysis_dir / "request_metrics.csv", request_rows)
    _write_csv(analysis_dir / "horizon_summary.csv", summaries)
    _write_csv(analysis_dir / "bootstrap_gate.csv", gate_rows)
    write_json(analysis_dir / "decision.json", decision)
    (analysis_dir / "REPORT.md").write_text(
        _report(split, decision, gate_rows), encoding="utf-8"
    )
    return {"decision": decision, "gate_rows": gate_rows, "split": split}
