from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import iter_trace_records, write_json


@dataclass(frozen=True)
class TokenRoute:
    request_id: int
    sample_id: str
    domain: str
    phase: str
    token_position: int
    routes: dict[int, tuple[int, ...]]


@dataclass
class MetricAccumulator:
    capacity: int
    top_k: int
    n_tokens: int = 0
    total_intersection: int = 0
    complete_tokens: int = 0
    churn_sum: float = 0.0
    jaccard_sum: float = 0.0
    churn_pairs: int = 0
    previous_candidates: tuple[int, ...] | None = None

    def add(
        self,
        candidates: tuple[int, ...],
        target: tuple[int, ...],
    ) -> None:
        if len(candidates) != self.capacity:
            raise ValueError("candidate set does not match configured capacity")
        if len(target) != self.top_k:
            raise ValueError("target route does not match configured top-k")
        intersection = len(set(candidates) & set(target))
        self.n_tokens += 1
        self.total_intersection += intersection
        self.complete_tokens += intersection == self.top_k
        if self.previous_candidates is not None:
            previous = set(self.previous_candidates)
            current = set(candidates)
            self.churn_sum += 1.0 - len(previous & current) / self.capacity
            self.jaccard_sum += len(previous & current) / len(previous | current)
            self.churn_pairs += 1
        self.previous_candidates = candidates

    def metrics(self) -> dict[str, Any]:
        if not self.n_tokens:
            raise ValueError("metric scope has no observations")
        return {
            "n_tokens": self.n_tokens,
            "routing_top_k": self.top_k,
            "selection_coverage": (
                self.total_intersection / (self.n_tokens * self.top_k)
            ),
            "complete_token_coverage": self.complete_tokens / self.n_tokens,
            "candidate_amplification": self.capacity / self.top_k,
            "useful_amplification": (
                self.n_tokens * self.capacity / self.total_intersection
                if self.total_intersection
                else math.inf
            ),
            "mean_candidate_replacement_fraction": (
                self.churn_sum / self.churn_pairs if self.churn_pairs else 0.0
            ),
            "mean_candidate_jaccard": (
                self.jaccard_sum / self.churn_pairs if self.churn_pairs else 1.0
            ),
        }


def _entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        count / total * math.log(count / total)
        for count in counter.values()
        if count
    )


def _ranked_candidates(
    scores: Counter[int] | dict[int, float],
    capacity: int,
    num_experts: int,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(num_experts),
            key=lambda expert: (-float(scores.get(expert, 0.0)), expert),
        )[:capacity]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_token_routes(
    run_dir: Path,
) -> tuple[list[TokenRoute], dict[int, dict[str, Any]]]:
    grouped: dict[
        tuple[int, str, str, str, int], dict[int, tuple[int, ...]]
    ] = defaultdict(dict)
    requests: dict[int, dict[str, Any]] = {}
    for record in iter_trace_records(run_dir):
        request_id = int(record["request_id"])
        sample_id = str(record["sample_id"])
        domain = str(record["domain"])
        phase = str(record["phase"])
        position = int(record["token_position"])
        layer = int(record["layer_id"])
        selected = tuple(int(expert) for expert in record["selected_expert_ids"])
        key = (request_id, sample_id, domain, phase, position)
        if layer in grouped[key]:
            raise ValueError(f"duplicate route record for {key}, layer {layer}")
        grouped[key][layer] = selected
        metadata = {"sample_id": sample_id, "domain": domain}
        previous = requests.setdefault(request_id, metadata)
        if previous != metadata:
            raise ValueError(f"inconsistent metadata for request {request_id}")

    tokens = [
        TokenRoute(
            request_id=key[0],
            sample_id=key[1],
            domain=key[2],
            phase=key[3],
            token_position=key[4],
            routes=routes,
        )
        for key, routes in grouped.items()
    ]
    tokens.sort(
        key=lambda token: (
            token.request_id,
            token.phase,
            token.token_position,
        )
    )
    return tokens, requests


def _stratified_split(
    requests: dict[int, dict[str, Any]],
    *,
    seed: int,
    test_per_domain: int,
) -> dict[str, Any]:
    by_domain: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for request_id, metadata in requests.items():
        by_domain[str(metadata["domain"])].append((request_id, metadata))

    train_ids: list[int] = []
    test_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    for domain, domain_requests in sorted(by_domain.items()):
        if len(domain_requests) <= test_per_domain:
            raise ValueError(
                f"domain {domain!r} has {len(domain_requests)} requests; "
                f"need more than {test_per_domain}"
            )

        def split_key(item: tuple[int, dict[str, Any]]) -> str:
            request_id, metadata = item
            payload = (
                f"{seed}:{request_id}:{metadata['sample_id']}:{domain}".encode()
            )
            return hashlib.sha256(payload).hexdigest()

        ordered = sorted(domain_requests, key=split_key)
        test = ordered[:test_per_domain]
        train = ordered[test_per_domain:]
        test_ids.extend(request_id for request_id, _ in test)
        train_ids.extend(request_id for request_id, _ in train)
        for split, selected in (("train", train), ("test", test)):
            for request_id, metadata in sorted(selected):
                rows.append(
                    {
                        "request_id": request_id,
                        "sample_id": metadata["sample_id"],
                        "domain": domain,
                        "split": split,
                    }
                )
    return {
        "split_seed": seed,
        "test_requests_per_domain": test_per_domain,
        "train_request_ids": sorted(train_ids),
        "test_request_ids": sorted(test_ids),
        "requests": rows,
    }


def _transition_candidates(
    source: tuple[int, ...],
    *,
    rows: dict[int, Counter[int]],
    marginal: Counter[int],
    capacity: int,
    num_experts: int,
) -> tuple[int, ...]:
    marginal_total = sum(marginal.values())
    scores: Counter[int] = Counter()
    for source_expert in source:
        row = rows.get(source_expert)
        row_total = sum(row.values()) if row else 0
        if row_total:
            for target_expert, count in row.items():
                scores[target_expert] += count / row_total
        elif marginal_total:
            for target_expert, count in marginal.items():
                scores[target_expert] += count / marginal_total
    if source:
        for target_expert in list(scores):
            scores[target_expert] /= len(source)
    return _ranked_candidates(scores, capacity, num_experts)


def _previous_window_candidates(
    test_tokens: list[TokenRoute],
    *,
    capacities: list[int],
    window_tokens: int,
    domain_counts: dict[tuple[str, str, int], Counter[int]],
    num_experts: int,
) -> dict[tuple[str, str, int, int, int, int], tuple[int, ...]]:
    grouped: dict[
        tuple[str, str, int], dict[int, list[tuple[int, tuple[int, ...]]]]
    ] = defaultdict(lambda: defaultdict(list))
    for token in test_tokens:
        for target_layer, target in token.routes.items():
            grouped[(token.phase, token.domain, target_layer)][token.request_id].append(
                (token.token_position, target)
            )

    result: dict[tuple[str, str, int, int, int, int], tuple[int, ...]] = {}
    for (phase, domain, target_layer), request_events in sorted(grouped.items()):
        windows: list[list[tuple[int, int, tuple[int, ...]]]] = []
        current: list[tuple[int, int, tuple[int, ...]]] = []
        for request_id in sorted(request_events):
            request = [
                (request_id, position, target)
                for position, target in sorted(request_events[request_id])
            ]
            current.extend(request)
            if len(current) >= window_tokens:
                windows.append(current)
                current = []
        if current:
            windows.append(current)

        fallback = domain_counts[(phase, domain, target_layer)]
        for window_index, window in enumerate(windows):
            previous = fallback
            if window_index > 0:
                previous = Counter(
                    expert
                    for _request_id, _position, target in windows[window_index - 1]
                    for expert in target
                )
            for capacity in capacities:
                candidates = _ranked_candidates(previous, capacity, num_experts)
                for request_id, position, _target in window:
                    result[
                        (
                            phase,
                            domain,
                            target_layer,
                            request_id,
                            position,
                            capacity,
                        )
                    ] = candidates
    return result


def _summarize_metrics(
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int, int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in metric_rows:
        grouped[
            (
                row["phase"],
                row["domain"],
                row["delta"],
                row["capacity"],
                row["baseline"],
            )
        ].append(row)
        grouped[
            (
                row["phase"],
                "__domain_balanced__",
                row["delta"],
                row["capacity"],
                row["baseline"],
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        phase, domain, delta, capacity, baseline = key
        summaries.append(
            {
                "phase": phase,
                "domain": domain,
                "delta": delta,
                "capacity": capacity,
                "baseline": baseline,
                "n_scopes": len(rows),
                "mean_selection_coverage": statistics.fmean(
                    row["selection_coverage"] for row in rows
                ),
                "mean_complete_token_coverage": statistics.fmean(
                    row["complete_token_coverage"] for row in rows
                ),
                "mean_candidate_amplification": statistics.fmean(
                    row["candidate_amplification"] for row in rows
                ),
                "mean_useful_amplification": statistics.fmean(
                    row["useful_amplification"] for row in rows
                ),
                "mean_candidate_replacement_fraction": statistics.fmean(
                    row["mean_candidate_replacement_fraction"] for row in rows
                ),
            }
        )
    return summaries


def _evaluate_gate(
    metric_rows: list[dict[str, Any]],
    gate_config: dict[str, Any],
    lookaheads: list[int],
) -> dict[str, Any]:
    phase = str(gate_config["phase"])
    baseline = str(gate_config["baseline"])
    comparator = str(gate_config["comparator"])
    capacity = int(gate_config["capacity_experts"])
    min_selection = float(gate_config["min_mean_selection_coverage_gain"])
    min_complete = float(gate_config["min_mean_complete_token_coverage_gain"])
    min_fraction = float(gate_config["min_positive_scope_fraction"])
    min_domains = int(gate_config["min_positive_domains"])

    lookup = {
        (
            row["phase"],
            row["domain"],
            row["source_layer"],
            row["delta"],
            row["capacity"],
            row["baseline"],
        ): row
        for row in metric_rows
    }
    delta_rows: list[dict[str, Any]] = []
    for delta in lookaheads:
        paired: list[dict[str, Any]] = []
        for key, candidate in lookup.items():
            row_phase, domain, source_layer, row_delta, row_capacity, row_baseline = (
                key
            )
            if (
                row_phase != phase
                or row_delta != delta
                or row_capacity != capacity
                or row_baseline != baseline
            ):
                continue
            reference = lookup.get(
                (
                    phase,
                    domain,
                    source_layer,
                    delta,
                    capacity,
                    comparator,
                )
            )
            if reference is None:
                continue
            paired.append(
                {
                    "domain": domain,
                    "source_layer": source_layer,
                    "selection_gain": (
                        candidate["selection_coverage"]
                        - reference["selection_coverage"]
                    ),
                    "complete_gain": (
                        candidate["complete_token_coverage"]
                        - reference["complete_token_coverage"]
                    ),
                }
            )
        domain_gains: dict[str, list[float]] = defaultdict(list)
        for row in paired:
            domain_gains[row["domain"]].append(row["selection_gain"])
        mean_selection = (
            statistics.fmean(row["selection_gain"] for row in paired)
            if paired
            else 0.0
        )
        mean_complete = (
            statistics.fmean(row["complete_gain"] for row in paired)
            if paired
            else 0.0
        )
        positive_fraction = (
            statistics.fmean(row["selection_gain"] > 0 for row in paired)
            if paired
            else 0.0
        )
        positive_domains = sum(
            statistics.fmean(gains) > 0 for gains in domain_gains.values()
        )
        passed = (
            bool(paired)
            and mean_selection >= min_selection
            and mean_complete >= min_complete
            and positive_fraction >= min_fraction
            and positive_domains >= min_domains
        )
        delta_rows.append(
            {
                "delta": delta,
                "eligible_scopes": len(paired),
                "mean_selection_coverage_gain": mean_selection,
                "mean_complete_token_coverage_gain": mean_complete,
                "positive_scope_fraction": positive_fraction,
                "positive_domains": positive_domains,
                "domain_mean_selection_gains": {
                    domain: statistics.fmean(gains)
                    for domain, gains in sorted(domain_gains.items())
                },
                "pass": passed,
            }
        )

    supported = any(row["pass"] for row in delta_rows)
    return {
        "hypothesis": "H2",
        "decision": "PILOT_SUPPORT" if supported else "PILOT_DOES_NOT_SUPPORT",
        "phase": phase,
        "baseline": baseline,
        "comparator": comparator,
        "capacity_experts": capacity,
        "thresholds": {
            "min_mean_selection_coverage_gain": min_selection,
            "min_mean_complete_token_coverage_gain": min_complete,
            "min_positive_scope_fraction": min_fraction,
            "min_positive_domains": min_domains,
        },
        "lookaheads": delta_rows,
        "interpretation": (
            "At least one routing-only transition baseline passed the held-out "
            f"{phase} gate; proceed to a lightweight external predictor after "
            "human figure review."
            if supported
            else f"No routing-only transition baseline passed the held-out {phase} "
            "gate; do not start a learned skip-layer predictor for this "
            "checkpoint without a revised hypothesis."
        ),
    }


def _write_report(
    path: Path,
    *,
    run_id: str,
    split: dict[str, Any],
    gate: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    capacity = gate["capacity_experts"]
    phase = gate["phase"]
    headline = [
        row
        for row in summaries
        if row["phase"] == phase
        and row["domain"] == "__domain_balanced__"
        and row["capacity"] == capacity
    ]
    lines = [
        f"# H2 result: `{run_id}`",
        "",
        f"**Decision:** {gate['decision']}",
        "",
        gate["interpretation"],
        "",
        "## Held-out design",
        "",
        f"- Train requests: {len(split['train_request_ids'])}",
        f"- Test requests: {len(split['test_request_ids'])}",
        "- Split unit: request, stratified by domain.",
        "- Prefill and decode are evaluated separately.",
        "",
        f"## Domain-balanced {phase} results at K={capacity}",
        "",
        "| Δ | Baseline | Selection coverage | Complete-token coverage | "
        "Candidate churn |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in sorted(headline, key=lambda item: (item["delta"], item["baseline"])):
        lines.append(
            f"| {row['delta']} | {row['baseline']} | "
            f"{100 * row['mean_selection_coverage']:.1f}% | "
            f"{100 * row['mean_complete_token_coverage']:.1f}% | "
            f"{100 * row['mean_candidate_replacement_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gate",
            "",
            "| Δ | Selection gain | Complete-token gain | Positive scopes | "
            "Positive domains | Pass |",
            "|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in gate["lookaheads"]:
        lines.append(
            f"| {row['delta']} | "
            f"{100 * row['mean_selection_coverage_gain']:+.1f} pp | "
            f"{100 * row['mean_complete_token_coverage_gain']:+.1f} pp | "
            f"{100 * row['positive_scope_fraction']:.1f}% | "
            f"{row['positive_domains']} | {'yes' if row['pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "This pilot establishes routing information only. It is not a "
            "latency, transfer-feasibility, or cross-model result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_h2(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    directory = Path(run_dir)
    configured_trace = Path(
        str(experiment_config.get("trace_run", directory))
    )
    if configured_trace.resolve() != directory.resolve():
        raise ValueError(
            f"config trace_run {configured_trace} does not match --run {directory}"
        )
    analysis_dir = directory / "analysis" / "h2"
    capacities = [int(value) for value in experiment_config["capacities"]]
    lookaheads = [int(value) for value in experiment_config["lookaheads"]]
    split_seed = int(experiment_config["split_seed"])
    test_per_domain = int(experiment_config["test_requests_per_domain"])
    window_tokens = int(experiment_config["previous_window_tokens"])

    model_report = json.loads(
        (directory / "model_report.json").read_text(encoding="utf-8")
    )
    layers = sorted(int(router["layer_id"]) for router in model_report["routers"])
    expert_counts = {int(router["num_experts"]) for router in model_report["routers"]}
    if len(expert_counts) != 1:
        raise ValueError("H2 currently requires one expert count across layers")
    num_experts = expert_counts.pop()
    if any(capacity > num_experts for capacity in capacities):
        raise ValueError("candidate capacity exceeds model expert count")

    tokens, requests = _load_token_routes(directory)
    expected_layers = set(layers)
    incomplete = [
        (token.request_id, token.phase, token.token_position)
        for token in tokens
        if set(token.routes) != expected_layers
    ]
    if incomplete:
        raise ValueError(f"tokens lack complete layer routes: {incomplete[:5]}")
    top_k_values = {
        len(route)
        for token in tokens
        for route in token.routes.values()
    }
    if len(top_k_values) != 1:
        raise ValueError(f"mixed routing top-k values: {sorted(top_k_values)}")

    config_fingerprint = hashlib.sha256(
        json.dumps(
            experiment_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    split = _stratified_split(
        requests,
        seed=split_seed,
        test_per_domain=test_per_domain,
    )
    split["analysis_id"] = str(
        experiment_config.get("analysis_id", "h2-analysis")
    )
    split["config_fingerprint"] = config_fingerprint
    split["experiment_config"] = experiment_config
    write_json(analysis_dir / "split.json", split)
    train_ids = set(split["train_request_ids"])
    test_ids = set(split["test_request_ids"])
    train_tokens = [token for token in tokens if token.request_id in train_ids]
    test_tokens = [token for token in tokens if token.request_id in test_ids]

    marginals: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    domain_marginals: dict[tuple[str, str, int], Counter[int]] = defaultdict(
        Counter
    )
    transitions: dict[
        tuple[str, int, int], dict[int, Counter[int]]
    ] = defaultdict(lambda: defaultdict(Counter))
    for token in train_tokens:
        for target_layer, target in token.routes.items():
            marginals[(token.phase, target_layer)].update(target)
            domain_marginals[(token.phase, token.domain, target_layer)].update(target)
        for source_layer in layers:
            source = token.routes[source_layer]
            for delta in lookaheads:
                target_layer = source_layer + delta
                target = token.routes.get(target_layer)
                if target is None:
                    continue
                table = transitions[(token.phase, source_layer, delta)]
                for source_expert in source:
                    table[source_expert].update(target)

    entropy_rows: list[dict[str, Any]] = []
    for (phase, source_layer, delta), table in sorted(transitions.items()):
        target_layer = source_layer + delta
        marginal_entropy = _entropy(marginals[(phase, target_layer)])
        row_weight = sum(sum(row.values()) for row in table.values())
        conditional_entropy = (
            sum(sum(row.values()) * _entropy(row) for row in table.values())
            / row_weight
            if row_weight
            else 0.0
        )
        entropy_rows.append(
            {
                "phase": phase,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "delta": delta,
                "marginal_entropy_nats": marginal_entropy,
                "conditional_entropy_nats": conditional_entropy,
                "entropy_reduction_nats": marginal_entropy - conditional_entropy,
                "normalized_entropy_reduction": (
                    (marginal_entropy - conditional_entropy) / marginal_entropy
                    if marginal_entropy
                    else 0.0
                ),
                "train_transition_observations": row_weight,
            }
        )

    previous_candidates = _previous_window_candidates(
        test_tokens,
        capacities=capacities,
        window_tokens=window_tokens,
        domain_counts=domain_marginals,
        num_experts=num_experts,
    )
    routing_top_k = next(iter(top_k_values))
    accumulators: dict[
        tuple[str, str, int, int, int, str], MetricAccumulator
    ] = {}
    max_capacity = max(capacities)
    static_rankings = {
        key: _ranked_candidates(counter, max_capacity, num_experts)
        for key, counter in marginals.items()
    }
    domain_rankings = {
        key: _ranked_candidates(counter, max_capacity, num_experts)
        for key, counter in domain_marginals.items()
    }
    for token in sorted(
        test_tokens,
        key=lambda item: (
            item.domain,
            item.phase,
            item.request_id,
            item.token_position,
        ),
    ):
        for source_layer in layers:
            source = token.routes[source_layer]
            for delta in lookaheads:
                target_layer = source_layer + delta
                target = token.routes.get(target_layer)
                if target is None:
                    continue
                static_ranking = static_rankings[(token.phase, target_layer)]
                domain_ranking = domain_rankings[
                    (token.phase, token.domain, target_layer)
                ]
                transition_ranking = _transition_candidates(
                    source,
                    rows=transitions[(token.phase, source_layer, delta)],
                    marginal=marginals[(token.phase, target_layer)],
                    capacity=max_capacity,
                    num_experts=num_experts,
                )
                for capacity in capacities:
                    predictions = {
                        "static": static_ranking[:capacity],
                        "domain_oracle": domain_ranking[:capacity],
                        "previous_window": previous_candidates[
                            (
                                token.phase,
                                token.domain,
                                target_layer,
                                token.request_id,
                                token.token_position,
                                capacity,
                            )
                        ],
                        "transition": transition_ranking[:capacity],
                    }
                    for baseline, candidates in predictions.items():
                        key = (
                            token.phase,
                            token.domain,
                            source_layer,
                            delta,
                            capacity,
                            baseline,
                        )
                        accumulator = accumulators.setdefault(
                            key,
                            MetricAccumulator(
                                capacity=capacity,
                                top_k=routing_top_k,
                            ),
                        )
                        accumulator.add(candidates, target)

    metric_rows = []
    for key, accumulator in sorted(accumulators.items()):
        metric_rows.append(
            {
                "phase": key[0],
                "domain": key[1],
                "source_layer": key[2],
                "target_layer": key[2] + key[3],
                "delta": key[3],
                "capacity": key[4],
                "baseline": key[5],
                **accumulator.metrics(),
            }
        )
    summary_rows = _summarize_metrics(metric_rows)
    gate = _evaluate_gate(
        metric_rows,
        experiment_config["decision_gate"],
        lookaheads,
    )

    _write_csv(analysis_dir / "metrics.csv", metric_rows)
    _write_csv(analysis_dir / "entropy.csv", entropy_rows)
    _write_csv(analysis_dir / "summary.csv", summary_rows)
    write_json(analysis_dir / "gate.json", gate)
    manifest = json.loads(
        (directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    summary = {
        "run_id": manifest["run_id"],
        "hypothesis": "H2",
        "analysis_id": split["analysis_id"],
        "config_fingerprint": config_fingerprint,
        "evidence_grade": "pilot",
        "trace_reused": str(directory),
        "new_inference_collection": False,
        "train_requests": len(train_ids),
        "test_requests": len(test_ids),
        "token_routes": len(tokens),
        "routing_top_k": routing_top_k,
        "num_experts": num_experts,
        "gate": gate,
        "outputs": {
            "split": str(analysis_dir / "split.json"),
            "metrics": str(analysis_dir / "metrics.csv"),
            "entropy": str(analysis_dir / "entropy.csv"),
            "summary": str(analysis_dir / "summary.csv"),
            "gate": str(analysis_dir / "gate.json"),
            "report": str(analysis_dir / "REPORT.md"),
        },
    }
    write_json(analysis_dir / "summary.json", summary)
    _write_report(
        analysis_dir / "REPORT.md",
        run_id=manifest["run_id"],
        split=split,
        gate=gate,
        summaries=summary_rows,
    )
    return summary
