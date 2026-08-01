from __future__ import annotations

import bisect
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ep_predict.analysis.admission import _transition_score_vector
from ep_predict.analysis.h2 import TokenRoute, _load_token_routes, _ranked_candidates
from ep_predict.analysis.h3 import _load_feature_map
from ep_predict.analysis.h5 import _load_linear_arrays, _transition_tables
from ep_predict.tracing.storage import write_json


@dataclass
class ResidentEntry:
    last_use: int
    movement_id: int | None = None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    scale = float(values.std())
    if scale <= 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def _linear_score_matrix(
    arrays: dict[str, np.ndarray],
    *,
    phase: str,
    source_layer: int,
    delta: int,
    features: np.ndarray,
) -> np.ndarray:
    prefix = f"{phase}_l{source_layer:02d}_d{delta}"
    weight = arrays[f"{prefix}_weight"]
    bias = arrays[f"{prefix}_bias"]
    mean = arrays[f"{prefix}_mean"]
    scale = arrays[f"{prefix}_scale"]
    return (
        ((features.astype(np.float32, copy=False) - mean) / scale) @ weight.T
        + bias
    ).astype(np.float32, copy=False)


def _next_use_tables(
    demands: list[tuple[int, ...]],
) -> list[dict[int, int]]:
    occurrences: dict[int, list[int]] = defaultdict(list)
    for wave_index, demand in enumerate(demands):
        for expert in demand:
            occurrences[expert].append(wave_index)
    next_uses: list[dict[int, int]] = []
    sentinel = len(demands) + 1
    for wave_index in range(len(demands)):
        next_uses.append(
            {
                expert: (
                    positions[position]
                    if (
                        position := bisect.bisect_right(
                            positions, wave_index
                        )
                    )
                    < len(positions)
                    else sentinel
                )
                for expert, positions in occurrences.items()
            }
        )
    return next_uses


def _fixed_metrics(
    *,
    demands: list[tuple[int, ...]],
    residents: tuple[int, ...],
    capacity: int,
    movement_budget: int,
    expert_bytes: int,
) -> dict[str, Any]:
    resident = set(residents)
    demanded = sum(len(demand) for demand in demands)
    misses = sum(len(set(demand) - resident) for demand in demands)
    cold_waves = sum(not set(demand) <= resident for demand in demands)
    waves = len(demands)
    return _metric_result(
        waves=waves,
        demanded=demanded,
        misses=misses,
        cold_waves=cold_waves,
        insertions=0,
        evictions=0,
        useful_movements=0,
        wasted_movements=0,
        capacity=capacity,
        movement_budget=movement_budget,
        expert_bytes=expert_bytes,
    )


def _metric_result(
    *,
    waves: int,
    demanded: int,
    misses: int,
    cold_waves: int,
    insertions: int,
    evictions: int,
    useful_movements: int,
    wasted_movements: int,
    capacity: int,
    movement_budget: int,
    expert_bytes: int,
) -> dict[str, Any]:
    resident_hits = demanded - misses
    complete_hits = waves - cold_waves
    return {
        "n_waves": waves,
        "demand_expert_occurrences": demanded,
        "resident_hit_experts": resident_hits,
        "residual_cold_expert_demand": misses,
        "residual_cold_expert_fraction": misses / demanded if demanded else 0.0,
        "expert_stall_work_reduction": (
            resident_hits / demanded if demanded else 0.0
        ),
        "cold_waves": cold_waves,
        "complete_resident_set_hits": complete_hits,
        "complete_resident_set_hit_coverage": (
            complete_hits / waves if waves else 0.0
        ),
        "first_order_wave_stall_reduction": (
            complete_hits / waves if waves else 0.0
        ),
        "runtime_movement_experts": insertions,
        "runtime_movement_bytes": insertions * expert_bytes,
        "useful_residency_movement_experts": useful_movements,
        "useful_residency_movement_bytes": useful_movements * expert_bytes,
        "wasted_residency_movement_experts": wasted_movements,
        "wasted_residency_movement_bytes": wasted_movements * expert_bytes,
        "useful_movement_fraction": (
            useful_movements / insertions if insertions else 0.0
        ),
        "evictions": evictions,
        "residency_churn_per_wave": insertions / waves if waves else 0.0,
        "mean_resident_replacement_fraction": (
            insertions / (capacity * waves) if waves else 0.0
        ),
        "movement_budget_utilization": (
            insertions / (movement_budget * waves)
            if movement_budget and waves
            else 0.0
        ),
        "initial_resident_experts": capacity,
        "initial_resident_bytes": capacity * expert_bytes,
    }


def _dynamic_replay(
    *,
    demands: list[tuple[int, ...]],
    initial_residents: tuple[int, ...],
    capacity: int,
    movement_budget: int,
    expert_bytes: int,
    policy: str,
    score_vectors: np.ndarray | None = None,
    ema_alpha: float = 0.25,
    initial_belief: np.ndarray | None = None,
) -> dict[str, Any]:
    if len(initial_residents) != capacity:
        raise ValueError("initial resident set does not match capacity")
    if score_vectors is not None and len(score_vectors) != len(demands):
        raise ValueError("score vectors do not align with demand waves")
    if policy in {"transition", "linear"} and score_vectors is None:
        raise ValueError(f"{policy} replay requires score vectors")

    cache = {
        expert: ResidentEntry(last_use=rank - capacity)
        for rank, expert in enumerate(reversed(initial_residents))
    }
    movement_useful: dict[int, bool] = {}
    next_movement_id = 0
    misses_total = 0
    cold_waves = 0
    insertions = 0
    evictions = 0
    useful_finalized = 0
    wasted_finalized = 0
    belief = (
        initial_belief.astype(np.float32, copy=True)
        if initial_belief is not None
        else np.zeros(max(initial_residents) + 1, dtype=np.float32)
    )
    next_uses = _next_use_tables(demands) if policy == "oracle" else []

    def finalize(expert: int) -> None:
        nonlocal useful_finalized, wasted_finalized
        movement_id = cache[expert].movement_id
        if movement_id is None:
            return
        if movement_useful[movement_id]:
            useful_finalized += 1
        else:
            wasted_finalized += 1

    for wave_index, demand in enumerate(demands):
        missing = [expert for expert in demand if expert not in cache]
        misses_total += len(missing)
        cold_waves += bool(missing)

        if policy in {"transition", "linear"}:
            current = _zscore(score_vectors[wave_index])
            belief = (1.0 - ema_alpha) * belief + ema_alpha * current

        for expert in demand:
            entry = cache.get(expert)
            if entry is None:
                continue
            if entry.movement_id is not None:
                movement_useful[entry.movement_id] = True
            entry.last_use = wave_index

        if policy == "lru":
            candidates = list(missing)
        elif policy in {"transition", "linear"}:
            candidates = sorted(missing, key=lambda expert: (-belief[expert], expert))
        elif policy == "oracle":
            candidates = sorted(
                missing,
                key=lambda expert: (
                    next_uses[wave_index].get(expert, len(demands) + 1),
                    expert,
                ),
            )
        else:
            raise ValueError(f"unsupported dynamic residency policy {policy!r}")

        for candidate in candidates[:movement_budget]:
            if candidate in cache:
                continue
            if policy == "lru":
                victim = min(
                    cache,
                    key=lambda expert: (cache[expert].last_use, expert),
                )
                admit = True
            elif policy in {"transition", "linear"}:
                victim = min(cache, key=lambda expert: (belief[expert], expert))
                admit = belief[candidate] > belief[victim]
            else:
                victim = max(
                    cache,
                    key=lambda expert: (
                        next_uses[wave_index].get(expert, len(demands) + 1),
                        expert,
                    ),
                )
                candidate_next = next_uses[wave_index].get(
                    candidate, len(demands) + 1
                )
                victim_next = next_uses[wave_index].get(
                    victim, len(demands) + 1
                )
                admit = candidate_next < victim_next
            if not admit:
                continue
            finalize(victim)
            del cache[victim]
            movement_id = next_movement_id
            next_movement_id += 1
            movement_useful[movement_id] = False
            cache[candidate] = ResidentEntry(
                last_use=wave_index,
                movement_id=movement_id,
            )
            insertions += 1
            evictions += 1

    for expert in list(cache):
        finalize(expert)
    if useful_finalized + wasted_finalized != insertions:
        raise RuntimeError("movement usefulness accounting does not close")
    demanded = sum(len(demand) for demand in demands)
    return _metric_result(
        waves=len(demands),
        demanded=demanded,
        misses=misses_total,
        cold_waves=cold_waves,
        insertions=insertions,
        evictions=evictions,
        useful_movements=useful_finalized,
        wasted_movements=wasted_finalized,
        capacity=capacity,
        movement_budget=movement_budget,
        expert_bytes=expert_bytes,
    )


def _add_oracle_recovery(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["phase"],
            row["domain"],
            row["source_layer"],
            row["target_layer"],
            row["delta"],
            row["capacity"],
        )
        grouped[key][str(row["policy"])] = row
    for policies in grouped.values():
        lru = policies["lru"]
        oracle = policies["oracle"]
        for row in policies.values():
            for metric, output in (
                (
                    "expert_stall_work_reduction",
                    "expert_oracle_recovery_over_lru",
                ),
                (
                    "first_order_wave_stall_reduction",
                    "wave_oracle_recovery_over_lru",
                ),
            ):
                denominator = float(oracle[metric]) - float(lru[metric])
                row[output] = (
                    (float(row[metric]) - float(lru[metric])) / denominator
                    if denominator > 1e-12
                    else 0.0
                )


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row["phase"]),
                int(row["delta"]),
                int(row["capacity"]),
                str(row["policy"]),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for key, scopes in sorted(grouped.items()):
        phase, delta, capacity, policy = key
        result.append(
            {
                "phase": phase,
                "domain": "__domain_balanced__",
                "delta": delta,
                "capacity": capacity,
                "policy": policy,
                "n_layer_domain_scopes": len(scopes),
                "mean_residual_cold_expert_fraction": statistics.fmean(
                    float(row["residual_cold_expert_fraction"]) for row in scopes
                ),
                "mean_expert_stall_work_reduction": statistics.fmean(
                    float(row["expert_stall_work_reduction"]) for row in scopes
                ),
                "mean_complete_resident_set_hit_coverage": statistics.fmean(
                    float(row["complete_resident_set_hit_coverage"])
                    for row in scopes
                ),
                "mean_runtime_movement_experts_per_wave": statistics.fmean(
                    float(row["residency_churn_per_wave"]) for row in scopes
                ),
                "mean_runtime_movement_mib_per_wave": statistics.fmean(
                    float(row["runtime_movement_bytes"])
                    / float(row["n_waves"])
                    / (1024**2)
                    for row in scopes
                ),
                "mean_useful_movement_mib_per_wave": statistics.fmean(
                    float(row["useful_residency_movement_bytes"])
                    / float(row["n_waves"])
                    / (1024**2)
                    for row in scopes
                ),
                "mean_wasted_movement_mib_per_wave": statistics.fmean(
                    float(row["wasted_residency_movement_bytes"])
                    / float(row["n_waves"])
                    / (1024**2)
                    for row in scopes
                ),
                "mean_useful_movement_fraction": statistics.fmean(
                    float(row["useful_movement_fraction"]) for row in scopes
                ),
                "mean_evictions_per_wave": statistics.fmean(
                    float(row["evictions"]) / float(row["n_waves"])
                    for row in scopes
                ),
                "mean_resident_replacement_fraction": statistics.fmean(
                    float(row["mean_resident_replacement_fraction"])
                    for row in scopes
                ),
                "mean_expert_oracle_recovery_over_lru": statistics.fmean(
                    float(row["expert_oracle_recovery_over_lru"])
                    for row in scopes
                ),
                "mean_wave_oracle_recovery_over_lru": statistics.fmean(
                    float(row["wave_oracle_recovery_over_lru"])
                    for row in scopes
                ),
            }
        )
    return result


def _evaluate_gate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    phase = str(config["phase"])
    capacity = int(config["capacity_experts"])
    delta = int(config["lookahead"])
    comparators = {str(value) for value in config["comparators"]}
    candidates = [str(value) for value in config["candidate_policies"]]
    scoped = [
        row
        for row in rows
        if row["phase"] == phase
        and int(row["capacity"]) == capacity
        and int(row["delta"]) == delta
    ]
    by_scope: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scoped:
        by_scope[(str(row["domain"]), int(row["target_layer"]))][
            str(row["policy"])
        ] = row

    candidate_results: list[dict[str, Any]] = []
    for candidate in candidates:
        gains: list[dict[str, Any]] = []
        for (domain, layer), policies in sorted(by_scope.items()):
            if candidate not in policies or not comparators <= set(policies):
                continue
            expert_reference = max(
                float(policies[name]["expert_stall_work_reduction"])
                for name in comparators
            )
            wave_reference = max(
                float(policies[name]["first_order_wave_stall_reduction"])
                for name in comparators
            )
            gains.append(
                {
                    "domain": domain,
                    "layer": layer,
                    "expert_gain": (
                        float(policies[candidate]["expert_stall_work_reduction"])
                        - expert_reference
                    ),
                    "complete_gain": (
                        float(
                            policies[candidate][
                                "first_order_wave_stall_reduction"
                            ]
                        )
                        - wave_reference
                    ),
                }
            )
        domain_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        layer_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in gains:
            domain_groups[row["domain"]].append(row)
            layer_groups[row["layer"]].append(row)
        mean_expert = statistics.fmean(
            row["expert_gain"] for row in gains
        ) if gains else 0.0
        mean_complete = statistics.fmean(
            row["complete_gain"] for row in gains
        ) if gains else 0.0
        positive_fraction = statistics.fmean(
            row["expert_gain"] > 0 and row["complete_gain"] > 0
            for row in gains
        ) if gains else 0.0
        positive_domains = sum(
            statistics.fmean(row["expert_gain"] for row in values) > 0
            and statistics.fmean(row["complete_gain"] for row in values) > 0
            for values in domain_groups.values()
        )
        positive_layers = sum(
            statistics.fmean(row["expert_gain"] for row in values) > 0
            and statistics.fmean(row["complete_gain"] for row in values) > 0
            for values in layer_groups.values()
        )
        passed = (
            mean_expert
            >= float(config["min_mean_expert_stall_reduction_gain"])
            and mean_complete
            >= float(config["min_mean_complete_set_hit_gain"])
            and positive_fraction
            >= float(config["min_positive_scope_fraction_both"])
            and positive_domains >= int(config["min_positive_domains_both"])
            and positive_layers >= int(config["min_positive_layers_both"])
        )
        candidate_results.append(
            {
                "policy": candidate,
                "eligible_layer_domain_scopes": len(gains),
                "mean_expert_stall_reduction_gain": mean_expert,
                "mean_complete_set_hit_gain": mean_complete,
                "positive_scope_fraction_both": positive_fraction,
                "positive_domains_both": positive_domains,
                "positive_layers_both": positive_layers,
                "pass": passed,
            }
        )
    supported = any(row["pass"] for row in candidate_results)
    return {
        "hypothesis": "H6",
        "decision": "PILOT_SUPPORT" if supported else "PILOT_DOES_NOT_SUPPORT",
        "primary_scope": {
            "phase": phase,
            "capacity_experts": capacity,
            "lookahead": delta,
        },
        "comparators": sorted(comparators),
        "thresholds": {
            key: config[key]
            for key in (
                "min_mean_expert_stall_reduction_gain",
                "min_mean_complete_set_hit_gain",
                "min_positive_scope_fraction_both",
                "min_positive_domains_both",
                "min_positive_layers_both",
            )
        },
        "candidate_policies": candidate_results,
        "interpretation": (
            "At least one frozen prediction-guided residency policy materially "
            "beats the strongest simple placement baseline across the required "
            "layer/domain breadth. Require human review and fresh confirmation "
            "before any new model."
            if supported
            else "Neither frozen prediction-guided residency policy materially "
            "beats the strongest simple placement baseline at equal capacity "
            "and movement budget. Routing is predictable, but this pilot does "
            "not establish value for the tested residency mechanism."
        ),
    }


def _write_report(
    path: Path,
    *,
    gate: dict[str, Any],
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    scope = gate["primary_scope"]
    headline = [
        row
        for row in summaries
        if row["phase"] == scope["phase"]
        and int(row["capacity"]) == scope["capacity_experts"]
        and int(row["delta"]) == scope["lookahead"]
    ]
    lines = [
        "# H6 result: prediction-guided residency",
        "",
        f"**Decision:** {gate['decision']}",
        "",
        gate["interpretation"],
        "",
        "## Frozen primary scope",
        "",
        f"- Phase: {scope['phase']}",
        f"- Capacity: {scope['capacity_experts']} experts per layer",
        f"- Lookahead: Δ={scope['lookahead']}",
        f"- Runtime movement cap: "
        f"{config['replay']['movement_budget_experts_per_wave']} expert per wave",
        "- Prediction can admit only an actually demanded miss; there is no "
        "candidate-only prefetch.",
        "",
        "| Policy | Residual cold demand | Complete resident-set hits | "
        "Useful / wasted MiB per wave | Churn | "
        "Oracle recovery (expert / wave) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(headline, key=lambda item: str(item["policy"])):
        lines.append(
            f"| {row['policy']} | "
            f"{100 * float(row['mean_residual_cold_expert_fraction']):.1f}% | "
            f"{100 * float(row['mean_complete_resident_set_hit_coverage']):.1f}% | "
            f"{float(row['mean_useful_movement_mib_per_wave']):.2f} / "
            f"{float(row['mean_wasted_movement_mib_per_wave']):.2f} | "
            f"{100 * float(row['mean_resident_replacement_fraction']):.2f}% | "
            f"{100 * float(row['mean_expert_oracle_recovery_over_lru']):.1f}% / "
            f"{100 * float(row['mean_wave_oracle_recovery_over_lru']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate",
            "",
            "| Guided policy | Expert-stall gain | Complete-set gain | "
            "Positive scopes | Domains | Layers | Pass |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in gate["candidate_policies"]:
        lines.append(
            f"| {row['policy']} | "
            f"{100 * row['mean_expert_stall_reduction_gain']:+.1f} pp | "
            f"{100 * row['mean_complete_set_hit_gain']:+.1f} pp | "
            f"{100 * row['positive_scope_fraction_both']:.1f}% | "
            f"{row['positive_domains_both']} | "
            f"{row['positive_layers_both']} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "- Results use the existing 32-request development split and are "
            "pilot evidence, not fresh confirmation.",
            "- Domains are replayed independently to expose within-domain "
            "placement value; domain-switch reconfiguration cost is excluded.",
            "- Useful movement means a newly resident copy serves a later hit "
            "before eviction; all unresolved insertions are charged as wasted.",
            "- Stall reduction is first-order cold-work/wave elimination, not "
            "measured end-to-end latency.",
            "- The broad phase/layer/horizon/capacity scan is descriptive and "
            "does not alter the primary gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_h6(experiment_config: dict[str, Any]) -> dict[str, Any]:
    output = Path(str(experiment_config["output_dir"]))
    h3_run = Path(str(experiment_config["h3_run"]))
    h3_analysis = Path(str(experiment_config["h3_analysis"]))
    replay = experiment_config["replay"]
    capacities = [int(value) for value in replay["capacities"]]
    lookaheads = [int(value) for value in replay["lookaheads"]]
    phases = [str(value) for value in replay["phases"]]
    movement_budget = int(replay["movement_budget_experts_per_wave"])
    ema_alpha = float(replay["prediction_ema_alpha"])
    expert_bytes = int(replay["expert_bytes"])
    if movement_budget < 1:
        raise ValueError("movement budget must be positive")
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("prediction EMA alpha must be in (0, 1]")
    measurement = json.loads(
        Path(str(experiment_config["h4_measurement"])).read_text(
            encoding="utf-8"
        )
    )
    measured_expert_bytes = int(
        measurement["transfer"]["exact_expert_bytes"]
    )
    if expert_bytes != measured_expert_bytes:
        raise ValueError(
            "configured H6 expert bytes do not match the H4 measurement"
        )

    tokens, _requests = _load_token_routes(h3_run)
    split = json.loads((h3_analysis / "split.json").read_text(encoding="utf-8"))
    train_ids = {int(value) for value in split["train_request_ids"]}
    test_ids = {int(value) for value in split["test_request_ids"]}
    train_tokens = [token for token in tokens if token.request_id in train_ids]
    test_tokens = [token for token in tokens if token.request_id in test_ids]
    if len(test_ids) != 32 or len(train_ids) != 96:
        raise ValueError("H6 requires the preserved 96/32 request split")

    model_report = json.loads(
        (h3_run / "model_report.json").read_text(encoding="utf-8")
    )
    layers = sorted(int(router["layer_id"]) for router in model_report["routers"])
    num_experts_values = {
        int(router["num_experts"]) for router in model_report["routers"]
    }
    if len(num_experts_values) != 1:
        raise ValueError("H6 requires a uniform expert count")
    num_experts = num_experts_values.pop()
    if max(capacities) > num_experts:
        raise ValueError("capacity exceeds model expert count")

    requested_deltas = set(lookaheads)
    marginals, transitions = _transition_tables(train_tokens, requested_deltas)
    domain_marginals: dict[tuple[str, str, int], Counter[int]] = defaultdict(
        Counter
    )
    for token in train_tokens:
        for layer, demand in token.routes.items():
            domain_marginals[(token.phase, token.domain, layer)].update(demand)

    features, feature_integrity = _load_feature_map(h3_run)
    linear = _load_linear_arrays(h3_analysis / "linear_predictors.npz")
    domains = sorted({str(request["domain"]) for request in split["requests"]})
    test_by_scope: dict[tuple[str, str], list[TokenRoute]] = {}
    for phase in phases:
        for domain in domains:
            scoped = sorted(
                [
                    token
                    for token in test_tokens
                    if token.phase == phase and token.domain == domain
                ],
                key=lambda token: (token.request_id, token.token_position),
            )
            if not scoped:
                raise ValueError(f"empty H6 scope {phase}/{domain}")
            test_by_scope[(phase, domain)] = scoped

    rows: list[dict[str, Any]] = []
    baseline_cache: dict[
        tuple[str, str, int, int, str], dict[str, Any]
    ] = {}
    for phase in phases:
        for domain in domains:
            scoped_tokens = test_by_scope[(phase, domain)]
            for delta in lookaheads:
                for source_layer in layers:
                    target_layer = source_layer + delta
                    if target_layer not in layers:
                        continue
                    demands = [
                        tuple(dict.fromkeys(token.routes[target_layer]))
                        for token in scoped_tokens
                    ]
                    feature_matrix = np.stack(
                        [
                            features[
                                (
                                    token.request_id,
                                    token.phase,
                                    token.token_position,
                                    source_layer,
                                )
                            ]
                            for token in scoped_tokens
                        ]
                    )
                    transition_scores = np.stack(
                        [
                            _transition_score_vector(
                                token.routes[source_layer],
                                rows=transitions[
                                    (phase, source_layer, delta)
                                ],
                                marginal=marginals[(phase, target_layer)],
                                num_experts=num_experts,
                            )
                            for token in scoped_tokens
                        ]
                    )
                    linear_scores = _linear_score_matrix(
                        linear,
                        phase=phase,
                        source_layer=source_layer,
                        delta=delta,
                        features=feature_matrix,
                    )
                    prior = np.asarray(
                        [
                            marginals[(phase, target_layer)].get(expert, 0)
                            for expert in range(num_experts)
                        ],
                        dtype=np.float32,
                    )
                    for capacity in capacities:
                        static = _ranked_candidates(
                            marginals[(phase, target_layer)],
                            capacity,
                            num_experts,
                        )
                        domain_static = _ranked_candidates(
                            domain_marginals[(phase, domain, target_layer)],
                            capacity,
                            num_experts,
                        )
                        scope_key = (
                            phase,
                            domain,
                            target_layer,
                            capacity,
                        )
                        policies: dict[str, dict[str, Any]] = {}
                        for name in ("static", "domain", "lru", "oracle"):
                            cache_key = (*scope_key, name)
                            if cache_key not in baseline_cache:
                                if name == "static":
                                    result = _fixed_metrics(
                                        demands=demands,
                                        residents=static,
                                        capacity=capacity,
                                        movement_budget=movement_budget,
                                        expert_bytes=expert_bytes,
                                    )
                                elif name == "domain":
                                    result = _fixed_metrics(
                                        demands=demands,
                                        residents=domain_static,
                                        capacity=capacity,
                                        movement_budget=movement_budget,
                                        expert_bytes=expert_bytes,
                                    )
                                else:
                                    result = _dynamic_replay(
                                        demands=demands,
                                        initial_residents=static,
                                        capacity=capacity,
                                        movement_budget=movement_budget,
                                        expert_bytes=expert_bytes,
                                        policy=name,
                                    )
                                baseline_cache[cache_key] = result
                            policies[name] = baseline_cache[cache_key]
                        policies["transition"] = _dynamic_replay(
                            demands=demands,
                            initial_residents=static,
                            capacity=capacity,
                            movement_budget=movement_budget,
                            expert_bytes=expert_bytes,
                            policy="transition",
                            score_vectors=transition_scores,
                            ema_alpha=ema_alpha,
                            initial_belief=_zscore(prior),
                        )
                        policies["linear"] = _dynamic_replay(
                            demands=demands,
                            initial_residents=static,
                            capacity=capacity,
                            movement_budget=movement_budget,
                            expert_bytes=expert_bytes,
                            policy="linear",
                            score_vectors=linear_scores,
                            ema_alpha=ema_alpha,
                            initial_belief=_zscore(prior),
                        )
                        for policy, metrics in policies.items():
                            rows.append(
                                {
                                    "phase": phase,
                                    "domain": domain,
                                    "source_layer": source_layer,
                                    "target_layer": target_layer,
                                    "delta": delta,
                                    "capacity": capacity,
                                    "policy": policy,
                                    "movement_budget_experts_per_wave": (
                                        movement_budget
                                    ),
                                    "prediction_ema_alpha": (
                                        ema_alpha
                                        if policy in {"transition", "linear"}
                                        else ""
                                    ),
                                    **metrics,
                                }
                            )
            print(
                f"[h6] complete {phase}/{domain}: "
                f"{len(scoped_tokens)} token waves per target layer"
            )

    _add_oracle_recovery(rows)
    summaries = _summaries(rows)
    gate = _evaluate_gate(rows, experiment_config["decision_gate"])
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "scope_metrics.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    write_json(output / "gate.json", gate)
    _write_report(
        output / "REPORT.md",
        gate=gate,
        summaries=summaries,
        config=experiment_config,
    )
    summary = {
        "hypothesis": "H6",
        "decision": gate["decision"],
        "evidence_grade": "trace_driven_pilot",
        "new_inference_collection": False,
        "new_predictor_training": False,
        "train_requests": len(train_ids),
        "test_requests": len(test_ids),
        "router_layers": len(layers),
        "expert_count": num_experts,
        "feature_integrity": feature_integrity,
        "replay": replay,
        "scope_rows": len(rows),
        "summary_rows": len(summaries),
        "gate": gate,
        "outputs": {
            "scope_metrics": str(output / "scope_metrics.csv"),
            "summary": str(output / "summary.csv"),
            "gate": str(output / "gate.json"),
            "report": str(output / "REPORT.md"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
