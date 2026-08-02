from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ep_predict.analysis.h2 import (
    MetricAccumulator,
    TokenRoute,
    _load_token_routes,
    _ranked_candidates,
    _stratified_split,
    _summarize_metrics,
    _transition_candidates,
    _write_csv,
)
from ep_predict.tracing.storage import write_json


FeatureKey = tuple[int, str, int, int]


@dataclass(frozen=True)
class LinearHead:
    phase: str
    source_layer: int
    delta: int
    weight: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    final_loss: float
    n_train: int

    def scores(self, features: np.ndarray) -> np.ndarray:
        standardized = (
            features.astype(np.float32, copy=False) - self.feature_mean
        ) / self.feature_scale
        return standardized @ self.weight.T + self.bias


def _load_feature_map(
    run_dir: Path,
) -> tuple[dict[FeatureKey, np.ndarray], dict[str, Any]]:
    feature_paths = sorted((run_dir / "features").glob("request-*.npz"))
    trace_paths = sorted((run_dir / "trace").glob("request-*.jsonl.gz"))
    if not feature_paths:
        raise FileNotFoundError(f"no feature shards under {run_dir / 'features'}")
    if len(feature_paths) != len(trace_paths):
        raise ValueError(
            f"feature/trace shard count mismatch: "
            f"{len(feature_paths)} vs {len(trace_paths)}"
        )

    trace_by_stem = {
        path.name.removesuffix(".jsonl.gz"): path for path in trace_paths
    }
    features: dict[FeatureKey, np.ndarray] = {}
    dimensions: set[int] = set()
    rows = 0
    for feature_path in feature_paths:
        stem = feature_path.name.removesuffix(".npz")
        trace_path = trace_by_stem.get(stem)
        if trace_path is None:
            raise ValueError(f"no routing shard matches {feature_path}")
        with np.load(feature_path, allow_pickle=False) as shard:
            required = {
                "hidden_feature",
                "request_id",
                "sample_id",
                "phase",
                "token_position",
                "input_token_id",
                "layer_id",
                "moe_layer_index",
            }
            missing = required - set(shard.files)
            if missing:
                raise ValueError(f"{feature_path} lacks arrays {sorted(missing)}")
            matrix = shard["hidden_feature"]
            if matrix.ndim != 2 or matrix.dtype != np.float16:
                raise ValueError(f"{feature_path} has invalid feature matrix")
            if not np.isfinite(matrix).all():
                raise ValueError(f"{feature_path} contains non-finite features")
            dimensions.add(int(matrix.shape[1]))
            metadata = {
                name: shard[name]
                for name in (
                    "phase",
                    "token_position",
                    "input_token_id",
                    "layer_id",
                    "moe_layer_index",
                )
            }
            request_id = int(shard["request_id"])
            sample_id = str(shard["sample_id"])
            if any(len(array) != len(matrix) for array in metadata.values()):
                raise ValueError(f"{feature_path} metadata does not align")

            with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
                trace_records = [json.loads(line) for line in handle]
            if len(trace_records) != len(matrix):
                raise ValueError(
                    f"{feature_path} has {len(matrix)} rows but "
                    f"{trace_path} has {len(trace_records)}"
                )
            for index, record in enumerate(trace_records):
                phase_code = 0 if record["phase"] == "prefill" else 1
                expected = (
                    request_id,
                    sample_id,
                    phase_code,
                    int(metadata["token_position"][index]),
                    int(metadata["input_token_id"][index]),
                    int(metadata["layer_id"][index]),
                    int(metadata["moe_layer_index"][index]),
                )
                observed = (
                    int(record["request_id"]),
                    str(record["sample_id"]),
                    int(metadata["phase"][index]),
                    int(record["token_position"]),
                    int(record["input_token_id"]),
                    int(record["layer_id"]),
                    int(record["moe_layer_index"]),
                )
                if observed != expected:
                    raise ValueError(
                        f"feature/route alignment mismatch at "
                        f"{feature_path}:{index}"
                    )
                key = (
                    request_id,
                    str(record["phase"]),
                    int(record["token_position"]),
                    int(record["layer_id"]),
                )
                if key in features:
                    raise ValueError(f"duplicate feature key {key}")
                features[key] = matrix[index].copy()
            rows += len(matrix)
    if len(dimensions) != 1:
        raise ValueError(f"mixed projected feature dimensions: {dimensions}")
    return features, {
        "feature_shards": len(feature_paths),
        "feature_rows": rows,
        "feature_dimension": dimensions.pop(),
        "feature_dtype": "float16",
        "feature_route_alignment": "pass",
        "non_finite_feature_rows": 0,
    }


def _training_arrays(
    tokens: list[TokenRoute],
    features: dict[FeatureKey, np.ndarray],
    *,
    phase: str,
    source_layer: int,
    delta: int,
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        token
        for token in tokens
        if token.phase == phase and source_layer + delta in token.routes
    ]
    if not selected:
        raise ValueError(
            f"no samples for phase={phase}, layer={source_layer}, delta={delta}"
        )
    x = np.stack(
        [
            features[
                (
                    token.request_id,
                    token.phase,
                    token.token_position,
                    source_layer,
                )
            ]
            for token in selected
        ]
    ).astype(np.float32)
    y = np.zeros((len(selected), num_experts), dtype=np.float32)
    for row, token in enumerate(selected):
        y[row, list(token.routes[source_layer + delta])] = 1.0
    return x, y


def _fit_linear_head(
    x: np.ndarray,
    y: np.ndarray,
    *,
    phase: str,
    source_layer: int,
    delta: int,
    config: dict[str, Any],
) -> LinearHead:
    import torch

    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("linear predictor inputs must be aligned rank-2 arrays")
    device_name = str(config.get("device", "cuda:0"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"predictor accelerator device {device_name} is unavailable; "
            "ROCm uses the same torch.cuda device API as CUDA"
        )
    device = torch.device(device_name)
    seed = int(config["training_seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_cpu = torch.from_numpy(x.astype(np.float32, copy=False))
    y_cpu = torch.from_numpy(y.astype(np.float32, copy=False))
    mean = x_cpu.mean(dim=0)
    scale = x_cpu.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    x_cpu = (x_cpu - mean) / scale

    model = torch.nn.Linear(x.shape[1], y.shape[1], bias=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    positive_weight = torch.full(
        (y.shape[1],),
        float(config["positive_class_weight"]),
        device=device,
    )
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    final_loss = math.nan
    model.train()
    for _epoch in range(epochs):
        loss_sum = 0.0
        observations = 0
        for start in range(0, len(x_cpu), batch_size):
            stop = min(start + batch_size, len(x_cpu))
            batch_x = x_cpu[start:stop].to(device)
            batch_y = y_cpu[start:stop].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            count = stop - start
            loss_sum += float(loss.detach().cpu()) * count
            observations += count
        final_loss = loss_sum / observations

    return LinearHead(
        phase=phase,
        source_layer=source_layer,
        delta=delta,
        weight=model.weight.detach().float().cpu().numpy(),
        bias=model.bias.detach().float().cpu().numpy(),
        feature_mean=mean.numpy(),
        feature_scale=scale.numpy(),
        final_loss=final_loss,
        n_train=len(x),
    )


def _save_heads(path: Path, heads: list[LinearHead]) -> str:
    arrays: dict[str, np.ndarray] = {}
    for head in heads:
        prefix = f"{head.phase}_l{head.source_layer:02d}_d{head.delta}"
        arrays[f"{prefix}_weight"] = head.weight
        arrays[f"{prefix}_bias"] = head.bias
        arrays[f"{prefix}_mean"] = head.feature_mean
        arrays[f"{prefix}_scale"] = head.feature_scale
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluate_gate(
    metric_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    phase = str(config["phase"])
    delta = int(config["lookahead"])
    capacity = int(config["capacity_experts"])
    baseline = str(config["baseline"])
    comparator = str(config["comparator"])
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
    paired: list[dict[str, Any]] = []
    for key, candidate in lookup.items():
        row_phase, domain, source_layer, row_delta, row_capacity, policy = key
        if (
            row_phase != phase
            or row_delta != delta
            or row_capacity != capacity
            or policy != baseline
        ):
            continue
        reference = lookup.get(
            (phase, domain, source_layer, delta, capacity, comparator)
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

    mean_selection = (
        statistics.fmean(row["selection_gain"] for row in paired) if paired else 0.0
    )
    mean_complete = (
        statistics.fmean(row["complete_gain"] for row in paired) if paired else 0.0
    )
    positive_selection_fraction = (
        statistics.fmean(row["selection_gain"] > 0 for row in paired)
        if paired
        else 0.0
    )
    positive_complete_fraction = (
        statistics.fmean(row["complete_gain"] > 0 for row in paired)
        if paired
        else 0.0
    )
    domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        domains[row["domain"]].append(row)
    domain_gains = {
        domain: {
            "selection_gain": statistics.fmean(
                row["selection_gain"] for row in rows
            ),
            "complete_gain": statistics.fmean(
                row["complete_gain"] for row in rows
            ),
        }
        for domain, rows in sorted(domains.items())
    }
    positive_domains = sum(
        gains["selection_gain"] > 0 and gains["complete_gain"] > 0
        for gains in domain_gains.values()
    )
    thresholds = {
        key: config[key]
        for key in (
            "min_mean_selection_coverage_gain",
            "min_mean_complete_token_coverage_gain",
            "min_positive_selection_scope_fraction",
            "min_positive_complete_scope_fraction",
            "min_positive_domains",
        )
    }
    passed = (
        bool(paired)
        and mean_selection
        >= float(thresholds["min_mean_selection_coverage_gain"])
        and mean_complete
        >= float(thresholds["min_mean_complete_token_coverage_gain"])
        and positive_selection_fraction
        >= float(thresholds["min_positive_selection_scope_fraction"])
        and positive_complete_fraction
        >= float(thresholds["min_positive_complete_scope_fraction"])
        and positive_domains >= int(thresholds["min_positive_domains"])
    )
    return {
        "hypothesis": "H3",
        "decision": "PILOT_SUPPORT" if passed else "PILOT_DOES_NOT_SUPPORT",
        "phase": phase,
        "lookahead": delta,
        "capacity_experts": capacity,
        "baseline": baseline,
        "comparator": comparator,
        "eligible_scopes": len(paired),
        "mean_selection_coverage_gain": mean_selection,
        "mean_complete_token_coverage_gain": mean_complete,
        "positive_selection_scope_fraction": positive_selection_fraction,
        "positive_complete_scope_fraction": positive_complete_fraction,
        "positive_domains_both_metrics": positive_domains,
        "domain_mean_gains": domain_gains,
        "thresholds": thresholds,
        "pass": passed,
        "interpretation": (
            "The fixed linear sidecar materially beats the transition table; "
            "carry it into H4 without predictor optimization."
            if passed
            else "The fixed linear sidecar does not materially beat the "
            "transition table on the primary gate; use the simpler transition "
            "policy in H4 and stop learned-predictor work for this checkpoint."
        ),
    }


def _compare_h2_transition(
    rows: list[dict[str, Any]],
    reference_path: Path,
) -> dict[str, Any]:
    h3 = {
        (
            row["phase"],
            row["domain"],
            int(row["source_layer"]),
            int(row["delta"]),
            int(row["capacity"]),
        ): row
        for row in rows
        if row["baseline"] == "transition"
    }
    with reference_path.open("r", encoding="utf-8", newline="") as handle:
        h2_rows = list(csv.DictReader(handle))
    differences: list[float] = []
    compared = 0
    for row in h2_rows:
        if row["baseline"] != "transition":
            continue
        key = (
            row["phase"],
            row["domain"],
            int(row["source_layer"]),
            int(row["delta"]),
            int(row["capacity"]),
        )
        candidate = h3.get(key)
        if candidate is None:
            raise ValueError(f"H3 lacks H2 transition scope {key}")
        for metric in ("selection_coverage", "complete_token_coverage"):
            differences.append(abs(float(candidate[metric]) - float(row[metric])))
        compared += 1
    maximum = max(differences, default=math.inf)
    if maximum > 1e-12:
        raise ValueError(f"H2 transition reproduction drift is {maximum}")
    return {
        "reference_metrics": str(reference_path),
        "compared_transition_scopes": compared,
        "max_absolute_coverage_difference": maximum,
        "status": "pass",
    }


def _write_report(
    path: Path,
    *,
    run_id: str,
    gate: dict[str, Any],
    summaries: list[dict[str, Any]],
    integrity: dict[str, Any],
) -> None:
    phase = gate["phase"]
    capacity = gate["capacity_experts"]
    headline = [
        row
        for row in summaries
        if row["phase"] == phase
        and row["domain"] == "__domain_balanced__"
        and row["capacity"] == capacity
        and row["baseline"] in {"static", "domain_oracle", "transition", "linear"}
    ]
    lines = [
        f"# H3 result: `{run_id}`",
        "",
        f"**Decision:** {gate['decision']}",
        "",
        gate["interpretation"],
        "",
        f"## Domain-balanced {phase} results at K={capacity}",
        "",
        "| Δ | Policy | Selection coverage | Complete-token coverage | "
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
            "## Preregistered primary gate",
            "",
            f"- Selection gain: "
            f"{100 * gate['mean_selection_coverage_gain']:+.1f} pp.",
            f"- Complete-token gain: "
            f"{100 * gate['mean_complete_token_coverage_gain']:+.1f} pp.",
            f"- Positive selection scopes: "
            f"{100 * gate['positive_selection_scope_fraction']:.1f}%.",
            f"- Positive complete-token scopes: "
            f"{100 * gate['positive_complete_scope_fraction']:.1f}%.",
            f"- Domains positive on both metrics: "
            f"{gate['positive_domains_both_metrics']}/4.",
            "",
            "## Integrity",
            "",
            f"- Feature/route alignment: "
            f"{integrity['features']['feature_route_alignment']}.",
            f"- Feature rows: {integrity['features']['feature_rows']:,}.",
            f"- H2 transition scopes reproduced: "
            f"{integrity['h2_transition_reproduction']['compared_transition_scopes']}.",
            f"- Maximum H2 coverage difference: "
            f"{integrity['h2_transition_reproduction']['max_absolute_coverage_difference']:.3g}.",
            "",
            "This is a single-checkpoint pilot. It does not establish physical "
            "transfer feasibility, latency improvement, or universal MoE "
            "behavior.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_h3(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    directory = Path(run_dir)
    if Path(str(experiment_config["output_dir"])).resolve() != directory.resolve():
        raise ValueError("experiment output_dir does not match --run")
    analysis_config = experiment_config["analysis"]
    predictor_config = experiment_config["linear_predictor"]
    capacities = [int(value) for value in analysis_config["capacities"]]
    lookaheads = [int(value) for value in analysis_config["lookaheads"]]
    output_name = str(analysis_config.get("output_name", "h3"))
    if not output_name or "/" in output_name or output_name in {".", ".."}:
        raise ValueError("analysis output_name must be one safe path component")
    analysis_dir = directory / "analysis" / output_name

    manifest = json.loads(
        (directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("state") != "complete":
        raise ValueError("H3 collection manifest is not complete")
    projection_report = json.loads(
        (directory / "projection_report.json").read_text(encoding="utf-8")
    )
    expected_dimension = int(experiment_config["hidden_features"]["dimension"])
    if projection_report["output_dimension"] != expected_dimension:
        raise ValueError("projection report does not match H3 config")

    model_report = json.loads(
        (directory / "model_report.json").read_text(encoding="utf-8")
    )
    layers = sorted(int(router["layer_id"]) for router in model_report["routers"])
    expert_counts = {int(router["num_experts"]) for router in model_report["routers"]}
    if len(expert_counts) != 1:
        raise ValueError("H3 requires one expert count across layers")
    num_experts = expert_counts.pop()
    if max(capacities) > num_experts:
        raise ValueError("candidate capacity exceeds expert count")

    tokens, requests = _load_token_routes(directory)
    expected_layers = set(layers)
    if any(set(token.routes) != expected_layers for token in tokens):
        raise ValueError("H3 trace contains incomplete token routes")
    top_k_values = {
        len(route) for token in tokens for route in token.routes.values()
    }
    if len(top_k_values) != 1:
        raise ValueError("H3 trace mixes routing top-k")
    routing_top_k = next(iter(top_k_values))
    features, feature_integrity = _load_feature_map(directory)
    if feature_integrity["feature_dimension"] != expected_dimension:
        raise ValueError("feature shards do not match configured projection size")
    if feature_integrity["feature_rows"] != len(tokens) * len(layers):
        raise ValueError("feature row count does not match token-layer count")

    split = _stratified_split(
        requests,
        seed=int(analysis_config["split_seed"]),
        test_per_domain=int(analysis_config["test_requests_per_domain"]),
    )
    reference_split_path = Path(str(analysis_config["reference_split"]))
    reference_split = json.loads(reference_split_path.read_text(encoding="utf-8"))
    for key in ("train_request_ids", "test_request_ids", "requests"):
        if split[key] != reference_split[key]:
            raise ValueError(f"H3 split differs from H2 for {key}")
    split["analysis_id"] = str(analysis_config["analysis_id"])
    split["reference_split"] = str(reference_split_path)
    split["split_reproduction"] = "pass"
    write_json(analysis_dir / "split.json", split)
    train_ids = set(split["train_request_ids"])
    test_ids = set(split["test_request_ids"])
    train_tokens = [token for token in tokens if token.request_id in train_ids]
    test_tokens = [token for token in tokens if token.request_id in test_ids]

    marginals: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    domain_marginals: dict[tuple[str, str, int], Counter[int]] = defaultdict(Counter)
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
                target = token.routes.get(source_layer + delta)
                if target is None:
                    continue
                table = transitions[(token.phase, source_layer, delta)]
                for source_expert in source:
                    table[source_expert].update(target)

    heads: list[LinearHead] = []
    training_rows: list[dict[str, Any]] = []
    for phase in ("prefill", "decode"):
        for source_layer in layers:
            for delta in lookaheads:
                if source_layer + delta not in expected_layers:
                    continue
                x, y = _training_arrays(
                    train_tokens,
                    features,
                    phase=phase,
                    source_layer=source_layer,
                    delta=delta,
                    num_experts=num_experts,
                )
                head = _fit_linear_head(
                    x,
                    y,
                    phase=phase,
                    source_layer=source_layer,
                    delta=delta,
                    config=predictor_config,
                )
                heads.append(head)
                training_rows.append(
                    {
                        "phase": phase,
                        "source_layer": source_layer,
                        "target_layer": source_layer + delta,
                        "delta": delta,
                        "n_train": head.n_train,
                        "feature_dimension": expected_dimension,
                        "output_dimension": num_experts,
                        "epochs": int(predictor_config["epochs"]),
                        "final_bce_loss": head.final_loss,
                    }
                )
                print(
                    f"[train] {phase} l{source_layer}->l{source_layer + delta}: "
                    f"{head.n_train} samples, loss={head.final_loss:.4f}"
                )

    predictor_path = analysis_dir / "linear_predictors.npz"
    predictor_sha256 = _save_heads(predictor_path, heads)
    _write_csv(analysis_dir / "training.csv", training_rows)

    max_capacity = max(capacities)
    static_rankings = {
        key: _ranked_candidates(counter, max_capacity, num_experts)
        for key, counter in marginals.items()
    }
    domain_rankings = {
        key: _ranked_candidates(counter, max_capacity, num_experts)
        for key, counter in domain_marginals.items()
    }
    accumulators: dict[
        tuple[str, str, int, int, int, str], MetricAccumulator
    ] = {}
    for head in heads:
        scoped_tokens = sorted(
            [
                token
                for token in test_tokens
                if token.phase == head.phase
                and head.source_layer + head.delta in token.routes
            ],
            key=lambda token: (
                token.domain,
                token.request_id,
                token.token_position,
            ),
        )
        x = np.stack(
            [
                features[
                    (
                        token.request_id,
                        token.phase,
                        token.token_position,
                        head.source_layer,
                    )
                ]
                for token in scoped_tokens
            ]
        )
        scores = head.scores(x)
        linear_rankings = np.argsort(-scores, axis=1, kind="stable")[
            :, :max_capacity
        ]
        target_layer = head.source_layer + head.delta
        for row_index, token in enumerate(scoped_tokens):
            target = token.routes[target_layer]
            transition = _transition_candidates(
                token.routes[head.source_layer],
                rows=transitions[(head.phase, head.source_layer, head.delta)],
                marginal=marginals[(head.phase, target_layer)],
                capacity=max_capacity,
                num_experts=num_experts,
            )
            policies = {
                "static": static_rankings[(head.phase, target_layer)],
                "domain_oracle": domain_rankings[
                    (head.phase, token.domain, target_layer)
                ],
                "transition": transition,
                "linear": tuple(int(value) for value in linear_rankings[row_index]),
            }
            for capacity in capacities:
                for baseline, ranking in policies.items():
                    key = (
                        head.phase,
                        token.domain,
                        head.source_layer,
                        head.delta,
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
                    accumulator.add(tuple(ranking[:capacity]), target)

    metric_rows: list[dict[str, Any]] = []
    for key, accumulator in sorted(accumulators.items()):
        phase, domain, source_layer, delta, capacity, baseline = key
        metric_rows.append(
            {
                "phase": phase,
                "domain": domain,
                "source_layer": source_layer,
                "target_layer": source_layer + delta,
                "delta": delta,
                "capacity": capacity,
                "baseline": baseline,
                **accumulator.metrics(),
            }
        )
    summaries = _summarize_metrics(metric_rows)
    gate = _evaluate_gate(metric_rows, experiment_config["decision_gate"])
    transition_reproduction = _compare_h2_transition(
        metric_rows,
        reference_split_path.parent / "metrics.csv",
    )
    integrity = {
        "features": feature_integrity,
        "split_reproduction": "pass",
        "routing_top_k": routing_top_k,
        "router_layers": len(layers),
        "h2_transition_reproduction": transition_reproduction,
    }

    _write_csv(analysis_dir / "metrics.csv", metric_rows)
    _write_csv(analysis_dir / "summary.csv", summaries)
    write_json(analysis_dir / "gate.json", gate)
    write_json(analysis_dir / "integrity.json", integrity)
    predictor_manifest = {
        "format": "numpy_npz_numeric_arrays_no_pickle",
        "path": str(predictor_path),
        "sha256": predictor_sha256,
        "head_count": len(heads),
        "feature_projection": projection_report,
        "training_config": predictor_config,
    }
    write_json(analysis_dir / "predictor_manifest.json", predictor_manifest)
    _write_report(
        analysis_dir / "REPORT.md",
        run_id=str(manifest["run_id"]),
        gate=gate,
        summaries=summaries,
        integrity=integrity,
    )
    summary = {
        "run_id": manifest["run_id"],
        "analysis_id": analysis_config["analysis_id"],
        "evidence_grade": "pilot",
        "train_requests": len(train_ids),
        "test_requests": len(test_ids),
        "token_routes": len(tokens),
        "feature_rows": feature_integrity["feature_rows"],
        "linear_heads": len(heads),
        "gate": gate,
        "integrity": integrity,
        "outputs": {
            "split": str(analysis_dir / "split.json"),
            "training": str(analysis_dir / "training.csv"),
            "metrics": str(analysis_dir / "metrics.csv"),
            "summary": str(analysis_dir / "summary.csv"),
            "gate": str(analysis_dir / "gate.json"),
            "integrity": str(analysis_dir / "integrity.json"),
            "predictors": str(predictor_path),
            "report": str(analysis_dir / "REPORT.md"),
        },
    }
    write_json(analysis_dir / "summary.json", summary)
    return summary
