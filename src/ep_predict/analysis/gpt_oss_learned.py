from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ep_predict.analysis.gpt_oss_prediction import TokenRoutes, _load_routes
from ep_predict.tracing.storage import write_json

PHASE_TO_INDEX = {"prefill": 0, "decode": 1}
BASELINES = ("global_static", "domain_static", "source_copy", "transition", "learned")
METRICS = ("selection_coverage", "routed_mass_coverage", "complete_route_coverage")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class CompactRoutes:
    route_ids: torch.Tensor
    route_weights: torch.Tensor
    phase_ids: torch.Tensor
    request_ids: np.ndarray
    sample_ids: tuple[str, ...]
    domains: tuple[str, ...]
    phases: tuple[str, ...]
    token_positions: np.ndarray

    @property
    def tokens(self) -> int:
        return int(self.route_ids.shape[0])


def compact_routes(tokens: list[TokenRoutes], request_ids: set[int], layers: int) -> CompactRoutes:
    selected = [token for token in tokens if token.request_id in request_ids]
    route_ids = torch.empty((len(selected), layers, 4), dtype=torch.int64)
    route_weights = torch.empty((len(selected), layers, 4), dtype=torch.float32)
    for token_index, token in enumerate(selected):
        if set(token.routes) != set(range(layers)):
            raise ValueError(
                f"incomplete route for request={token.request_id}, phase={token.phase}, "
                f"position={token.token_position}"
            )
        for layer in range(layers):
            route = token.routes[layer]
            if len(route.expert_ids) != 4 or len(route.weights) != 4:
                raise ValueError("Milestone F requires complete top-4 routes")
            route_ids[token_index, layer] = torch.tensor(route.expert_ids)
            route_weights[token_index, layer] = torch.tensor(route.weights)
    return CompactRoutes(
        route_ids=route_ids,
        route_weights=route_weights,
        phase_ids=torch.tensor([PHASE_TO_INDEX[token.phase] for token in selected]),
        request_ids=np.asarray([token.request_id for token in selected], dtype=np.int32),
        sample_ids=tuple(token.sample_id for token in selected),
        domains=tuple(token.domain for token in selected),
        phases=tuple(token.phase for token in selected),
        token_positions=np.asarray([token.token_position for token in selected], dtype=np.int32),
    )


def layer_pairs(layers: int, lookaheads: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    allowed = set(lookaheads)
    pairs = [
        (source, target)
        for source in range(layers)
        for target in range(source + 1, layers)
        if target - source in allowed
    ]
    return (
        torch.tensor([pair[0] for pair in pairs], dtype=torch.int64),
        torch.tensor([pair[1] for pair in pairs], dtype=torch.int64),
    )


class SharedRouteMLP(nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        experts: int,
        source_embedding_width: int,
        target_embedding_width: int,
        phase_embedding_width: int,
        hidden_width: int,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.experts = experts
        self.source_layer_embedding = nn.Embedding(layers, source_embedding_width)
        self.target_layer_embedding = nn.Embedding(layers, target_embedding_width)
        self.phase_embedding = nn.Embedding(2, phase_embedding_width)
        input_width = (
            experts
            + source_embedding_width
            + target_embedding_width
            + phase_embedding_width
        )
        self.input = nn.Linear(input_width, hidden_width)
        self.activation = nn.GELU(approximate="none")
        self.output = nn.Linear(hidden_width, experts)

    def forward(
        self,
        route: torch.Tensor,
        source_layer: torch.Tensor,
        target_layer: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                route,
                self.source_layer_embedding(source_layer),
                self.target_layer_embedding(target_layer),
                self.phase_embedding(phase),
            ),
            dim=1,
        )
        return self.output(self.activation(self.input(features)))


def _dense_routes(
    route_ids: torch.Tensor, route_weights: torch.Tensor, experts: int
) -> torch.Tensor:
    dense = torch.zeros((route_ids.shape[0], experts), dtype=torch.float32)
    dense.scatter_(1, route_ids, route_weights)
    return dense


def _targets(route_ids: torch.Tensor, experts: int) -> torch.Tensor:
    labels = torch.zeros((route_ids.shape[0], experts), dtype=torch.float32)
    labels.scatter_(1, route_ids, 1.0)
    return labels


def _model_from_config(config: dict[str, Any]) -> SharedRouteMLP:
    geometry = config["geometry"]
    model = config["model"]
    return SharedRouteMLP(
        layers=int(geometry["layers"]),
        experts=int(geometry["experts"]),
        source_embedding_width=int(model["source_layer_embedding_width"]),
        target_embedding_width=int(model["target_layer_embedding_width"]),
        phase_embedding_width=int(model["phase_embedding_width"]),
        hidden_width=int(model["hidden_width"]),
    )


def train_model(
    data: CompactRoutes, config: dict[str, Any]
) -> tuple[SharedRouteMLP, list[dict[str, Any]]]:
    geometry = config["geometry"]
    training = config["training"]
    experts = int(geometry["experts"])
    source_pairs, target_pairs = layer_pairs(
        int(geometry["layers"]), [int(value) for value in training["fit_lookaheads"]]
    )
    pairs_per_token = len(source_pairs)
    examples = data.tokens * pairs_per_token
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(training["deterministic_algorithms"]))
    model = _model_from_config(config).to(config["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(training["positive_weight"]), device=config["device"])
    )
    generator = torch.Generator(device="cpu").manual_seed(int(training["shuffle_seed"]))
    batch_size = int(training["batch_size"])
    history: list[dict[str, Any]] = []
    device = torch.device(config["device"])
    model.train()
    for epoch in range(1, int(training["epochs"]) + 1):
        started = time.monotonic()
        order = torch.randperm(examples, generator=generator)
        loss_sum = 0.0
        for offset in range(0, examples, batch_size):
            flat = order[offset : offset + batch_size]
            token_index = torch.div(flat, pairs_per_token, rounding_mode="floor")
            pair_index = flat.remainder(pairs_per_token)
            source = source_pairs[pair_index]
            target = target_pairs[pair_index]
            source_ids = data.route_ids[token_index, source]
            source_weights = data.route_weights[token_index, source]
            labels = _targets(data.route_ids[token_index, target], experts)
            route = _dense_routes(source_ids, source_weights, experts)
            phase = data.phase_ids[token_index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                route.to(device),
                source.to(device),
                target.to(device),
                phase.to(device),
            )
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(flat)
        history.append(
            {
                "epoch": epoch,
                "examples": examples,
                "mean_training_loss": loss_sum / examples,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    return model, history


@torch.inference_mode()
def predict_scores(
    model: SharedRouteMLP, data: CompactRoutes, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geometry = config["geometry"]
    evaluation = config["evaluation"]
    experts = int(geometry["experts"])
    source_pairs, target_pairs = layer_pairs(
        int(geometry["layers"]), [int(value) for value in evaluation["lookaheads"]]
    )
    pairs_per_token = len(source_pairs)
    examples = data.tokens * pairs_per_token
    scores = np.empty((examples, experts), dtype=np.float32)
    max_candidates = max(int(value) for value in evaluation["candidate_counts"])
    ranks = np.empty((examples, max_candidates), dtype=np.uint8)
    batch_size = int(evaluation["inference_batch_size"])
    device = torch.device(config["device"])
    model.eval()
    for offset in range(0, examples, batch_size):
        flat = torch.arange(offset, min(offset + batch_size, examples))
        token_index = torch.div(flat, pairs_per_token, rounding_mode="floor")
        pair_index = flat.remainder(pairs_per_token)
        source = source_pairs[pair_index]
        target = target_pairs[pair_index]
        route = _dense_routes(
            data.route_ids[token_index, source], data.route_weights[token_index, source], experts
        )
        logits = model(
            route.to(device), source.to(device), target.to(device), data.phase_ids[token_index].to(device)
        )
        probabilities = torch.sigmoid(logits).cpu().numpy()
        scores[offset : offset + len(flat)] = probabilities
        ranks[offset : offset + len(flat)] = np.argsort(
            -probabilities, axis=1, kind="stable"
        )[:, :max_candidates]
    return scores, ranks, source_pairs.numpy(), target_pairs.numpy()


def model_accounting(model: SharedRouteMLP, config: dict[str, Any]) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    bytes_ = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    widths = config["model"]
    input_width = (
        int(widths["route_width"])
        + int(widths["source_layer_embedding_width"])
        + int(widths["target_layer_embedding_width"])
        + int(widths["phase_embedding_width"])
    )
    hidden = int(widths["hidden_width"])
    output = int(widths["output_width"])
    return {
        "parameters": parameters,
        "serialized_parameter_bytes": bytes_,
        "parameter_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "multiply_accumulates_per_forecast": input_width * hidden + hidden * output,
        "linear_bias_additions_per_forecast": hidden + output,
        "gelu_evaluations_per_forecast": hidden,
        "embedding_lookups_per_forecast": 3,
        "input_width": input_width,
        "hidden_width": hidden,
        "output_width": output,
    }


def _fit_baselines(
    data: CompactRoutes, layers: int, experts: int, domains: list[str]
) -> dict[str, np.ndarray]:
    domain_to_index = {domain: index for index, domain in enumerate(domains)}
    phase = data.phase_ids.numpy()
    domain = np.asarray([domain_to_index[value] for value in data.domains], dtype=np.int64)
    ids = data.route_ids.numpy()
    marginals = np.zeros((2, layers, experts), dtype=np.int64)
    domain_marginals = np.zeros((2, len(domains), layers, experts), dtype=np.int64)
    for layer in range(layers):
        for slot in range(4):
            np.add.at(marginals, (phase, layer, ids[:, layer, slot]), 1)
            np.add.at(
                domain_marginals,
                (phase, domain, layer, ids[:, layer, slot]),
                1,
            )
    transitions = np.zeros((2, layers, layers, experts, experts), dtype=np.int32)
    for source in range(layers):
        source_ids = ids[:, source]
        for target in range(source + 1, layers):
            target_ids = ids[:, target]
            table = transitions[:, source, target]
            for source_slot in range(4):
                for target_slot in range(4):
                    np.add.at(
                        table,
                        (
                            phase,
                            source_ids[:, source_slot],
                            target_ids[:, target_slot],
                        ),
                        1,
                    )
    return {
        "marginals": marginals,
        "domain_marginals": domain_marginals,
        "transitions": transitions,
    }


def _static_ranks(counts: np.ndarray) -> np.ndarray:
    return np.argsort(-counts, axis=-1, kind="stable").astype(np.uint8)


def _baseline_ranks(
    baseline: str,
    data: CompactRoutes,
    metadata: dict[str, np.ndarray],
    fitted: dict[str, np.ndarray],
    max_candidates: int,
    batch_size: int = 65536,
) -> np.ndarray:
    phase = metadata["phase"]
    source = metadata["source"]
    target = metadata["target"]
    token = metadata["token"]
    domain = metadata["domain"]
    marginal_ranks = _static_ranks(fitted["marginals"])
    if baseline == "global_static":
        return marginal_ranks[phase, target, :max_candidates]
    if baseline == "domain_static":
        ranks = _static_ranks(fitted["domain_marginals"])
        return ranks[phase, domain, target, :max_candidates]

    result = np.empty((len(token), max_candidates), dtype=np.uint8)
    route_ids = data.route_ids.numpy()
    for offset in range(0, len(token), batch_size):
        stop = min(offset + batch_size, len(token))
        selection = slice(offset, stop)
        current_ids = route_ids[token[selection], source[selection]]
        if baseline == "source_copy":
            # Integer rank scores reproduce source order first, followed by the
            # target-layer marginal ordering with expert-ID tie breaking.
            static_order = marginal_ranks[phase[selection], target[selection]]
            ordinal = np.empty_like(static_order, dtype=np.int16)
            rows = np.arange(stop - offset)[:, None]
            ordinal[rows, static_order] = np.arange(32, 0, -1, dtype=np.int16)
            ordinal[rows, current_ids] = np.asarray([100, 99, 98, 97], dtype=np.int16)
            result[selection] = np.argsort(-ordinal, axis=1, kind="stable")[:, :max_candidates]
            continue
        if baseline != "transition":
            raise ValueError(f"unknown baseline: {baseline}")
        scores = np.zeros((stop - offset, 32), dtype=np.float64)
        marginal = fitted["marginals"][phase[selection], target[selection]].astype(np.float64)
        marginal_total = marginal.sum(axis=1, keepdims=True)
        marginal_distribution = np.divide(
            marginal,
            marginal_total,
            out=np.zeros_like(marginal),
            where=marginal_total != 0,
        )
        for slot in range(4):
            rows_ = fitted["transitions"][
                phase[selection],
                source[selection],
                target[selection],
                current_ids[:, slot],
            ].astype(np.float64)
            totals = rows_.sum(axis=1, keepdims=True)
            distribution = np.divide(
                rows_, totals, out=np.zeros_like(rows_), where=totals != 0
            )
            empty = totals[:, 0] == 0
            distribution[empty] = marginal_distribution[empty]
            scores += distribution
        scores /= 4
        result[selection] = np.argsort(-scores, axis=1, kind="stable")[:, :max_candidates]
    return result


def _evaluation_metadata(
    data: CompactRoutes,
    source_pairs: np.ndarray,
    target_pairs: np.ndarray,
    domains: list[str],
) -> dict[str, np.ndarray]:
    pairs = len(source_pairs)
    token = np.repeat(np.arange(data.tokens, dtype=np.int32), pairs)
    source = np.tile(source_pairs.astype(np.int16), data.tokens)
    target = np.tile(target_pairs.astype(np.int16), data.tokens)
    phase_per_token = data.phase_ids.numpy().astype(np.int8)
    phase = phase_per_token[token]
    domain_to_index = {domain: index for index, domain in enumerate(domains)}
    domain_per_token = np.asarray(
        [domain_to_index[value] for value in data.domains], dtype=np.int8
    )
    request_values = sorted({int(value) for value in data.request_ids})
    request_to_index = {request_id: index for index, request_id in enumerate(request_values)}
    request_per_token = np.asarray(
        [request_to_index[int(value)] for value in data.request_ids], dtype=np.int16
    )
    return {
        "token": token,
        "pair": np.tile(np.arange(pairs, dtype=np.int16), data.tokens),
        "source": source,
        "target": target,
        "delta": (target - source).astype(np.int8),
        "phase": phase,
        "domain": domain_per_token[token],
        "request": request_per_token[token],
        "request_values": np.asarray(request_values, dtype=np.int32),
    }


def _coverage_arrays(
    ranks: np.ndarray,
    target_ids: np.ndarray,
    target_weights: np.ndarray,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = ranks[:, :candidate_count]
    hits = (target_ids[:, :, None] == candidates[:, None, :]).any(axis=2)
    selection = hits.sum(axis=1).astype(np.float64)
    mass = (hits * target_weights).sum(axis=1, dtype=np.float64)
    complete = hits.all(axis=1).astype(np.float64)
    return hits, selection, mass, complete


def _aggregate_fixed_k(
    *,
    baseline: str,
    candidate_count: int,
    ranks: np.ndarray,
    target_ids: np.ndarray,
    target_weights: np.ndarray,
    metadata: dict[str, np.ndarray],
    data: CompactRoutes,
    domains: list[str],
    source_pairs: np.ndarray,
    target_pairs: np.ndarray,
    brier: np.ndarray | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, selections, masses, completes = _coverage_arrays(
        ranks, target_ids, target_weights, candidate_count
    )
    pair_count = len(source_pairs)
    scope_code = (
        (metadata["phase"].astype(np.int64) * len(domains) + metadata["domain"])
        * pair_count
        + metadata["pair"]
    )
    scope_size = 2 * len(domains) * pair_count
    scope_n = np.bincount(scope_code, minlength=scope_size)
    scope_selection = np.bincount(scope_code, weights=selections, minlength=scope_size)
    scope_mass = np.bincount(scope_code, weights=masses, minlength=scope_size)
    scope_complete = np.bincount(scope_code, weights=completes, minlength=scope_size)
    scope_brier = (
        np.bincount(scope_code, weights=brier, minlength=scope_size)
        if brier is not None
        else None
    )
    scope_rows: list[dict[str, Any]] = []
    for code in np.flatnonzero(scope_n):
        pair = int(code % pair_count)
        outer = int(code // pair_count)
        domain_index = outer % len(domains)
        phase_index = outer // len(domains)
        count = int(scope_n[code])
        covered = float(scope_selection[code])
        selection_coverage = covered / (count * 4)
        row: dict[str, Any] = {
            "phase": ("prefill", "decode")[phase_index],
            "domain": domains[domain_index],
            "source_layer": int(source_pairs[pair]),
            "target_layer": int(target_pairs[pair]),
            "delta": int(target_pairs[pair] - source_pairs[pair]),
            "candidate_count": candidate_count,
            "baseline": baseline,
            "n_tokens": count,
            "selection_coverage": selection_coverage,
            "candidate_precision": covered / (count * candidate_count),
            "routed_mass_coverage": float(scope_mass[code]) / count,
            "complete_route_coverage": float(scope_complete[code]) / count,
            "candidate_amplification": candidate_count / 4,
            "candidate_set_fraction": candidate_count / 32,
            "useful_candidate_amplification": (
                candidate_count / (covered / count) if covered else math.inf
            ),
        }
        if scope_brier is not None:
            row["brier_score"] = float(scope_brier[code]) / count
        scope_rows.append(row)

    # Request scopes pool eligible source-target cells within each request and
    # lookahead. Subsequent summaries average requests within domain, then domains.
    request_count = len(metadata["request_values"])
    request_code = (
        (metadata["request"].astype(np.int64) * 2 + metadata["phase"]) * 24
        + metadata["delta"]
    )
    request_size = request_count * 2 * 24
    request_n = np.bincount(request_code, minlength=request_size)
    request_selection = np.bincount(request_code, weights=selections, minlength=request_size)
    request_mass = np.bincount(request_code, weights=masses, minlength=request_size)
    request_complete = np.bincount(request_code, weights=completes, minlength=request_size)
    request_brier = (
        np.bincount(request_code, weights=brier, minlength=request_size)
        if brier is not None
        else None
    )
    request_rows: list[dict[str, Any]] = []
    for code in np.flatnonzero(request_n):
        delta = int(code % 24)
        outer = int(code // 24)
        phase_index = outer % 2
        request_index = outer // 2
        count = int(request_n[code])
        covered = float(request_selection[code])
        token_indexes = np.flatnonzero(
            (np.asarray([int(value) for value in data.request_ids]) == metadata["request_values"][request_index])
            & (data.phase_ids.numpy() == phase_index)
        )
        first = int(token_indexes[0])
        row = {
            "request_id": int(metadata["request_values"][request_index]),
            "sample_id": data.sample_ids[first],
            "domain": data.domains[first],
            "phase": ("prefill", "decode")[phase_index],
            "delta": delta,
            "candidate_count": candidate_count,
            "baseline": baseline,
            "n_tokens": count,
            "selection_coverage": covered / (count * 4),
            "candidate_precision": covered / (count * candidate_count),
            "routed_mass_coverage": float(request_mass[code]) / count,
            "complete_route_coverage": float(request_complete[code]) / count,
            "candidate_amplification": candidate_count / 4,
            "candidate_set_fraction": candidate_count / 32,
            "useful_candidate_amplification": (
                candidate_count / (covered / count) if covered else math.inf
            ),
        }
        if request_brier is not None:
            row["brier_score"] = float(request_brier[code]) / count
        request_rows.append(row)
    return scope_rows, request_rows


def _domain_balanced(rows: list[dict[str, Any]], metric: str) -> float:
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(float(row[metric]))
    if not by_domain:
        raise ValueError(f"empty domain-balanced scope for {metric}")
    return float(np.mean([np.mean(values) for values in by_domain.values()]))


def horizon_summaries(request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        grouped[
            (
                str(row["phase"]),
                int(row["delta"]),
                int(row["candidate_count"]),
                str(row["baseline"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (phase, delta, candidates, baseline), rows in sorted(grouped.items()):
        summary = {
            "phase": phase,
            "delta": delta,
            "candidate_count": candidates,
            "baseline": baseline,
            "requests": len(rows),
            "domains": len({str(row["domain"]) for row in rows}),
            "selection_coverage": _domain_balanced(rows, "selection_coverage"),
            "candidate_precision": _domain_balanced(rows, "candidate_precision"),
            "routed_mass_coverage": _domain_balanced(rows, "routed_mass_coverage"),
            "complete_route_coverage": _domain_balanced(rows, "complete_route_coverage"),
            "candidate_amplification": candidates / 4,
            "candidate_set_fraction": candidates / 32,
            "useful_candidate_amplification": _domain_balanced(
                rows, "useful_candidate_amplification"
            ),
        }
        learned_brier = [row for row in rows if "brier_score" in row]
        if learned_brier:
            summary["brier_score"] = _domain_balanced(learned_brier, "brier_score")
        summaries.append(summary)
    return summaries


def calibration_tables(
    scores: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    domains: list[str],
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reliability: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for phase_index, phase_name in enumerate(("prefill", "decode")):
        for domain_index, domain_name in enumerate(domains):
            for delta in range(1, 24):
                example_mask = (
                    (metadata["phase"] == phase_index)
                    & (metadata["domain"] == domain_index)
                    & (metadata["delta"] == delta)
                )
                probabilities = scores[example_mask].reshape(-1)
                outcomes = labels[example_mask].reshape(-1)
                if not len(probabilities):
                    continue
                bin_ids = np.minimum(
                    np.searchsorted(edges, probabilities, side="right") - 1,
                    bins - 1,
                )
                absolute_error = 0.0
                for bin_index in range(bins):
                    mask = bin_ids == bin_index
                    count = int(mask.sum())
                    if not count:
                        continue
                    mean_score = float(probabilities[mask].mean())
                    frequency = float(outcomes[mask].mean())
                    absolute_error += count * abs(mean_score - frequency)
                    reliability.append(
                        {
                            "phase": phase_name,
                            "domain": domain_name,
                            "delta": delta,
                            "bin": bin_index,
                            "bin_low": edges[bin_index],
                            "bin_high": edges[bin_index + 1],
                            "predictions": count,
                            "mean_score": mean_score,
                            "empirical_frequency": frequency,
                        }
                    )
                summaries.append(
                    {
                        "phase": phase_name,
                        "domain": domain_name,
                        "delta": delta,
                        "predictions": len(probabilities),
                        "positive_frequency": float(outcomes.mean()),
                        "mean_score": float(probabilities.mean()),
                        "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
                        "expected_calibration_error": absolute_error / len(probabilities),
                    }
                )
    return reliability, summaries


def threshold_frontier(
    scores: np.ndarray,
    labels: np.ndarray,
    target_ids: np.ndarray,
    target_weights: np.ndarray,
    metadata: dict[str, np.ndarray],
    data: CompactRoutes,
    domains: list[str],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    request_count = len(metadata["request_values"])
    request_code = (
        (metadata["request"].astype(np.int64) * 2 + metadata["phase"]) * 24
        + metadata["delta"]
    )
    size = request_count * 2 * 24
    request_domain: dict[int, str] = {}
    for request_id, domain in zip(data.request_ids, data.domains, strict=True):
        request_domain[int(request_id)] = domain
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = scores >= threshold
        candidates = selected.sum(axis=1).astype(np.float64)
        hits = (selected & labels).sum(axis=1).astype(np.float64)
        masses = np.zeros(len(scores), dtype=np.float64)
        target_selected = np.take_along_axis(selected, target_ids, axis=1)
        masses[:] = (target_selected * target_weights).sum(axis=1)
        completes = target_selected.all(axis=1).astype(np.float64)
        count = np.bincount(request_code, minlength=size)
        candidate_sum = np.bincount(request_code, weights=candidates, minlength=size)
        hit_sum = np.bincount(request_code, weights=hits, minlength=size)
        mass_sum = np.bincount(request_code, weights=masses, minlength=size)
        complete_sum = np.bincount(request_code, weights=completes, minlength=size)
        request_rows: list[dict[str, Any]] = []
        for code in np.flatnonzero(count):
            delta = int(code % 24)
            outer = int(code // 24)
            phase_index = outer % 2
            request_index = outer // 2
            request_id = int(metadata["request_values"][request_index])
            n = int(count[code])
            request_rows.append(
                {
                    "request_id": request_id,
                    "domain": request_domain[request_id],
                    "phase": ("prefill", "decode")[phase_index],
                    "delta": delta,
                    "mean_candidate_count": float(candidate_sum[code]) / n,
                    "selection_coverage": float(hit_sum[code]) / (n * 4),
                    "candidate_precision": (
                        float(hit_sum[code]) / float(candidate_sum[code])
                        if candidate_sum[code]
                        else 0.0
                    ),
                    "routed_mass_coverage": float(mass_sum[code]) / n,
                    "complete_route_coverage": float(complete_sum[code]) / n,
                }
            )
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in request_rows:
            grouped[(str(row["phase"]), int(row["delta"]))].append(row)
        for (phase, delta), group in sorted(grouped.items()):
            mean_candidates = _domain_balanced(group, "mean_candidate_count")
            rows.append(
                {
                    "phase": phase,
                    "delta": delta,
                    "score_threshold": threshold,
                    "selection_coverage": _domain_balanced(group, "selection_coverage"),
                    "candidate_precision": _domain_balanced(group, "candidate_precision"),
                    "routed_mass_coverage": _domain_balanced(group, "routed_mass_coverage"),
                    "complete_route_coverage": _domain_balanced(
                        group, "complete_route_coverage"
                    ),
                    "mean_candidate_count": mean_candidates,
                    "candidate_amplification": mean_candidates / 4,
                    "candidate_set_fraction": mean_candidates / 32,
                }
            )
    return rows


def churn_summary(
    ranks: np.ndarray,
    metadata: dict[str, np.ndarray],
    data: CompactRoutes,
    domains: list[str],
    candidate_counts: list[int],
) -> list[dict[str, Any]]:
    pair_lookup = {
        (int(source), int(target)): pair
        for pair, (source, target) in enumerate(
            zip(metadata["source"][:276], metadata["target"][:276], strict=True)
        )
    }
    pairs = 276
    reshaped = ranks.reshape(data.tokens, pairs, ranks.shape[1])
    grouped: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    for token_index in range(data.tokens):
        for target in range(2, 24):
            for later_source in range(1, target):
                earlier = reshaped[token_index, pair_lookup[(later_source - 1, target)]]
                later = reshaped[token_index, pair_lookup[(later_source, target)]]
                for candidates in candidate_counts:
                    overlap = len(set(earlier[:candidates]) & set(later[:candidates]))
                    grouped[
                        (
                            data.phases[token_index],
                            data.domains[token_index],
                            target,
                            candidates,
                        )
                    ].append(1 - overlap / candidates)
    rows: list[dict[str, Any]] = []
    outer: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for (phase, domain, target, candidates), values in grouped.items():
        outer[(phase, target, candidates)].append(
            {"domain": domain, "replacement_fraction": float(np.mean(values))}
        )
    for (phase, target, candidates), values in sorted(outer.items()):
        rows.append(
            {
                "phase": phase,
                "target_layer": target,
                "candidate_count": candidates,
                "candidate_replacement_fraction": _domain_balanced(
                    values, "replacement_fraction"
                ),
                "candidate_retention_fraction": 1
                - _domain_balanced(values, "replacement_fraction"),
                "comparison": "source_layer_l_minus_1_vs_l_for_fixed_target",
            }
        )
    return rows


def development_gate(
    request_rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation = config["evaluation"]
    gate = config["decision_gate"]
    phase = str(evaluation["primary_phase"])
    candidates = int(evaluation["primary_candidate_count"])
    deltas = [int(value) for value in evaluation["primary_lookaheads"]]
    filtered = [
        row
        for row in request_rows
        if row["phase"] == phase and int(row["candidate_count"]) == candidates
    ]
    domains = sorted({str(row["domain"]) for row in filtered})
    request_ids = {
        domain: sorted(
            {int(row["request_id"]) for row in filtered if row["domain"] == domain}
        )
        for domain in domains
    }
    by_key = {
        (int(row["request_id"]), int(row["delta"]), str(row["baseline"])): row
        for row in filtered
    }
    rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    resamples = int(evaluation["bootstrap_resamples"])
    confidence = float(evaluation["confidence_level"])
    cheap_names = ("domain_static", "source_copy")
    rows: list[dict[str, Any]] = []
    for delta in deltas:
        model_rows = [
            row for row in filtered if row["delta"] == delta and row["baseline"] == "learned"
        ]
        transition_rows = [
            row
            for row in filtered
            if row["delta"] == delta and row["baseline"] == "transition"
        ]
        cheap_values = {
            name: _domain_balanced(
                [
                    row
                    for row in filtered
                    if row["delta"] == delta and row["baseline"] == name
                ],
                "selection_coverage",
            )
            for name in cheap_names
        }
        cheap = max(cheap_values, key=cheap_values.get)
        points: dict[str, float] = {}
        for metric in METRICS:
            learned_value = _domain_balanced(model_rows, metric)
            transition_value = _domain_balanced(transition_rows, metric)
            cheap_value = _domain_balanced(
                [
                    row
                    for row in filtered
                    if row["delta"] == delta and row["baseline"] == cheap
                ],
                metric,
            )
            points[f"learned_{metric}"] = learned_value
            points[f"transition_{metric}"] = transition_value
            points[f"cheap_{metric}"] = cheap_value
            points[f"learned_minus_transition_{metric}"] = learned_value - transition_value
            points[f"learned_minus_cheap_{metric}"] = learned_value - cheap_value
        domain_gains: dict[str, float] = {}
        for domain in domains:
            learned = np.mean(
                [
                    float(row["selection_coverage"])
                    for row in model_rows
                    if row["domain"] == domain
                ]
            )
            strongest = max(
                np.mean(
                    [
                        float(row["selection_coverage"])
                        for row in filtered
                        if row["delta"] == delta
                        and row["baseline"] == name
                        and row["domain"] == domain
                    ]
                )
                for name in cheap_names
            )
            domain_gains[domain] = float(learned - strongest)

        bootstrap: dict[str, list[float]] = defaultdict(list)
        for _ in range(resamples):
            sampled = {
                domain: rng.choice(ids, size=len(ids), replace=True).tolist()
                for domain, ids in request_ids.items()
            }
            for metric in METRICS:
                means: dict[str, float] = {}
                for baseline in ("learned", "transition", *cheap_names):
                    means[baseline] = float(
                        np.mean(
                            [
                                np.mean(
                                    [
                                        float(by_key[(request_id, delta, baseline)][metric])
                                        for request_id in sampled[domain]
                                    ]
                                )
                                for domain in domains
                            ]
                        )
                    )
                bootstrap[f"learned_minus_transition_{metric}"].append(
                    means["learned"] - means["transition"]
                )
                bootstrap[f"learned_minus_cheap_{metric}"].append(
                    means["learned"] - max(means[name] for name in cheap_names)
                )
        tail = (1 - confidence) / 2
        intervals: dict[str, float] = {}
        for name, values in bootstrap.items():
            intervals[f"{name}_ci_low"] = float(np.quantile(values, tail))
            intervals[f"{name}_ci_high"] = float(np.quantile(values, 1 - tail))

        checks = {
            "absolute_selection": points["learned_selection_coverage"]
            >= float(gate["min_selection_coverage"]) - 1e-12,
            "absolute_complete": points["learned_complete_route_coverage"]
            >= float(gate["min_complete_route_coverage"]) - 1e-12,
            "transition_selection_noninferiority": points[
                "learned_minus_transition_selection_coverage"
            ]
            >= -float(gate["max_selection_deficit_vs_transition"]) - 1e-12,
            "transition_complete_noninferiority": points[
                "learned_minus_transition_complete_route_coverage"
            ]
            >= -float(gate["max_complete_route_deficit_vs_transition"]) - 1e-12,
            "cheap_selection_gain": points["learned_minus_cheap_selection_coverage"]
            >= float(gate["min_selection_gain_vs_cheap_comparator"]) - 1e-12,
            "all_domain_selection_gains_positive": all(
                value > 0 for value in domain_gains.values()
            ),
        }
        rows.append(
            {
                "delta": delta,
                "cheap_comparator": cheap,
                **points,
                **intervals,
                "domain_selection_gains_json": json.dumps(
                    domain_gains, sort_keys=True, separators=(",", ":")
                ),
                **{f"check_{name}": passed for name, passed in checks.items()},
                "pass": all(checks.values()),
            }
        )
    passing = sum(bool(row["pass"]) for row in rows)
    required = int(gate["min_passing_lookaheads"])
    decision = {
        "milestone": "F",
        "evidence_stage": "development",
        "decision": "DEVELOPMENT_PASS" if passing >= required else "DEVELOPMENT_FAIL",
        "primary_phase": phase,
        "primary_candidate_count": candidates,
        "candidate_amplification": candidates / 4,
        "candidate_set_fraction": candidates / 32,
        "passing_lookaheads": passing,
        "required_passing_lookaheads": required,
        "bootstrap_resamples": resamples,
        "confidence_level": confidence,
        "gate_config": gate,
        "lookaheads": rows,
        "fresh_confirmation_authorized": passing >= required,
        "claim_boundary": (
            "development evidence on 32 previously inspected requests; a learned-model "
            "paper claim requires the unchanged gate on 64 previously unused requests"
        ),
    }
    return rows, decision


def _verify_baseline_reproduction(
    request_rows: list[dict[str, Any]], source_run: Path
) -> dict[str, Any]:
    prior_path = source_run / "analysis" / "prediction" / "request_metrics.csv"
    with prior_path.open(newline="", encoding="utf-8") as handle:
        prior = list(csv.DictReader(handle))
    key_fields = ("request_id", "phase", "delta", "candidate_count", "baseline")
    current = {
        tuple(str(row[field]) for field in key_fields): row
        for row in request_rows
        if int(row["candidate_count"]) == 8 and row["baseline"] != "learned"
    }
    maximum = 0.0
    compared = 0
    missing: list[tuple[str, ...]] = []
    for row in prior:
        key = tuple(str(row[field]) for field in key_fields)
        candidate = current.get(key)
        if candidate is None:
            missing.append(key)
            continue
        for metric in METRICS:
            maximum = max(maximum, abs(float(row[metric]) - float(candidate[metric])))
            compared += 1
    return {
        "reference": str(prior_path),
        "reference_sha256": sha256_file(prior_path),
        "metric_values_compared": compared,
        "missing_rows": len(missing),
        "max_abs_metric_difference": maximum,
        "exact_within_1e_12": not missing and maximum <= 1e-12,
    }


def _plot_results(
    summaries: list[dict[str, Any]], thresholds: list[dict[str, Any]], output_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "global_static": "#999999",
        "domain_static": "#cc78bc",
        "source_copy": "#dd8452",
        "transition": "#4c72b0",
        "learned": "#55a868",
    }
    labels = {
        "global_static": "Global popularity",
        "domain_static": "Domain popularity",
        "source_copy": "Route copy",
        "transition": "Transition",
        "learned": "Shared route MLP",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharex=True, sharey=True)
    for axis, delta in zip(axes, (1, 2, 3), strict=True):
        for baseline in BASELINES:
            rows = sorted(
                [
                    row
                    for row in summaries
                    if row["phase"] == "decode"
                    and int(row["delta"]) == delta
                    and row["baseline"] == baseline
                ],
                key=lambda row: int(row["candidate_count"]),
            )
            axis.plot(
                [int(row["candidate_count"]) / 4 for row in rows],
                [100 * float(row["selection_coverage"]) for row in rows],
                marker="o",
                linewidth=1.7,
                color=colors[baseline],
                label=labels[baseline],
            )
        axis.axvline(2, color="black", linestyle=":", linewidth=0.8)
        axis.axhline(82, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"Decode Δ={delta}")
        axis.set_xlabel("Candidate amplification K/4")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Expert-selection coverage (%)")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
        frameon=False,
    )
    fig.suptitle("Milestone F fixed-budget frontier", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fixed_png = figure_dir / "fig1_fixed_k_frontier.png"
    fixed_pdf = figure_dir / "fig1_fixed_k_frontier.pdf"
    fig.savefig(fixed_png, dpi=450, bbox_inches="tight")
    fig.savefig(fixed_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharex=True, sharey=True)
    for axis, delta in zip(axes, (1, 2, 3), strict=True):
        rows = sorted(
            [
                row
                for row in thresholds
                if row["phase"] == "decode" and int(row["delta"]) == delta
            ],
            key=lambda row: float(row["mean_candidate_count"]),
        )
        axis.plot(
            [float(row["mean_candidate_count"]) / 4 for row in rows],
            [100 * float(row["selection_coverage"]) for row in rows],
            marker=".",
            color=colors["learned"],
            label="Selection coverage",
        )
        axis.plot(
            [float(row["mean_candidate_count"]) / 4 for row in rows],
            [100 * float(row["candidate_precision"]) for row in rows],
            marker=".",
            color="#c44e52",
            label="Candidate precision",
        )
        axis.axvline(2, color="black", linestyle=":", linewidth=0.8)
        axis.set_title(f"Decode Δ={delta}")
        axis.set_xlabel("Mean candidate amplification")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Metric (%)")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Milestone F score-threshold frontier", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    threshold_png = figure_dir / "fig2_score_threshold_frontier.png"
    threshold_pdf = figure_dir / "fig2_score_threshold_frontier.pdf"
    fig.savefig(threshold_png, dpi=450, bbox_inches="tight")
    fig.savefig(threshold_pdf, bbox_inches="tight")
    plt.close(fig)
    return [fixed_png, fixed_pdf, threshold_png, threshold_pdf]


def _report(
    decision: dict[str, Any], accounting: dict[str, Any], reproduction: dict[str, Any]
) -> str:
    lines = [
        "# GPT-OSS 20B Milestone F development result",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        (
            "The fixed shared route MLP was fit on 96 retained requests and evaluated "
            "once on the 32-request development split. This split had already been "
            "inspected during Milestone E, so this result is not confirmatory."
        ),
        "",
        "| Δ | Learned selection | vs transition | vs cheap | Learned complete | vs transition | Domains positive | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in decision["lookaheads"]:
        domain_gains = json.loads(row["domain_selection_gains_json"])
        lines.append(
            f"| {row['delta']} | {100 * row['learned_selection_coverage']:.1f}% | "
            f"{100 * row['learned_minus_transition_selection_coverage']:+.1f} pp | "
            f"{100 * row['learned_minus_cheap_selection_coverage']:+.1f} pp | "
            f"{100 * row['learned_complete_route_coverage']:.1f}% | "
            f"{100 * row['learned_minus_transition_complete_route_coverage']:+.1f} pp | "
            f"{sum(value > 0 for value in domain_gains.values())}/{len(domain_gains)} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                f"{decision['passing_lookaheads']}/{decision['required_passing_lookaheads']} "
                "required lookaheads passed the unchanged gate."
            ),
            "",
            "## Compactness and audit checks",
            "",
            f"- Parameters: {accounting['parameters']:,} ({accounting['serialized_parameter_bytes']:,} FP32 bytes).",
            f"- Forecast cost: {accounting['multiply_accumulates_per_forecast']:,} multiply-accumulates.",
            (
                f"- Milestone E baseline reproduction: {reproduction['metric_values_compared']:,} "
                f"metric values, maximum absolute difference "
                f"{reproduction['max_abs_metric_difference']:.3g}; "
                f"{'pass' if reproduction['exact_within_1e_12'] else 'FAIL'}."
            ),
            "",
            (
                "No cache state, cold-expert label, token text, domain label, hidden state, "
                "or development request entered model fitting. Fresh confirmation is required "
                "before this can support a confirmatory learned-predictor claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = {
        str(path): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    manifest = {"schema_version": 1, "durable_files": files}
    write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def run_development(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    output_dir = Path(config["output_dir"])
    source_run = Path(config["source_run_dir"])
    split_path = Path(config["development_split"])
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite or resume a frozen experiment directory: {output_dir}"
        )
    if sha256_file(source_run / "artifact_manifest.json") != str(
        config["source_artifact_manifest_sha256"]
    ):
        raise ValueError("Milestone E artifact manifest hash does not match frozen config")
    if sha256_file(split_path) != str(config["development_split_sha256"]):
        raise ValueError("development split hash does not match frozen config")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = {int(value) for value in split["train_request_ids"]}
    development_ids = {int(value) for value in split["test_request_ids"]}
    if len(train_ids) != 96 or len(development_ids) != 32 or train_ids & development_ids:
        raise ValueError("frozen request split is not the required disjoint 96/32 split")
    integrity = json.loads((source_run / "integrity.json").read_text(encoding="utf-8"))
    totals = integrity["totals"]
    integrity_config = config["integrity"]
    mismatch_fraction = int(totals["dispatch_weight_mismatches"]) / int(
        totals["dispatch_consumed_pairs"]
    )
    integrity_checks = {
        "exact_dispatch_ids": int(totals["dispatch_id_mismatches"]) == 0,
        "complete_token_layer_coverage": bool(
            integrity["gate_checks"]["complete_layer_token_coverage"]
        ),
        "legacy_weight_error_within_bf16_bound": float(
            totals["dispatch_max_abs_weight_error"]
        )
        <= float(integrity_config["legacy_max_abs_weight_error"]),
        "legacy_weight_mismatch_fraction_within_bound": mismatch_fraction
        <= float(integrity_config["legacy_max_mismatch_fraction"]),
        "dispatch_consumed_weights_retained": str(integrity_config["weight_source"])
        == "dispatch_consumed",
    }
    if not all(integrity_checks.values()):
        raise ValueError(f"source trace fails Milestone F integrity checks: {integrity_checks}")

    output_dir.mkdir(parents=True)
    frozen_config = output_dir / "frozen_config.toml"
    shutil.copyfile(config_file, frozen_config)
    run_definition = {
        "schema_version": 1,
        "milestone": "F",
        "evidence_stage": "development",
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "frozen_config_path": str(frozen_config),
        "source_run": str(source_run),
        "source_artifact_manifest_sha256": sha256_file(
            source_run / "artifact_manifest.json"
        ),
        "development_split": str(split_path),
        "development_split_sha256": sha256_file(split_path),
        "integrity_checks": integrity_checks,
        "legacy_weight_mismatch_fraction": mismatch_fraction,
        "environment": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "torch_hip": torch.version.hip,
            "device_count": torch.cuda.device_count(),
            "device_name": (
                torch.cuda.get_device_name(torch.device(config["device"]))
                if torch.cuda.is_available()
                else None
            ),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
        "git_commit_before_implementation_run": _git_commit(),
        "created_before_first_fit": True,
    }
    write_json(output_dir / "run_definition.json", run_definition)

    tokens, requests = _load_routes(source_run)
    if set(requests) != train_ids | development_ids:
        raise ValueError("trace request IDs do not exactly match frozen split")
    domains = sorted({metadata["domain"] for metadata in requests.values()})
    for request_set, expected_per_domain, name in (
        (train_ids, 24, "train"),
        (development_ids, 8, "development"),
    ):
        counts = Counter(requests[value]["domain"] for value in request_set)
        if set(counts.values()) != {expected_per_domain} or set(counts) != set(domains):
            raise ValueError(f"{name} request split is not domain balanced: {counts}")
    split_rows = sorted(split["requests"], key=lambda row: (row["split"], row["domain"], row["request_id"]))
    write_json(
        output_dir / "request_split.json",
        {
            "source": str(split_path),
            "source_sha256": sha256_file(split_path),
            "train_request_ids": sorted(train_ids),
            "development_request_ids": sorted(development_ids),
            "requests": split_rows,
        },
    )
    _write_csv(output_dir / "request_split.csv", split_rows)

    layers = int(config["geometry"]["layers"])
    experts = int(config["geometry"]["experts"])
    train_data = compact_routes(tokens, train_ids, layers)
    development_data = compact_routes(tokens, development_ids, layers)
    model, history = train_model(train_data, config)
    _write_csv(output_dir / "training_history.csv", history)
    checkpoint = output_dir / "model.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config_sha256": sha256_file(frozen_config),
            "checkpoint_rule": config["training"]["checkpoint_rule"],
            "epoch": int(config["training"]["epochs"]),
        },
        checkpoint,
    )
    accounting = model_accounting(model, config)
    accounting["checkpoint_file_bytes"] = checkpoint.stat().st_size
    accounting["checkpoint_sha256"] = sha256_file(checkpoint)
    write_json(output_dir / "model_accounting.json", accounting)

    scores, learned_ranks, source_pairs, target_pairs = predict_scores(
        model, development_data, config
    )
    metadata = _evaluation_metadata(
        development_data, source_pairs, target_pairs, domains
    )
    target_ids = development_data.route_ids.numpy()[
        metadata["token"], metadata["target"]
    ]
    target_weights = development_data.route_weights.numpy()[
        metadata["token"], metadata["target"]
    ]
    labels = np.zeros((len(scores), experts), dtype=bool)
    np.put_along_axis(labels, target_ids, True, axis=1)
    brier = np.mean((scores - labels) ** 2, axis=1)
    fitted = _fit_baselines(train_data, layers, experts, domains)

    scope_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    max_candidates = max(int(value) for value in config["evaluation"]["candidate_counts"])
    for baseline in BASELINES:
        print(json.dumps({"evaluating": baseline}), flush=True)
        ranks = (
            learned_ranks
            if baseline == "learned"
            else _baseline_ranks(
                baseline,
                development_data,
                metadata,
                fitted,
                max_candidates,
            )
        )
        for candidate_count in config["evaluation"]["candidate_counts"]:
            scopes, requests_ = _aggregate_fixed_k(
                baseline=baseline,
                candidate_count=int(candidate_count),
                ranks=ranks,
                target_ids=target_ids,
                target_weights=target_weights,
                metadata=metadata,
                data=development_data,
                domains=domains,
                source_pairs=source_pairs,
                target_pairs=target_pairs,
                brier=brier if baseline == "learned" else None,
            )
            scope_rows.extend(scopes)
            request_rows.extend(requests_)

    summaries = horizon_summaries(request_rows)
    reliability, calibration = calibration_tables(
        scores,
        labels,
        metadata,
        domains,
        int(config["evaluation"]["calibration_bins"]),
    )
    thresholds = threshold_frontier(
        scores,
        labels,
        target_ids,
        target_weights,
        metadata,
        development_data,
        domains,
        [float(value) for value in config["evaluation"]["score_thresholds"]],
    )
    churn = churn_summary(
        learned_ranks,
        metadata,
        development_data,
        domains,
        [int(value) for value in config["evaluation"]["candidate_counts"]],
    )
    gate_rows, decision = development_gate(request_rows, config)
    reproduction = _verify_baseline_reproduction(request_rows, source_run)
    decision["baseline_reproduction"] = reproduction
    decision["trace_integrity"] = integrity_checks
    if not reproduction["exact_within_1e_12"]:
        decision["decision"] = "INVALID_BASELINE_REPRODUCTION"
        decision["fresh_confirmation_authorized"] = False

    table_dir = output_dir / "analysis"
    _write_csv(table_dir / "scope_metrics.csv", scope_rows)
    _write_csv(table_dir / "request_metrics.csv", request_rows)
    _write_csv(table_dir / "horizon_summary.csv", summaries)
    _write_csv(table_dir / "bootstrap_gate.csv", gate_rows)
    _write_csv(table_dir / "calibration_reliability.csv", reliability)
    _write_csv(table_dir / "calibration_summary.csv", calibration)
    _write_csv(table_dir / "score_threshold_frontier.csv", thresholds)
    _write_csv(table_dir / "candidate_churn.csv", churn)
    write_json(table_dir / "decision.json", decision)
    write_json(table_dir / "baseline_reproduction.json", reproduction)
    _plot_results(summaries, thresholds, output_dir)
    report = _report(decision, accounting, reproduction)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    result = {
        "decision": decision,
        "model_accounting": accounting,
        "training": {
            "tokens": train_data.tokens,
            "examples_per_epoch": train_data.tokens * len(source_pairs),
            "epochs": len(history),
            "final_training_loss": history[-1]["mean_training_loss"],
        },
        "development": {
            "requests": len(development_ids),
            "tokens": development_data.tokens,
            "forecast_examples": len(scores),
        },
        "artifacts": {
            "report": str(output_dir / "REPORT.md"),
            "decision": str(table_dir / "decision.json"),
            "checkpoint": str(checkpoint),
        },
    }
    write_json(output_dir / "result.json", result)
    _artifact_manifest(output_dir)
    return result
