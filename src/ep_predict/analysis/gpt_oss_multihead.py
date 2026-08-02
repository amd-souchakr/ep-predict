from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ep_predict.analysis.gpt_oss_learned import (
    CompactRoutes,
    _artifact_manifest,
    _baseline_ranks,
    _evaluation_metadata,
    _fit_baselines,
    compact_routes,
    development_gate,
    layer_pairs,
    sha256_file,
)
from ep_predict.analysis.gpt_oss_prediction import _load_routes
from ep_predict.tracing.storage import write_json


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cross_validation_splits(
    requests: dict[int, dict[str, str]],
    eligible_ids: set[int],
    *,
    folds: int,
    validation_per_domain: int,
    ordering_seed: int,
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[int]] = defaultdict(list)
    for request_id in eligible_ids:
        by_domain[requests[request_id]["domain"]].append(request_id)
    expected = folds * validation_per_domain
    if set(map(len, by_domain.values())) != {expected}:
        raise ValueError(
            f"cross-validation requires {expected} requests per domain: "
            f"{ {key: len(value) for key, value in by_domain.items()} }"
        )
    ordered: dict[str, list[int]] = {}
    for domain, ids in by_domain.items():
        ordered[domain] = sorted(
            ids,
            key=lambda request_id: hashlib.sha256(
                f"{ordering_seed}:{domain}:{request_id}:{requests[request_id]['sample_id']}".encode()
            ).hexdigest(),
        )
    result: list[dict[str, Any]] = []
    for fold in range(folds):
        validation: list[int] = []
        pool: dict[str, list[int]] = {}
        for domain in sorted(ordered):
            start = fold * validation_per_domain
            stop = start + validation_per_domain
            domain_validation = ordered[domain][start:stop]
            validation.extend(domain_validation)
            pool[domain] = [
                request_id
                for request_id in ordered[domain]
                if request_id not in set(domain_validation)
            ]
        result.append(
            {
                "fold": fold,
                "validation_request_ids": sorted(validation),
                "training_pool_by_domain": pool,
            }
        )
    return result


def route_features(data: CompactRoutes, mode: str) -> torch.Tensor:
    ids = data.route_ids
    weights = data.route_weights
    weighted = torch.zeros((data.tokens, ids.shape[1], 32), dtype=torch.float32)
    weighted.scatter_(2, ids, weights)
    if mode == "weighted":
        return weighted
    binary = torch.zeros_like(weighted)
    binary.scatter_(2, ids, 1.0)
    if mode == "binary":
        return binary
    if mode == "weighted_binary":
        return torch.cat((weighted, binary), dim=2)
    raise ValueError(f"unknown feature mode: {mode}")


class PairwiseRouteHeads(nn.Module):
    def __init__(self, pairs: int, feature_width: int, experts: int = 32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(pairs, feature_width, experts))
        self.bias = nn.Parameter(torch.empty(pairs, experts))
        bound = 1 / math.sqrt(feature_width)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bpf,pfe->bpe", pair_features, self.weight) + self.bias


def _pair_inputs_and_targets(
    data: CompactRoutes, mode: str, source_pairs: torch.Tensor, target_pairs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    features = route_features(data, mode)[:, source_pairs]
    target_ids = data.route_ids[:, target_pairs]
    targets = torch.zeros(
        (data.tokens, len(source_pairs), 32), dtype=torch.float32
    )
    targets.scatter_(2, target_ids, 1.0)
    return features, targets


def train_pairwise_heads(
    data: CompactRoutes,
    mode: str,
    config: dict[str, Any],
) -> tuple[PairwiseRouteHeads, list[float]]:
    training = config["training"]
    source_pairs, target_pairs = layer_pairs(24, list(range(1, 24)))
    features, targets = _pair_inputs_and_targets(data, mode, source_pairs, target_pairs)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(training["deterministic_algorithms"]))
    model = PairwiseRouteHeads(len(source_pairs), features.shape[2]).to(config["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
    )
    objective = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(training["positive_weight"]), device=config["device"])
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(training["batch_size_tokens"])
    history: list[float] = []
    model.train()
    for _ in range(int(training["epochs"])):
        order = torch.randperm(data.tokens, generator=generator)
        total = 0.0
        for offset in range(0, data.tokens, batch_size):
            indexes = order[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indexes].to(config["device"]))
            loss = objective(logits, targets[indexes].to(config["device"]))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indexes)
        history.append(total / data.tokens)
    return model, history


@torch.inference_mode()
def predict_pairwise_scores(
    model: PairwiseRouteHeads,
    data: CompactRoutes,
    mode: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_pairs, target_pairs = layer_pairs(24, list(range(1, 24)))
    features = route_features(data, mode)[:, source_pairs]
    batch_size = int(config["training"]["batch_size_tokens"])
    scores = np.empty((data.tokens, len(source_pairs), 32), dtype=np.float32)
    model.eval()
    for offset in range(0, data.tokens, batch_size):
        logits = model(features[offset : offset + batch_size].to(config["device"]))
        values = torch.sigmoid(logits).cpu().numpy()
        scores[offset : offset + len(values)] = values
    return scores, source_pairs.numpy(), target_pairs.numpy()


def predict_pairwise_ranks(
    model: PairwiseRouteHeads,
    data: CompactRoutes,
    mode: str,
    config: dict[str, Any],
    max_candidates: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, source_pairs, target_pairs = predict_pairwise_scores(
        model, data, mode, config
    )
    ranks = np.argsort(-scores, axis=2, kind="stable")[:, :, :max_candidates].astype(
        np.uint8
    )
    return ranks, source_pairs, target_pairs


def evaluate_ranks(
    ranks: np.ndarray,
    data: CompactRoutes,
    source_pairs: np.ndarray,
    target_pairs: np.ndarray,
    *,
    baseline: str,
    fold: int | str,
    training_requests: int,
    training_tokens: int,
    mode: str,
) -> list[dict[str, Any]]:
    target_ids = data.route_ids.numpy()[:, target_pairs]
    hits = (target_ids[:, :, :, None] == ranks[:, :, None, :]).any(axis=3)
    rows: list[dict[str, Any]] = []
    domains = sorted(set(data.domains))
    delta_values = target_pairs - source_pairs
    for delta in (1, 2, 3):
        pair_mask = delta_values == delta
        for domain in (*domains, "ALL"):
            token_mask = (
                np.asarray([value == domain for value in data.domains])
                if domain != "ALL"
                else np.ones(data.tokens, dtype=bool)
            )
            selected = hits[token_mask][:, pair_mask]
            rows.append(
                {
                    "fold": fold,
                    "training_requests": training_requests,
                    "training_tokens": training_tokens,
                    "feature_mode": mode,
                    "baseline": baseline,
                    "domain": domain,
                    "delta": delta,
                    "evaluation_tokens": int(token_mask.sum()),
                    "selection_coverage": float(selected.mean()),
                    "complete_route_coverage": float(selected.all(axis=2).mean()),
                }
            )
    return rows


def transition_ranks(
    training_data: CompactRoutes,
    evaluation_data: CompactRoutes,
    domains: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_pairs, target_pairs = layer_pairs(24, list(range(1, 24)))
    metadata = _evaluation_metadata(
        evaluation_data, source_pairs.numpy(), target_pairs.numpy(), domains
    )
    fitted = _fit_baselines(training_data, 24, 32, domains)
    flat = _baseline_ranks(
        "transition", evaluation_data, metadata, fitted, max_candidates=8
    )
    return (
        flat.reshape(evaluation_data.tokens, len(source_pairs), 8),
        source_pairs.numpy(),
        target_pairs.numpy(),
    )


def _compact_decode(tokens: list[Any], ids: set[int]) -> CompactRoutes:
    decode = [token for token in tokens if token.phase == "decode"]
    return compact_routes(decode, ids, 24)


def _select_mode(rows: list[dict[str, Any]], largest_requests: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    modes = sorted(
        {
            str(row["feature_mode"])
            for row in rows
            if row["baseline"] == "pairwise_head"
        }
    )
    for mode in modes:
        selected = [
            row
            for row in rows
            if row["baseline"] == "pairwise_head"
            and row["feature_mode"] == mode
            and row["domain"] == "ALL"
            and int(row["training_requests"]) == largest_requests
        ]
        candidates.append(
            {
                "feature_mode": mode,
                "mean_primary_selection_coverage": float(
                    np.mean([row["selection_coverage"] for row in selected])
                ),
                "mean_primary_complete_route_coverage": float(
                    np.mean([row["complete_route_coverage"] for row in selected])
                ),
                "fold_delta_cells": len(selected),
            }
        )
    winner = max(
        candidates,
        key=lambda row: (
            row["mean_primary_selection_coverage"],
            row["mean_primary_complete_route_coverage"],
            -{"weighted": 32, "binary": 32, "weighted_binary": 64}[
                row["feature_mode"]
            ],
        ),
    )
    return {"selection_rule": "highest mean selection at 72 requests", "candidates": candidates, "winner": winner}


def _cheap_domain_metrics(source_run: Path) -> dict[tuple[str, int], float]:
    path = source_run / "analysis" / "prediction" / "request_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, int], float] = {}
    domains = sorted({row["domain"] for row in rows})
    for domain in domains:
        for delta in (1, 2, 3):
            values: dict[str, float] = {}
            for baseline in ("domain_static", "source_copy"):
                selected = [
                    float(row["selection_coverage"])
                    for row in rows
                    if row["phase"] == "decode"
                    and int(row["delta"]) == delta
                    and int(row["candidate_count"]) == 8
                    and row["baseline"] == baseline
                    and row["domain"] == domain
                ]
                values[baseline] = float(np.mean(selected))
            result[(domain, delta)] = max(values.values())
    return result


def _final_decision(
    rows: list[dict[str, Any]], source_run: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cheap = _cheap_domain_metrics(source_run)
    result: list[dict[str, Any]] = []
    for delta in (1, 2, 3):
        learned = next(
            row
            for row in rows
            if row["baseline"] == "pairwise_head"
            and row["domain"] == "ALL"
            and int(row["delta"]) == delta
        )
        transition = next(
            row
            for row in rows
            if row["baseline"] == "transition"
            and row["domain"] == "ALL"
            and int(row["delta"]) == delta
        )
        domains = sorted(
            row["domain"]
            for row in rows
            if row["baseline"] == "pairwise_head"
            and row["domain"] != "ALL"
            and int(row["delta"]) == delta
        )
        domain_gains = {
            domain: next(
                float(row["selection_coverage"])
                for row in rows
                if row["baseline"] == "pairwise_head"
                and row["domain"] == domain
                and int(row["delta"]) == delta
            )
            - cheap[(domain, delta)]
            for domain in domains
        }
        checks = {
            "absolute_selection": learned["selection_coverage"] >= 0.82,
            "absolute_complete": learned["complete_route_coverage"] >= 0.50,
            "transition_selection_noninferiority": learned["selection_coverage"]
            - transition["selection_coverage"]
            >= -0.03,
            "transition_complete_noninferiority": learned["complete_route_coverage"]
            - transition["complete_route_coverage"]
            >= -0.05,
            "cheap_selection_gain": bool(
                np.mean(list(domain_gains.values())) >= 0.10
            ),
            "all_domain_selection_gains_positive": all(
                value > 0 for value in domain_gains.values()
            ),
        }
        result.append(
            {
                "delta": delta,
                "selection_coverage": learned["selection_coverage"],
                "complete_route_coverage": learned["complete_route_coverage"],
                "transition_selection_coverage": transition["selection_coverage"],
                "transition_complete_route_coverage": transition[
                    "complete_route_coverage"
                ],
                "selection_deficit_vs_transition": learned["selection_coverage"]
                - transition["selection_coverage"],
                "complete_deficit_vs_transition": learned["complete_route_coverage"]
                - transition["complete_route_coverage"],
                "mean_gain_vs_strongest_cheap": float(
                    np.mean(list(domain_gains.values()))
                ),
                "domain_gains": domain_gains,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    passing = sum(row["pass"] for row in result)
    return result, {
        "evidence_role": "exploratory counterfactual application of the Milestone F gate",
        "decision": "EXPLORATORY_GATE_EQUIVALENT_PASS" if passing >= 2 else "EXPLORATORY_GATE_EQUIVALENT_FAIL",
        "passing_lookaheads": passing,
        "required": 2,
        "confirmation_claim_allowed": False,
        "lookaheads": result,
    }


def run_exploratory(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(config["output_dir"])
    source_run = Path(config["source_run_dir"])
    split_path = Path(config["milestone_f_split"])
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256_file(source_run / "artifact_manifest.json") != config[
        "source_artifact_manifest_sha256"
    ]:
        raise ValueError("source artifact manifest changed")
    if sha256_file(split_path) != config["milestone_f_split_sha256"]:
        raise ValueError("Milestone F split changed")
    output_dir.mkdir(parents=True)
    shutil.copyfile(config_path, output_dir / "frozen_config.toml")
    write_json(
        output_dir / "run_definition.json",
        {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_run": str(source_run),
            "source_manifest_sha256": sha256_file(
                source_run / "artifact_manifest.json"
            ),
            "split_sha256": sha256_file(split_path),
            "evidence_role": config["evidence_role"],
            "created_before_first_fit": True,
        },
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    eligible = {int(value) for value in split["train_request_ids"]}
    development = {int(value) for value in split["test_request_ids"]}
    tokens, requests = _load_routes(source_run)
    domains = sorted({requests[value]["domain"] for value in eligible})
    cv = config["cross_validation"]
    folds = cross_validation_splits(
        requests,
        eligible,
        folds=int(cv["folds"]),
        validation_per_domain=int(cv["validation_requests_per_domain"]),
        ordering_seed=int(cv["ordering_seed"]),
    )
    write_json(output_dir / "cross_validation_splits.json", {"folds": folds})
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    started = time.monotonic()
    for fold_definition in folds:
        fold = int(fold_definition["fold"])
        validation_ids = set(fold_definition["validation_request_ids"])
        validation_data = _compact_decode(tokens, validation_ids)
        for per_domain in [int(value) for value in cv["train_requests_per_domain"]]:
            training_ids = {
                request_id
                for domain in domains
                for request_id in fold_definition["training_pool_by_domain"][domain][
                    :per_domain
                ]
            }
            training_data = _compact_decode(tokens, training_ids)
            transition, source_pairs, target_pairs = transition_ranks(
                training_data, validation_data, domains
            )
            rows.extend(
                evaluate_ranks(
                    transition,
                    validation_data,
                    source_pairs,
                    target_pairs,
                    baseline="transition",
                    fold=fold,
                    training_requests=len(training_ids),
                    training_tokens=training_data.tokens,
                    mode="table",
                )
            )
            for mode in config["models"]["feature_modes"]:
                model, history = train_pairwise_heads(training_data, str(mode), config)
                ranks, source_pairs, target_pairs = predict_pairwise_ranks(
                    model, validation_data, str(mode), config
                )
                rows.extend(
                    evaluate_ranks(
                        ranks,
                        validation_data,
                        source_pairs,
                        target_pairs,
                        baseline="pairwise_head",
                        fold=fold,
                        training_requests=len(training_ids),
                        training_tokens=training_data.tokens,
                        mode=str(mode),
                    )
                )
                histories.append(
                    {
                        "fold": fold,
                        "training_requests": len(training_ids),
                        "training_tokens": training_data.tokens,
                        "feature_mode": mode,
                        "initial_loss": history[0],
                        "final_loss": history[-1],
                    }
                )
                print(
                    json.dumps(
                        {
                            "fold": fold,
                            "training_requests": len(training_ids),
                            "mode": mode,
                            "final_loss": history[-1],
                        }
                    ),
                    flush=True,
                )
    largest = 4 * max(int(value) for value in cv["train_requests_per_domain"])
    selection = _select_mode(rows, largest)
    selected_mode = selection["winner"]["feature_mode"]

    final_training = _compact_decode(tokens, eligible)
    final_development = _compact_decode(tokens, development)
    final_model, final_history = train_pairwise_heads(
        final_training, selected_mode, config
    )
    final_ranks, source_pairs, target_pairs = predict_pairwise_ranks(
        final_model, final_development, selected_mode, config
    )
    final_rows = evaluate_ranks(
        final_ranks,
        final_development,
        source_pairs,
        target_pairs,
        baseline="pairwise_head",
        fold="final",
        training_requests=96,
        training_tokens=final_training.tokens,
        mode=selected_mode,
    )
    final_transition, source_pairs, target_pairs = transition_ranks(
        final_training, final_development, domains
    )
    final_rows.extend(
        evaluate_ranks(
            final_transition,
            final_development,
            source_pairs,
            target_pairs,
            baseline="transition",
            fold="final",
            training_requests=96,
            training_tokens=final_training.tokens,
            mode="table",
        )
    )
    _gate_rows, decision = _final_decision(final_rows, source_run)
    parameters = sum(value.numel() for value in final_model.parameters())
    checkpoint = output_dir / "model.pt"
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in final_model.state_dict().items()
            },
            "feature_mode": selected_mode,
            "config_sha256": sha256_file(output_dir / "frozen_config.toml"),
        },
        checkpoint,
    )
    accounting = {
        "selected_feature_mode": selected_mode,
        "parameters": parameters,
        "parameter_bytes_fp32": parameters * 4,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "macs_per_forecast": {"weighted": 1024, "binary": 1024, "weighted_binary": 2048}[
            selected_mode
        ],
        "heads": 276,
        "phase": "decode",
    }
    _write_csv(output_dir / "learning_curve.csv", rows)
    _write_csv(output_dir / "training_summary.csv", histories)
    _write_csv(output_dir / "final_development_metrics.csv", final_rows)
    write_json(output_dir / "model_selection.json", selection)
    write_json(output_dir / "model_accounting.json", accounting)
    write_json(output_dir / "decision.json", decision)
    result = {
        "evidence_role": config["evidence_role"],
        "elapsed_seconds": time.monotonic() - started,
        "cross_validation_model_selection": selection,
        "final_training_loss": final_history[-1],
        "model_accounting": accounting,
        "decision": decision,
    }
    write_json(output_dir / "result.json", result)
    _artifact_manifest(output_dir)
    return result


def evaluate_request_ranks(
    ranks: np.ndarray,
    data: CompactRoutes,
    source_pairs: np.ndarray,
    target_pairs: np.ndarray,
    *,
    baseline: str,
    candidate_count: int,
) -> list[dict[str, Any]]:
    ranks = ranks[:, :, :candidate_count]
    target_ids = data.route_ids.numpy()[:, target_pairs]
    target_weights = data.route_weights.numpy()[:, target_pairs]
    hits = (target_ids[:, :, :, None] == ranks[:, :, None, :]).any(axis=3)
    deltas = target_pairs - source_pairs
    rows: list[dict[str, Any]] = []
    for request_id in sorted({int(value) for value in data.request_ids}):
        token_mask = data.request_ids == request_id
        first = int(np.flatnonzero(token_mask)[0])
        for delta in (1, 2, 3):
            pair_mask = deltas == delta
            selected = hits[token_mask][:, pair_mask]
            weights = target_weights[token_mask][:, pair_mask]
            rows.append(
                {
                    "request_id": request_id,
                    "sample_id": data.sample_ids[first],
                    "domain": data.domains[first],
                    "phase": "decode",
                    "delta": delta,
                    "candidate_count": candidate_count,
                    "baseline": baseline,
                    "n_forecasts": int(selected.shape[0] * selected.shape[1]),
                    "selection_coverage": float(selected.mean()),
                    "candidate_precision": float(selected.mean()) * 4 / candidate_count,
                    "routed_mass_coverage": float((selected * weights).sum() / np.prod(selected.shape[:2])),
                    "complete_route_coverage": float(selected.all(axis=2).mean()),
                    "candidate_amplification": candidate_count / 4,
                    "candidate_set_fraction": candidate_count / 32,
                }
            )
    return rows


def summarize_request_metrics(request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        grouped[
            (int(row["delta"]), int(row["candidate_count"]), str(row["baseline"]))
        ].append(row)
    result: list[dict[str, Any]] = []
    for (delta, candidates, baseline), rows in sorted(grouped.items()):
        domains = sorted({str(row["domain"]) for row in rows})
        summary: dict[str, Any] = {
            "phase": "decode",
            "delta": delta,
            "candidate_count": candidates,
            "baseline": baseline,
            "requests": len(rows),
            "candidate_amplification": candidates / 4,
            "candidate_set_fraction": candidates / 32,
        }
        for metric in (
            "selection_coverage",
            "candidate_precision",
            "routed_mass_coverage",
            "complete_route_coverage",
        ):
            summary[metric] = float(
                np.mean(
                    [
                        np.mean(
                            [float(row[metric]) for row in rows if row["domain"] == domain]
                        )
                        for domain in domains
                    ]
                )
            )
        covered_per_forecast = 4 * summary["selection_coverage"]
        summary["useful_candidate_amplification"] = (
            candidates / covered_per_forecast if covered_per_forecast else math.inf
        )
        result.append(summary)
    return result


def calibration_from_pair_scores(
    scores: np.ndarray,
    data: CompactRoutes,
    source_pairs: np.ndarray,
    target_pairs: np.ndarray,
    bins: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_ids = data.route_ids.numpy()[:, target_pairs]
    labels = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(labels, target_ids, True, axis=2)
    deltas = target_pairs - source_pairs
    edges = np.linspace(0, 1, bins + 1)
    reliability: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for domain in sorted(set(data.domains)):
        token_mask = np.asarray([value == domain for value in data.domains])
        for delta in (1, 2, 3):
            pair_mask = deltas == delta
            probabilities = scores[token_mask][:, pair_mask].reshape(-1)
            outcomes = labels[token_mask][:, pair_mask].reshape(-1)
            bin_ids = np.minimum(
                np.searchsorted(edges, probabilities, side="right") - 1, bins - 1
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
                        "phase": "decode",
                        "domain": domain,
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
                    "phase": "decode",
                    "domain": domain,
                    "delta": delta,
                    "predictions": len(probabilities),
                    "positive_frequency": float(outcomes.mean()),
                    "mean_score": float(probabilities.mean()),
                    "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
                    "expected_calibration_error": absolute_error / len(probabilities),
                }
            )
    return reliability, summaries


def _confirmation_report(decision: dict[str, Any], accounting: dict[str, Any]) -> str:
    lines = [
        "# GPT-OSS 20B MTP-style route-head confirmation",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        (
            "The frozen 276-head weighted+binary route predictor was evaluated without "
            "refitting on 64 fresh requests (16 per domain) with zero prompt or sample-ID "
            "overlap against the previous 128 requests."
        ),
        "",
        "| Δ | Learned selection | Transition | Difference | Learned complete | Transition complete | Domains positive | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in decision["lookaheads"]:
        gains = json.loads(row["domain_selection_gains_json"])
        lines.append(
            f"| {row['delta']} | {100 * row['learned_selection_coverage']:.1f}% | "
            f"{100 * row['transition_selection_coverage']:.1f}% | "
            f"{100 * row['learned_minus_transition_selection_coverage']:+.1f} pp | "
            f"{100 * row['learned_complete_route_coverage']:.1f}% | "
            f"{100 * row['transition_complete_route_coverage']:.1f}% | "
            f"{sum(value > 0 for value in gains.values())}/{len(gains)} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"{decision['passing_lookaheads']}/{decision['required_passing_lookaheads']} required lookaheads passed.",
            "",
            (
                f"The predictor has {accounting['parameters']:,} FP32 parameters "
                f"({accounting['parameter_bytes_fp32'] / 1024 / 1024:.2f} MiB) and "
                f"costs {accounting['macs_per_forecast']:,} MACs per forecast."
            ),
            "",
            (
                "This confirms route-only expert-demand prediction on one checkpoint and "
                "four workload domains. It does not establish latency benefit, language "
                "quality, or the accuracy of a future hidden-state/jointly-trained head."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _confirmation_manifest(output_dir: Path) -> dict[str, Any]:
    trace = {
        str(path): sha256_file(path)
        for path in sorted((output_dir / "trace").glob("*.gz"))
    }
    durable = {
        str(path): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and "trace" not in path.relative_to(output_dir).parts
        and path.name != "artifact_manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "durable_files": durable,
        "disposable_trace_files": trace,
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def run_confirmation_analysis(
    confirmation_config: dict[str, Any], confirmation_config_path: str | Path
) -> dict[str, Any]:
    output_dir = Path(confirmation_config["output_dir"])
    integrity = json.loads((output_dir / "integrity.json").read_text(encoding="utf-8"))
    if integrity["decision"] != "TRACE_COMPLETE":
        raise ValueError("fresh confirmation trace is not complete")
    if int(integrity["totals"]["dispatch_id_mismatches"]) != 0:
        raise ValueError("fresh confirmation trace has dispatch ID mismatches")
    for path_key, hash_key in (
        ("prompt_file", "prompt_file_sha256"),
        ("prompt_manifest", "prompt_manifest_sha256"),
        ("predictor_checkpoint", "predictor_checkpoint_sha256"),
        ("predictor_config", "predictor_config_sha256"),
    ):
        if sha256_file(confirmation_config[path_key]) != confirmation_config[hash_key]:
            raise ValueError(f"frozen confirmation input changed: {path_key}")
    prompt_manifest = json.loads(
        Path(confirmation_config["prompt_manifest"]).read_text(encoding="utf-8")
    )
    if prompt_manifest["sample_id_overlap_with_original"] != 0 or prompt_manifest[
        "prompt_hash_overlap_with_original"
    ] != 0:
        raise ValueError("confirmation workload is not fresh")

    predictor_config_path = Path(confirmation_config["predictor_config"])
    import tomllib

    with predictor_config_path.open("rb") as handle:
        predictor_config = tomllib.load(handle)
    source_run = Path(predictor_config["source_run_dir"])
    split = json.loads(Path(predictor_config["milestone_f_split"]).read_text())
    original_tokens, original_requests = _load_routes(source_run)
    train_ids = {int(value) for value in split["train_request_ids"]}
    training_data = compact_routes(original_tokens, train_ids, 24)
    fresh_tokens, fresh_requests = _load_routes(output_dir)
    if len(fresh_requests) != 64:
        raise ValueError("confirmation trace does not contain exactly 64 requests")
    fresh_data = _compact_decode(fresh_tokens, set(fresh_requests))
    domains = sorted({metadata["domain"] for metadata in original_requests.values()})

    feature_mode = str(confirmation_config["predictor_feature_mode"])
    feature_width = {"weighted": 32, "binary": 32, "weighted_binary": 64}[
        feature_mode
    ]
    source_pairs, _target_pairs = layer_pairs(24, list(range(1, 24)))
    model = PairwiseRouteHeads(len(source_pairs), feature_width)
    checkpoint = torch.load(
        confirmation_config["predictor_checkpoint"], weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(confirmation_config["device"])
    learned_scores, source_np, target_np = predict_pairwise_scores(
        model, fresh_data, feature_mode, predictor_config
    )
    learned = np.argsort(-learned_scores, axis=2, kind="stable")[:, :, :16].astype(
        np.uint8
    )
    request_rows: list[dict[str, Any]] = []
    for candidate_count in (4, 8, 12, 16):
        request_rows.extend(
            evaluate_request_ranks(
                learned,
                fresh_data,
                source_np,
                target_np,
                baseline="learned",
                candidate_count=candidate_count,
            )
        )
    fitted = _fit_baselines(training_data, 24, 32, domains)
    metadata = _evaluation_metadata(fresh_data, source_np, target_np, domains)
    for baseline in ("global_static", "domain_static", "source_copy", "transition"):
        flat = _baseline_ranks(
            baseline, fresh_data, metadata, fitted, max_candidates=16
        )
        ranks = flat.reshape(fresh_data.tokens, len(source_np), 16)
        for candidate_count in (4, 8, 12, 16):
            request_rows.extend(
                evaluate_request_ranks(
                    ranks,
                    fresh_data,
                    source_np,
                    target_np,
                    baseline=baseline,
                    candidate_count=candidate_count,
                )
            )

    gate_config = {
        "evaluation": confirmation_config["analysis"],
        "decision_gate": confirmation_config["decision_gate"],
    }
    gate_rows, decision = development_gate(request_rows, gate_config)
    passed = decision["decision"] == "DEVELOPMENT_PASS"
    decision["evidence_stage"] = "fresh_confirmation"
    decision["decision"] = "CONFIRMATION_PASS" if passed else "CONFIRMATION_FAIL"
    decision["fresh_requests"] = 64
    decision["requests_per_domain"] = 16
    decision["predictor_refit"] = False
    decision["prompt_overlap"] = {
        "sample_ids": prompt_manifest["sample_id_overlap_with_original"],
        "prompt_hashes": prompt_manifest["prompt_hash_overlap_with_original"],
    }
    decision["claim_boundary"] = (
        "confirmed route-only prediction on one GPT-OSS 20B checkpoint and four "
        "domains; no latency, language-quality, or jointly-trained-head claim"
    )
    accounting = json.loads(
        (Path(confirmation_config["predictor_checkpoint"]).parent / "model_accounting.json").read_text()
    )
    analysis_dir = output_dir / "analysis" / "multihead_confirmation"
    reliability, calibration = calibration_from_pair_scores(
        learned_scores, fresh_data, source_np, target_np
    )
    _write_csv(analysis_dir / "request_metrics.csv", request_rows)
    _write_csv(
        analysis_dir / "horizon_summary.csv",
        summarize_request_metrics(request_rows),
    )
    _write_csv(analysis_dir / "bootstrap_gate.csv", gate_rows)
    _write_csv(analysis_dir / "calibration_reliability.csv", reliability)
    _write_csv(analysis_dir / "calibration_summary.csv", calibration)
    write_json(analysis_dir / "decision.json", decision)
    (output_dir / "REPORT.md").write_text(
        _confirmation_report(decision, accounting), encoding="utf-8"
    )
    result = {
        "decision": decision,
        "model_accounting": accounting,
        "trace_totals": integrity["totals"],
        "confirmation_config_sha256": sha256_file(confirmation_config_path),
        "predictor_checkpoint_sha256": sha256_file(
            confirmation_config["predictor_checkpoint"]
        ),
    }
    write_json(output_dir / "result.json", result)
    _confirmation_manifest(output_dir)
    return result
