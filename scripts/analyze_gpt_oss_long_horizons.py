#!/usr/bin/env python3
"""Post-hoc Δ=1..23 evaluation of the frozen GPT-OSS route heads."""

from __future__ import annotations

import csv
import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ep_predict.analysis.gpt_oss_learned import (
    _baseline_ranks,
    _evaluation_metadata,
    _fit_baselines,
    compact_routes,
    layer_pairs,
    sha256_file,
)
from ep_predict.analysis.gpt_oss_multihead import (
    PairwiseRouteHeads,
    predict_pairwise_scores,
)
from ep_predict.analysis.gpt_oss_prediction import _load_routes
from ep_predict.tracing.storage import write_json

CONFIRMATION = Path("artifacts/runs/gpt-oss-20b-mtp-head-confirmation")
CHECKPOINT = Path("artifacts/runs/gpt-oss-20b-mtp-head-exploratory/model.pt")
PREDICTOR_CONFIG = Path(
    "artifacts/runs/gpt-oss-20b-mtp-head-exploratory/frozen_config.toml"
)
OUTPUT = Path("artifacts/analysis/gpt-oss-20b-long-horizon-exploratory")

LAYERS = 24
EXPERTS = 32
TOP_K = 4
CANDIDATE_COUNTS = (4, 8, 12, 16)
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260804


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def request_horizon_metrics(
    ranks: np.ndarray,
    data: Any,
    source_pairs: np.ndarray,
    target_pairs: np.ndarray,
    *,
    baseline: str,
    candidate_count: int,
) -> list[dict[str, Any]]:
    candidates = ranks[:, :, :candidate_count]
    target_ids = data.route_ids.numpy()[:, target_pairs]
    target_weights = data.route_weights.numpy()[:, target_pairs]
    hits = (target_ids[:, :, :, None] == candidates[:, :, None, :]).any(axis=3)
    deltas = target_pairs - source_pairs
    rows: list[dict[str, Any]] = []
    for request_id in sorted({int(value) for value in data.request_ids}):
        token_mask = data.request_ids == request_id
        first = int(np.flatnonzero(token_mask)[0])
        for delta in range(1, LAYERS):
            pair_mask = deltas == delta
            selected = hits[token_mask][:, pair_mask]
            weights = target_weights[token_mask][:, pair_mask]
            forecasts = int(selected.shape[0] * selected.shape[1])
            rows.append(
                {
                    "request_id": request_id,
                    "sample_id": data.sample_ids[first],
                    "domain": data.domains[first],
                    "phase": "decode",
                    "baseline": baseline,
                    "delta": delta,
                    "layer_pairs": int(pair_mask.sum()),
                    "candidate_count": candidate_count,
                    "candidate_amplification": candidate_count / TOP_K,
                    "forecasts": forecasts,
                    "selection_coverage": float(selected.mean()),
                    "routed_mass_coverage": float(
                        (selected * weights).sum() / forecasts
                    ),
                    "complete_route_coverage": float(selected.all(axis=2).mean()),
                }
            )
    return rows


def summarize_with_request_bootstrap(
    request_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        grouped[
            (str(row["baseline"]), int(row["candidate_count"]), int(row["delta"]))
        ].append(row)
    metrics = (
        "selection_coverage",
        "routed_mass_coverage",
        "complete_route_coverage",
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries: list[dict[str, Any]] = []
    for (baseline, candidate_count, delta), rows in sorted(grouped.items()):
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_domain[str(row["domain"])].append(row)
        if set(map(len, by_domain.values())) != {16}:
            raise ValueError("long-horizon bootstrap expects 16 requests per domain")
        summary: dict[str, Any] = {
            "baseline": baseline,
            "candidate_count": candidate_count,
            "candidate_amplification": candidate_count / TOP_K,
            "delta": delta,
            "layer_pairs": LAYERS - delta,
            "requests": len(rows),
            "forecasts_per_request": int(rows[0]["forecasts"]),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
        }
        for metric in metrics:
            domain_values = [
                np.asarray([float(row[metric]) for row in by_domain[domain]])
                for domain in sorted(by_domain)
            ]
            summary[metric] = float(
                np.mean([float(values.mean()) for values in domain_values])
            )
            bootstrapped_domains = []
            for values in domain_values:
                sampled = rng.integers(
                    0,
                    len(values),
                    size=(BOOTSTRAP_RESAMPLES, len(values)),
                )
                bootstrapped_domains.append(values[sampled].mean(axis=1))
            bootstrapped = np.mean(bootstrapped_domains, axis=0)
            summary[f"{metric}_ci_low"] = float(np.quantile(bootstrapped, 0.025))
            summary[f"{metric}_ci_high"] = float(np.quantile(bootstrapped, 0.975))
        summaries.append(summary)
    return summaries


def _series(
    summaries: list[dict[str, Any]], baseline: str, candidates: int
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in summaries
            if row["baseline"] == baseline
            and int(row["candidate_count"]) == candidates
        ],
        key=lambda row: int(row["delta"]),
    )


def plot_long_horizon_tradeoff(summaries: list[dict[str, Any]]) -> None:
    colors = {
        "selection": "#3a6ea5",
        "complete": "#009e73",
        4: "#d55e00",
        8: "#3a6ea5",
        12: "#009e73",
        16: "#8172b3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True, sharey=True)
    for axis in axes:
        axis.set_xlim(1, 23)
        axis.set_ylim(50, 100)
        axis.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])
        axis.set_xlabel("Lookahead Δ (layers)")
        axis.grid(axis="y", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    learned = _series(summaries, "learned", 8)
    deltas = np.asarray([int(row["delta"]) for row in learned])
    for metric, label, color in (
        ("selection_coverage", "Expert selection", colors["selection"]),
        ("complete_route_coverage", "Complete top-4 route", colors["complete"]),
    ):
        values = 100 * np.asarray([float(row[metric]) for row in learned])
        low = 100 * np.asarray([float(row[f"{metric}_ci_low"]) for row in learned])
        high = 100 * np.asarray([float(row[f"{metric}_ci_high"]) for row in learned])
        axes[0].plot(
            deltas,
            values,
            color=color,
            linewidth=2.2,
            marker="o" if metric == "selection_coverage" else "s",
            markersize=3.5,
            markevery=2,
            label=label,
        )
        axes[0].fill_between(deltas, low, high, color=color, alpha=0.14)
    axes[0].set_title("K=8 prediction coverage")
    axes[0].set_ylabel("Coverage (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    for candidate_count in CANDIDATE_COUNTS:
        rows = _series(summaries, "learned", candidate_count)
        values = 100 * np.asarray(
            [float(row["selection_coverage"]) for row in rows]
        )
        low = 100 * np.asarray(
            [float(row["selection_coverage_ci_low"]) for row in rows]
        )
        high = 100 * np.asarray(
            [float(row["selection_coverage_ci_high"]) for row in rows]
        )
        axes[1].plot(
            deltas,
            values,
            linewidth=2.0,
            color=colors[candidate_count],
            marker="o",
            markersize=3.0,
            markevery=2,
            label=f"K={candidate_count} ({candidate_count / TOP_K:.0f}×)",
        )
        axes[1].fill_between(
            deltas, low, high, color=colors[candidate_count], alpha=0.09
        )
    axes[1].set_title("Candidate amplification")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left", ncol=2)

    fig.suptitle(
        "Expert-demand prediction across lookahead horizons",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.92), w_pad=2.0)
    fig.savefig(OUTPUT / "fig1_long_horizon_tradeoff.png", dpi=300)
    fig.savefig(OUTPUT / "fig1_long_horizon_tradeoff.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with PREDICTOR_CONFIG.open("rb") as handle:
        predictor_config = tomllib.load(handle)
    cpu_config = {**predictor_config, "device": "cpu"}
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint["config_sha256"] != sha256_file(PREDICTOR_CONFIG):
        raise ValueError("predictor checkpoint and frozen config disagree")

    fresh_tokens, fresh_requests = _load_routes(CONFIRMATION)
    fresh_decode = [token for token in fresh_tokens if token.phase == "decode"]
    fresh_data = compact_routes(fresh_decode, set(fresh_requests), LAYERS)
    source_pairs, _target_pairs = layer_pairs(LAYERS, list(range(1, LAYERS)))
    model = PairwiseRouteHeads(len(source_pairs), feature_width=64)
    model.load_state_dict(checkpoint["state_dict"])
    learned_scores, source_np, target_np = predict_pairwise_scores(
        model, fresh_data, checkpoint["feature_mode"], cpu_config
    )
    learned_ranks = np.argsort(-learned_scores, axis=2, kind="stable")[
        :, :, : max(CANDIDATE_COUNTS)
    ].astype(np.uint8)

    split = json.loads(
        Path(predictor_config["milestone_f_split"]).read_text(encoding="utf-8")
    )
    original_tokens, original_requests = _load_routes(
        Path(predictor_config["source_run_dir"])
    )
    training_ids = {int(value) for value in split["train_request_ids"]}
    training_data = compact_routes(original_tokens, training_ids, LAYERS)
    domains = sorted({value["domain"] for value in original_requests.values()})
    fitted = _fit_baselines(training_data, LAYERS, EXPERTS, domains)
    metadata = _evaluation_metadata(fresh_data, source_np, target_np, domains)
    transition_flat = _baseline_ranks(
        "transition",
        fresh_data,
        metadata,
        fitted,
        max_candidates=max(CANDIDATE_COUNTS),
    )
    transition_ranks = transition_flat.reshape(
        fresh_data.tokens, len(source_np), max(CANDIDATE_COUNTS)
    )

    request_rows: list[dict[str, Any]] = []
    for candidate_count in CANDIDATE_COUNTS:
        request_rows.extend(
            request_horizon_metrics(
                learned_ranks,
                fresh_data,
                source_np,
                target_np,
                baseline="learned",
                candidate_count=candidate_count,
            )
        )
    request_rows.extend(
        request_horizon_metrics(
            transition_ranks,
            fresh_data,
            source_np,
            target_np,
            baseline="transition",
            candidate_count=8,
        )
    )
    summaries = summarize_with_request_bootstrap(request_rows)
    _write_csv(OUTPUT / "request_metrics.csv", request_rows)
    _write_csv(OUTPUT / "horizon_summary.csv", summaries)
    plot_long_horizon_tradeoff(summaries)

    learned_k8 = _series(summaries, "learned", 8)
    result = {
        "schema_version": 1,
        "evidence_role": (
            "post-hoc exploratory long-horizon evaluation; only delta 1-3 were "
            "predeclared and confirmatory"
        ),
        "predictor_training_horizons": list(range(1, LAYERS)),
        "evaluation_horizons": list(range(1, LAYERS)),
        "candidate_counts": list(CANDIDATE_COUNTS),
        "requests": len(fresh_requests),
        "decode_tokens": fresh_data.tokens,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "selected_points": {
            str(delta): next(row for row in learned_k8 if int(row["delta"]) == delta)
            for delta in (1, 3, 6, 12, 18, 23)
        },
        "claim_boundary": (
            "accuracy is averaged over all valid source-target pairs at each delta; the "
            "pair set shrinks with delta, so this is operational horizon performance, "
            "not a causal estimate of distance alone"
        ),
    }
    write_json(OUTPUT / "result.json", result)
    durable = {
        str(path): sha256_file(path)
        for path in sorted(OUTPUT.glob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    write_json(
        OUTPUT / "artifact_manifest.json",
        {
            "schema_version": 1,
            "source": {
                "confirmation_manifest_sha256": sha256_file(
                    CONFIRMATION / "artifact_manifest.json"
                ),
                "predictor_checkpoint_sha256": sha256_file(CHECKPOINT),
                "predictor_config_sha256": sha256_file(PREDICTOR_CONFIG),
            },
            "durable_files": durable,
        },
    )
    print(json.dumps(result["selected_points"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
