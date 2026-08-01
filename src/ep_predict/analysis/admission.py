from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ep_predict.analysis.h2 import _load_token_routes
from ep_predict.analysis.h3 import _load_feature_map
from ep_predict.analysis.h4 import _decode_waves
from ep_predict.analysis.h5 import (
    _load_linear_arrays,
    _test_residency,
    _transition_tables,
)
from ep_predict.tracing.storage import write_json


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _transition_score_vector(
    source: tuple[int, ...],
    *,
    rows: dict[int, Counter[int]],
    marginal: Counter[int],
    num_experts: int,
) -> np.ndarray:
    marginal_total = sum(marginal.values())
    scores = np.zeros(num_experts, dtype=np.float32)
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
        scores /= len(source)
    return scores


def _linear_score_vector(
    arrays: dict[str, np.ndarray],
    *,
    source_layer: int,
    delta: int,
    features: np.ndarray,
) -> np.ndarray:
    # Reuse the exact persisted head semantics while returning all expert logits.
    prefix = f"decode_l{source_layer:02d}_d{delta}"
    weight = arrays[f"{prefix}_weight"]
    bias = arrays[f"{prefix}_bias"]
    mean = arrays[f"{prefix}_mean"]
    scale = arrays[f"{prefix}_scale"]
    return (
        ((features.astype(np.float32) - mean) / scale) @ weight.T + bias
    ).astype(np.float32, copy=False)


def _average_tie_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool, copy=False)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop] == ordered[start]:
            stop += 1
        # One-indexed mean rank for the tied group.
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _score_matrices(
    *,
    policy: str,
    delta: int,
    capacity: int,
    layers: int,
    test_tokens: list[Any],
    token_index: dict[tuple[int, int], int],
    residency: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]],
    transitions: dict[
        tuple[str, int, int], dict[int, Counter[int]]
    ],
    marginals: dict[tuple[str, int], Counter[int]],
    linear: dict[str, np.ndarray],
    features: dict[tuple[int, str, int, int], np.ndarray],
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_rows: list[np.ndarray] = []
    cold_rows: list[np.ndarray] = []
    resident_rows: list[np.ndarray] = []
    for token in test_tokens:
        index = token_index[(token.request_id, token.token_position)]
        for source_layer in range(layers - delta):
            target_layer = source_layer + delta
            resident, cold = residency[(index, target_layer)]
            if not cold:
                continue
            if policy == "transition":
                scores = _transition_score_vector(
                    token.routes[source_layer],
                    rows=transitions[("decode", source_layer, delta)],
                    marginal=marginals[("decode", target_layer)],
                    num_experts=num_experts,
                )
            else:
                scores = _linear_score_vector(
                    linear,
                    source_layer=source_layer,
                    delta=delta,
                    features=features[
                        (
                            token.request_id,
                            token.phase,
                            token.token_position,
                            source_layer,
                        )
                    ],
                )
            scale = float(scores.std())
            standardized = (
                (scores - float(scores.mean())) / scale
                if scale > 1e-8
                else np.zeros_like(scores)
            )
            cold_mask = np.zeros(num_experts, dtype=bool)
            cold_mask[list(cold)] = True
            resident_mask = np.zeros(num_experts, dtype=bool)
            resident_mask[list(resident)] = True
            score_rows.append(standardized)
            cold_rows.append(cold_mask)
            resident_rows.append(resident_mask)
    if not score_rows:
        raise ValueError(f"no cold waves for {policy}, Δ={delta}")
    return (
        np.stack(score_rows),
        np.stack(cold_rows),
        np.stack(resident_rows),
    )


def _threshold_sweep(
    *,
    scores: np.ndarray,
    cold: np.ndarray,
    resident: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    nonresident = ~resident
    cold_experts = int(cold.sum())
    cold_waves = len(scores)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        admitted = nonresident & (scores >= threshold)
        useful_by_wave = (admitted & cold).sum(axis=1)
        admitted_by_wave = admitted.sum(axis=1)
        useful = int(useful_by_wave.sum())
        admitted_count = int(admitted_by_wave.sum())
        false = admitted_count - useful
        complete = int(np.all(~cold | admitted, axis=1).sum())
        rows.append(
            {
                "standardized_score_threshold": threshold,
                "eligible_cold_waves": cold_waves,
                "cold_demand_experts": cold_experts,
                "admitted_candidate_experts": admitted_count,
                "useful_admitted_experts": useful,
                "false_admitted_experts": false,
                "cold_expert_coverage": useful / cold_experts,
                "complete_cold_set_coverage": complete / cold_waves,
                "candidate_transfer_amplification": (
                    admitted_count / useful if useful else float("inf")
                ),
                "mean_admitted_experts_per_cold_wave": (
                    admitted_count / cold_waves
                ),
                "candidate_precision": (
                    useful / admitted_count if admitted_count else 0.0
                ),
            }
        )
    return rows


def _histogram_rows(
    *,
    scores: np.ndarray,
    cold: np.ndarray,
    resident: np.ndarray,
    bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nonresident = ~resident
    values = scores[nonresident]
    labels = cold[nonresident]
    useful = values[labels]
    useless = values[~labels]
    edges = np.linspace(-4.0, 5.0, bins + 1)
    # Keep the rare out-of-range standardized scores in the edge bins. This
    # makes the class distributions proper probability distributions for the
    # divergence estimates below instead of silently dropping tail mass.
    lower = np.nextafter(edges[0], edges[-1])
    upper = np.nextafter(edges[-1], edges[0])
    useful_counts, _ = np.histogram(np.clip(useful, lower, upper), bins=edges)
    useless_counts, _ = np.histogram(
        np.clip(useless, lower, upper), bins=edges
    )
    rows: list[dict[str, Any]] = []
    for label, counts in (
        ("useful_cold", useful_counts),
        ("useless", useless_counts),
    ):
        widths = np.diff(edges)
        density = counts / (counts.sum() * widths)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "candidate_class": label,
                    "bin_left": edges[index],
                    "bin_right": edges[index + 1],
                    "bin_center": (edges[index] + edges[index + 1]) / 2,
                    "count": int(count),
                    "density": density[index],
                }
            )
    divergence = _histogram_divergence(useful_counts, useless_counts)
    summary = {
        "useful_expert_scores": len(useful),
        "useless_expert_scores": len(useless),
        "useful_mean_standardized_score": float(useful.mean()),
        "useless_mean_standardized_score": float(useless.mean()),
        "useful_median_standardized_score": float(np.median(useful)),
        "useless_median_standardized_score": float(np.median(useless)),
        "useful_vs_useless_auroc": _average_tie_auroc(values, labels),
        **divergence,
    }
    return rows, summary


def _histogram_divergence(
    useful_counts: np.ndarray,
    useless_counts: np.ndarray,
) -> dict[str, float]:
    """Measure class-conditional score separation on a shared histogram.

    Jensen-Shannon divergence and overlap are the primary descriptive
    quantities because they are symmetric and finite. Directed KL estimates
    use a Jeffreys 0.5-count correction and are included only to make the
    relationship to KL explicit; all histogram divergences remain bin-aware
    descriptive statistics rather than predictor objectives.
    """

    useful_counts = np.asarray(useful_counts, dtype=np.float64)
    useless_counts = np.asarray(useless_counts, dtype=np.float64)
    useful_total = float(useful_counts.sum())
    useless_total = float(useless_counts.sum())
    if useful_total <= 0 or useless_total <= 0:
        raise ValueError("both score classes must contain observations")

    p = useful_counts / useful_total
    q = useless_counts / useless_total
    midpoint = 0.5 * (p + q)

    def finite_kl(left: np.ndarray, right: np.ndarray) -> float:
        active = left > 0
        return float(np.sum(left[active] * np.log2(left[active] / right[active])))

    js_bits = 0.5 * finite_kl(p, midpoint) + 0.5 * finite_kl(q, midpoint)
    total_variation = 0.5 * float(np.abs(p - q).sum())

    # KL is otherwise infinite whenever one finite-sample histogram has an
    # empty bin. The correction makes the reported estimate reproducible.
    p_smoothed = (useful_counts + 0.5) / (
        useful_total + 0.5 * len(useful_counts)
    )
    q_smoothed = (useless_counts + 0.5) / (
        useless_total + 0.5 * len(useless_counts)
    )

    total = useful_total + useless_total
    useful_prior = useful_total / total
    label_entropy_bits = -(
        useful_prior * np.log2(useful_prior)
        + (1.0 - useful_prior) * np.log2(1.0 - useful_prior)
    )
    joint = np.stack((useful_counts, useless_counts)) / total
    score_bin_prior = joint.sum(axis=0)
    class_prior = joint.sum(axis=1)
    mutual_information_bits = 0.0
    for class_index in range(2):
        active = joint[class_index] > 0
        mutual_information_bits += float(
            np.sum(
                joint[class_index, active]
                * np.log2(
                    joint[class_index, active]
                    / (
                        class_prior[class_index]
                        * score_bin_prior[active]
                    )
                )
            )
        )

    return {
        "useful_base_rate": useful_prior,
        "score_js_divergence_bits": js_bits,
        "score_distribution_total_variation": total_variation,
        "score_distribution_overlap": 1.0 - total_variation,
        "score_kl_useful_to_useless_bits_smoothed": finite_kl(
            p_smoothed, q_smoothed
        ),
        "score_kl_useless_to_useful_bits_smoothed": finite_kl(
            q_smoothed, p_smoothed
        ),
        "score_label_mutual_information_bits": mutual_information_bits,
        "score_label_entropy_bits": float(label_entropy_bits),
        "score_label_entropy_explained": (
            mutual_information_bits / label_entropy_bits
        ),
    }


def analyze_admission(experiment_config: dict[str, Any]) -> dict[str, Any]:
    output = Path(experiment_config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    settings = experiment_config["analysis"]
    capacity = int(settings["resident_capacity"])
    lookaheads = [int(value) for value in settings["lookaheads"]]
    policies = [str(value) for value in settings["policies"]]
    threshold_config = settings["threshold_sweep"]
    thresholds = [
        float(value)
        for value in np.arange(
            float(threshold_config["start"]),
            float(threshold_config["stop"])
            + 0.5 * float(threshold_config["step"]),
            float(threshold_config["step"]),
        )
    ]

    h1_run = Path(experiment_config["h1_run"])
    h3_run = Path(experiment_config["h3_run"])
    h3_analysis = Path(experiment_config["h3_analysis"])
    model_report = json.loads(
        (h1_run / "model_report.json").read_text(encoding="utf-8")
    )
    layers = int(model_report["router_count"])
    expert_counts = {int(row["num_experts"]) for row in model_report["routers"]}
    if len(expert_counts) != 1:
        raise ValueError("admission analysis requires one expert count")
    num_experts = expert_counts.pop()

    h3_tokens, _ = _load_token_routes(h3_run)
    h1_tokens, _ = _load_token_routes(h1_run)
    split = json.loads((h3_analysis / "split.json").read_text(encoding="utf-8"))
    train_ids = {int(value) for value in split["train_request_ids"]}
    test_ids = {int(value) for value in split["test_request_ids"]}
    train_tokens = [token for token in h3_tokens if token.request_id in train_ids]
    test_tokens = [
        token
        for token in h3_tokens
        if token.request_id in test_ids and token.phase == "decode"
    ]
    h1_decode = [token for token in h1_tokens if token.phase == "decode"]
    h3_decode = [token for token in h3_tokens if token.phase == "decode"]
    if len(h1_decode) != len(h3_decode):
        raise ValueError("H1/H3 decode token count differs")
    token_index: dict[tuple[int, int], int] = {}
    for index, (h1, h3) in enumerate(zip(h1_decode, h3_decode, strict=True)):
        identity_h1 = (h1.request_id, h1.token_position)
        identity_h3 = (h3.request_id, h3.token_position)
        if identity_h1 != identity_h3 or h1.routes != h3.routes:
            raise ValueError(f"H1/H3 route mismatch at decode token {index}")
        token_index[identity_h3] = index

    waves, _token_count = _decode_waves(h1_run)
    residency = _test_residency(waves, capacity=capacity, test_ids=test_ids)
    marginals, transitions = _transition_tables(
        train_tokens, set(lookaheads)
    )
    features, feature_integrity = _load_feature_map(h3_run)
    linear = _load_linear_arrays(h3_analysis / "linear_predictors.npz")

    frontier_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    separation_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    max_amplification = float(
        experiment_config["screen"]["max_candidate_transfer_amplification"]
    )
    for delta in lookaheads:
        for policy in policies:
            scores, cold, resident = _score_matrices(
                policy=policy,
                delta=delta,
                capacity=capacity,
                layers=layers,
                test_tokens=test_tokens,
                token_index=token_index,
                residency=residency,
                transitions=transitions,
                marginals=marginals,
                linear=linear,
                features=features,
                num_experts=num_experts,
            )
            sweep_rows = _threshold_sweep(
                scores=scores,
                cold=cold,
                resident=resident,
                thresholds=thresholds,
            )
            for row in sweep_rows:
                frontier_rows.append(
                    {
                        "phase": "decode",
                        "resident_capacity": capacity,
                        "lookahead": delta,
                        "policy": policy,
                        **row,
                    }
                )
            eligible = [
                row
                for row in sweep_rows
                if row["candidate_transfer_amplification"]
                <= max_amplification + 1e-12
            ]
            finite = [
                row
                for row in sweep_rows
                if np.isfinite(row["candidate_transfer_amplification"])
            ]
            if eligible:
                best = max(
                    eligible,
                    key=lambda row: (
                        row["complete_cold_set_coverage"],
                        row["cold_expert_coverage"],
                        -row["candidate_transfer_amplification"],
                    ),
                )
            else:
                best = min(
                    finite,
                    key=lambda row: (
                        row["candidate_transfer_amplification"],
                        -row["complete_cold_set_coverage"],
                    ),
                )
            best_rows.append(
                {
                    "phase": "decode",
                    "resident_capacity": capacity,
                    "lookahead": delta,
                    "policy": policy,
                    "within_2x_window": bool(eligible),
                    **best,
                }
            )
            histogram, separation = _histogram_rows(
                scores=scores,
                cold=cold,
                resident=resident,
                bins=int(settings["score_histogram_bins"]),
            )
            for row in histogram:
                histogram_rows.append(
                    {
                        "phase": "decode",
                        "resident_capacity": capacity,
                        "lookahead": delta,
                        "policy": policy,
                        **row,
                    }
                )
            separation_rows.append(
                {
                    "phase": "decode",
                    "resident_capacity": capacity,
                    "lookahead": delta,
                    "policy": policy,
                    **separation,
                }
            )

    _write_csv(output / "admission_frontier.csv", frontier_rows)
    _write_csv(output / "best_at_2x.csv", best_rows)
    _write_csv(output / "score_histograms.csv", histogram_rows)
    _write_csv(output / "score_separation.csv", separation_rows)
    reference_coverage = float(
        experiment_config["screen"]["reference_complete_cold_set_coverage"]
    )
    reference_boundary_rows: list[dict[str, Any]] = []
    for delta in lookaheads:
        for policy in policies:
            candidates = [
                row
                for row in frontier_rows
                if row["lookahead"] == delta
                and row["policy"] == policy
                and row["complete_cold_set_coverage"] >= reference_coverage
                and np.isfinite(row["candidate_transfer_amplification"])
            ]
            boundary = min(
                candidates,
                key=lambda row: row["candidate_transfer_amplification"],
            )
            reference_boundary_rows.append(boundary)
    _write_csv(
        output / "boundary_at_reference_coverage.csv",
        reference_boundary_rows,
    )
    passing = [
        row
        for row in best_rows
        if row["within_2x_window"]
        and row["complete_cold_set_coverage"] >= reference_coverage
    ]
    summary = {
        "analysis": "h5_admission_separation",
        "status": "post_hoc_mechanism_diagnosis",
        "evidence_grade": "held_out_trace_driven_analytical_pilot",
        "test_requests": len(test_ids),
        "resident_capacity": capacity,
        "lookaheads": lookaheads,
        "policies": policies,
        "screen": experiment_config["screen"],
        "best_at_2x": best_rows,
        "boundary_at_reference_coverage": reference_boundary_rows,
        "cells_reaching_reference_coverage_at_2x": [
            {"lookahead": row["lookahead"], "policy": row["policy"]}
            for row in passing
        ],
        "score_separation": separation_rows,
        "feature_integrity": feature_integrity,
        "outputs": {
            "frontier": str(output / "admission_frontier.csv"),
            "best_at_2x": str(output / "best_at_2x.csv"),
            "histograms": str(output / "score_histograms.csv"),
            "separation": str(output / "score_separation.csv"),
            "reference_boundary": str(
                output / "boundary_at_reference_coverage.csv"
            ),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
