#!/usr/bin/env python3
"""Render qualitative lookahead mechanics from one real confirmation token."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from ep_predict.analysis.gpt_oss_learned import compact_routes, layer_pairs, sha256_file
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
OUTPUT = Path("artifacts/visuals/gpt-oss-lookahead")

EXPERTS = 32
LAYERS = 24
TOP_K = 4
PRIMARY_K = 8
ANIMATION_DELTA = 2

BACKGROUND = "#f7f7f5"
PREDICTED = "#8ecae6"
HIT = "#009e73"
MISS = "#d55e00"
UNSCORED = "#8d99ae"
PENDING = "#3a6ea5"
CURRENT = "#e69f00"
GRID = "#ffffff"
TEXT = "#17212b"

# Integer cells make the legend semantic and prevent interpolation artifacts.
CELL_COLORS = [BACKGROUND, PREDICTED, HIT, MISS, UNSCORED, PENDING]
CELL_CMAP = ListedColormap(CELL_COLORS)
CELL_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), CELL_CMAP.N)


def _candidate_hits(
    ranks: np.ndarray,
    route_ids: np.ndarray,
    target_pairs: np.ndarray,
) -> np.ndarray:
    targets = route_ids[:, target_pairs]
    return (targets[:, :, :, None] == ranks[:, :, None, :]).any(axis=3)


def _choose_representative_token(
    hits: np.ndarray,
    source_pairs: np.ndarray,
    target_pairs: np.ndarray,
    data: Any,
) -> tuple[int, dict[str, Any]]:
    """Choose the empirical medoid, while requiring visible Δ=2 misses."""

    deltas = target_pairs - source_pairs
    primary = np.isin(deltas, (1, 2, 3))
    delta_two = deltas == ANIMATION_DELTA
    per_token_selection = hits[:, primary].mean(axis=(1, 2))
    per_token_complete = hits[:, primary].all(axis=2).mean(axis=1)
    per_token_delta_two = hits[:, delta_two].mean(axis=(1, 2))
    target = np.asarray(
        [
            np.median(per_token_selection),
            np.median(per_token_complete),
            np.median(per_token_delta_two),
        ]
    )
    values = np.column_stack(
        (per_token_selection, per_token_complete, per_token_delta_two)
    )
    distance = np.abs(values - target) @ np.asarray([1.0, 0.35, 0.75])
    delta_two_misses = (~hits[:, delta_two]).sum(axis=(1, 2))
    eligible = np.flatnonzero((delta_two_misses >= 2) & (delta_two_misses <= 20))
    if not len(eligible):
        eligible = np.arange(data.tokens)
    key = lambda index: (
        float(distance[index]),
        int(data.request_ids[index]),
        int(data.token_positions[index]),
    )
    selected = min((int(index) for index in eligible), key=key)
    metadata = {
        "selection_rule": (
            "closest token to the empirical medians of K=8 selection, complete-route, "
            "and delta-2 selection coverage; require 2-20 delta-2 misses so both hit and "
            "miss mechanics remain visible; deterministic request/position tie-break"
        ),
        "token_index": selected,
        "request_id": int(data.request_ids[selected]),
        "sample_id": data.sample_ids[selected],
        "domain": data.domains[selected],
        "phase": data.phases[selected],
        "token_position": int(data.token_positions[selected]),
        "k8_selection_coverage_delta_1_to_3": float(per_token_selection[selected]),
        "k8_complete_route_coverage_delta_1_to_3": float(
            per_token_complete[selected]
        ),
        "k8_selection_coverage_delta_2": float(per_token_delta_two[selected]),
        "delta_2_misses": int(delta_two_misses[selected]),
        "population_medians": {
            "selection_delta_1_to_3": float(target[0]),
            "complete_delta_1_to_3": float(target[1]),
            "selection_delta_2": float(target[2]),
        },
    }
    return selected, metadata


def _pair_lookup(
    source_pairs: np.ndarray, target_pairs: np.ndarray
) -> dict[tuple[int, int], int]:
    return {
        (int(source), int(target)): index
        for index, (source, target) in enumerate(
            zip(source_pairs, target_pairs, strict=True)
        )
    }


def _resolved_matrix(
    token_index: int,
    route_ids: np.ndarray,
    ranks: np.ndarray,
    pair_lookup: dict[tuple[int, int], int],
    *,
    delta: int,
    candidate_count: int,
    revealed_through: int = LAYERS - 1,
) -> np.ndarray:
    """Encode predicted-only, hit, miss, and unscored actual cells."""

    matrix = np.zeros((EXPERTS, LAYERS), dtype=np.uint8)
    for target in range(min(revealed_through, LAYERS - 1) + 1):
        actual = route_ids[token_index, target]
        if target < delta:
            matrix[actual, target] = 4
            continue
        pair = pair_lookup[(target - delta, target)]
        predicted = ranks[token_index, pair, :candidate_count]
        matrix[predicted, target] = 1
        actual_set = {int(value) for value in actual}
        predicted_set = {int(value) for value in predicted}
        for expert in actual_set:
            matrix[expert, target] = 2 if expert in predicted_set else 3
    return matrix


def _coverage(
    token_index: int,
    route_ids: np.ndarray,
    ranks: np.ndarray,
    pair_lookup: dict[tuple[int, int], int],
    *,
    delta: int,
    candidate_count: int,
    revealed_through: int = LAYERS - 1,
) -> dict[str, int | float]:
    hits = 0
    routes_complete = 0
    forecasts = 0
    for target in range(delta, min(revealed_through, LAYERS - 1) + 1):
        pair = pair_lookup[(target - delta, target)]
        predicted = {
            int(value) for value in ranks[token_index, pair, :candidate_count]
        }
        actual = [int(value) for value in route_ids[token_index, target]]
        covered = [expert in predicted for expert in actual]
        hits += sum(covered)
        routes_complete += all(covered)
        forecasts += 1
    selections = forecasts * TOP_K
    return {
        "forecasts": forecasts,
        "hits": hits,
        "misses": selections - hits,
        "selection_coverage": hits / selections if selections else 0.0,
        "complete_routes": routes_complete,
        "complete_route_coverage": routes_complete / forecasts if forecasts else 0.0,
        "candidate_slots": forecasts * candidate_count,
    }


def _draw_heatmap(axis: plt.Axes, matrix: np.ndarray) -> None:
    axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="none",
        cmap=CELL_CMAP,
        norm=CELL_NORM,
        extent=(-0.5, LAYERS - 0.5, -0.5, EXPERTS - 0.5),
    )
    axis.set_xlim(-0.5, LAYERS - 0.5)
    axis.set_ylim(-0.5, EXPERTS - 0.5)
    axis.set_xticks(np.arange(0, LAYERS, 2))
    axis.set_yticks(np.arange(0, EXPERTS, 4))
    axis.set_xticks(np.arange(-0.5, LAYERS, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, EXPERTS, 1), minor=True)
    axis.grid(which="minor", color=GRID, linewidth=0.22, alpha=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#aab2bd")
        spine.set_linewidth(0.7)


def _legend(include_pending: bool = False) -> list[Patch]:
    items = [
        Patch(facecolor=HIT, label="cache hit: actual ∩ predicted"),
        Patch(facecolor=MISS, label="cache miss: actual only"),
        Patch(facecolor=PREDICTED, label="prefetched, not used"),
        Patch(facecolor=UNSCORED, label="actual; no earlier forecast"),
    ]
    if include_pending:
        items.insert(0, Patch(facecolor=PENDING, label="pending lookahead prediction"))
    return items


def plot_horizon_map(
    token_index: int,
    metadata: dict[str, Any],
    route_ids: np.ndarray,
    ranks: np.ndarray,
    pair_lookup: dict[tuple[int, int], int],
) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharex=True, sharey=True)
    metrics: list[dict[str, Any]] = []
    for axis, delta in zip(axes, (1, 2, 3), strict=True):
        matrix = _resolved_matrix(
            token_index,
            route_ids,
            ranks,
            pair_lookup,
            delta=delta,
            candidate_count=PRIMARY_K,
        )
        values = _coverage(
            token_index,
            route_ids,
            ranks,
            pair_lookup,
            delta=delta,
            candidate_count=PRIMARY_K,
        )
        metrics.append({"delta": delta, "candidate_count": PRIMARY_K, **values})
        _draw_heatmap(axis, matrix)
        axis.set_title(
            f"Δ={delta}: predict layer N+{delta}\n"
            f"{values['hits']}/{values['forecasts'] * TOP_K} expert demands hit "
            f"({100 * values['selection_coverage']:.1f}%)",
            fontsize=11,
            color=TEXT,
        )
        axis.set_xlabel("target layer", color=TEXT)
    axes[0].set_ylabel("expert ID", color=TEXT)
    fig.suptitle(
        "One decoder token: frozen K=8 lookahead predictions versus actual routing",
        fontsize=15,
        color=TEXT,
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        f"Fresh confirmation • {metadata['domain']} • {metadata['sample_id']} • "
        f"sequence position {metadata['token_position']} (decode)",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.legend(
        handles=_legend(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=9,
    )
    fig.text(
        0.5,
        0.065,
        "Each target layer actually selects 4 experts; the predictor nominates 8. "
        "Blue cells are bandwidth/capacity cost, not errors in coverage.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.025, 0.11, 1, 0.88), w_pad=1.0)
    fig.savefig(OUTPUT / "fig1_single_token_horizons.png", dpi=300)
    fig.savefig(OUTPUT / "fig1_single_token_horizons.pdf")
    plt.close(fig)
    return metrics


def plot_candidate_amplification(
    token_index: int,
    metadata: dict[str, Any],
    route_ids: np.ndarray,
    ranks: np.ndarray,
    pair_lookup: dict[tuple[int, int], int],
) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), sharex=True, sharey=True)
    metrics: list[dict[str, Any]] = []
    for axis, candidate_count in zip(axes.flat, (4, 8, 12, 16), strict=True):
        matrix = _resolved_matrix(
            token_index,
            route_ids,
            ranks,
            pair_lookup,
            delta=ANIMATION_DELTA,
            candidate_count=candidate_count,
        )
        values = _coverage(
            token_index,
            route_ids,
            ranks,
            pair_lookup,
            delta=ANIMATION_DELTA,
            candidate_count=candidate_count,
        )
        metrics.append(
            {
                "delta": ANIMATION_DELTA,
                "candidate_count": candidate_count,
                **values,
            }
        )
        _draw_heatmap(axis, matrix)
        axis.set_title(
            f"K={candidate_count}  ({candidate_count / TOP_K:.0f}× candidates)\n"
            f"hit {100 * values['selection_coverage']:.1f}% • "
            f"complete {values['complete_routes']}/{values['forecasts']} routes",
            fontsize=11,
            color=TEXT,
        )
    for axis in axes[-1]:
        axis.set_xlabel("target layer", color=TEXT)
    for axis in axes[:, 0]:
        axis.set_ylabel("expert ID", color=TEXT)
    fig.suptitle(
        "Candidate amplification trades more prefetched experts for fewer misses",
        fontsize=15,
        color=TEXT,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        f"Same real token, fixed Δ=2 • {metadata['domain']} • "
        f"sequence position {metadata['token_position']} (decode)",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.legend(
        handles=_legend(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=9,
    )
    fig.text(
        0.5,
        0.055,
        "More blue removes red, but every extra blue cell represents an expert nominated "
        "for residency or transfer and never used by this token.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.025, 0.09, 1, 0.91), h_pad=2.0, w_pad=1.2)
    fig.savefig(OUTPUT / "fig3_candidate_amplification.png", dpi=300)
    fig.savefig(OUTPUT / "fig3_candidate_amplification.pdf")
    plt.close(fig)
    return metrics


def animate_lookahead(
    token_index: int,
    metadata: dict[str, Any],
    route_ids: np.ndarray,
    ranks: np.ndarray,
    pair_lookup: dict[tuple[int, int], int],
) -> None:
    fig, axis = plt.subplots(figsize=(10.8, 8.2))
    frames = [0, 0, *range(LAYERS), LAYERS - 1, LAYERS - 1, LAYERS - 1]

    def draw(source: int) -> None:
        axis.clear()
        matrix = _resolved_matrix(
            token_index,
            route_ids,
            ranks,
            pair_lookup,
            delta=ANIMATION_DELTA,
            candidate_count=PRIMARY_K,
            revealed_through=source,
        )
        target = source + ANIMATION_DELTA
        if target < LAYERS:
            pair = pair_lookup[(source, target)]
            predicted = ranks[token_index, pair, :PRIMARY_K]
            matrix[predicted, target] = 5
        _draw_heatmap(axis, matrix)
        axis.add_patch(
            Rectangle(
                (source - 0.5, -0.5),
                1,
                EXPERTS,
                fill=False,
                edgecolor=CURRENT,
                linewidth=2.4,
                clip_on=False,
            )
        )
        axis.set_xlabel("model layer", color=TEXT)
        axis.set_ylabel("expert ID", color=TEXT)
        resolved = _coverage(
            token_index,
            route_ids,
            ranks,
            pair_lookup,
            delta=ANIMATION_DELTA,
            candidate_count=PRIMARY_K,
            revealed_through=source,
        )
        if target < LAYERS:
            action = f"Layer {source} route observed  →  prefetch K=8 for layer {target}"
        else:
            action = f"Layer {source} route observed  →  no Δ=2 target remains"
        if resolved["forecasts"]:
            cumulative = (
                f"resolved so far: {resolved['hits']} hits / "
                f"{resolved['forecasts'] * TOP_K} demands "
                f"({100 * resolved['selection_coverage']:.1f}%)"
            )
        else:
            cumulative = "no lookahead prediction has resolved yet"
        axis.set_title(
            f"{action}\n{cumulative}",
            fontsize=12,
            color=TEXT,
            pad=11,
        )
        if target < LAYERS:
            axis.add_patch(
                Rectangle(
                    (target - 0.5, -0.5),
                    1,
                    EXPERTS,
                    fill=False,
                    edgecolor=PENDING,
                    linewidth=2.4,
                    linestyle="--",
                    clip_on=False,
                )
            )
        axis.text(
            source,
            -2.3,
            "current",
            ha="center",
            va="top",
            fontsize=8,
            color=CURRENT,
            fontweight="bold",
            clip_on=False,
        )
        if target < LAYERS:
            axis.text(
                target,
                -2.3,
                "prefetch",
                ha="center",
                va="top",
                fontsize=8,
                color=PENDING,
                fontweight="bold",
                clip_on=False,
            )

    animation = FuncAnimation(fig, draw, frames=frames, interval=520, repeat=True)
    fig.suptitle(
        "Lookahead demand prediction becomes cache hit or cache miss",
        fontsize=15,
        color=TEXT,
        y=0.985,
    )
    fig.text(
        0.5,
        0.94,
        f"One fresh decoder token • fixed Δ=2, K=8 • {metadata['sample_id']} • "
        f"sequence position {metadata['token_position']}",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.legend(
        handles=_legend(include_pending=True),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.10, 1, 0.91))
    animation.save(
        OUTPUT / "fig2_lookahead_cache_dynamics.gif",
        writer=PillowWriter(fps=2),
        dpi=115,
    )
    plt.close(fig)


def _panel(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    edgecolor: str = "#c7ced8",
    facecolor: str = "#fbfbfa",
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.1,
        )
    )


def _expert_tile(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    expert: int,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    textcolor: str = TEXT,
    suffix: str = "",
    linewidth: float = 0.8,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=facecolor,
            edgecolor=edgecolor or facecolor,
            linewidth=linewidth,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        f"E{expert}{suffix}",
        ha="center",
        va="center",
        fontsize=8.3,
        color=textcolor,
        fontweight="bold" if facecolor in (HIT, MISS, PENDING) else "normal",
    )


def _draw_current_route(
    axis: plt.Axes,
    x: float,
    route_ids: np.ndarray,
    route_weights: np.ndarray,
) -> None:
    for row, (expert, weight) in enumerate(
        zip(route_ids, route_weights, strict=True)
    ):
        y = 0.625 - row * 0.092
        _expert_tile(
            axis,
            x,
            y,
            0.105,
            0.066,
            int(expert),
            facecolor="#fff4d6",
            edgecolor=CURRENT,
        )
        bar_width = 0.096 * min(max(float(weight), 0.0), 1.0)
        axis.add_patch(
            Rectangle(
                (x + 0.0045, y + 0.006),
                bar_width,
                0.008,
                facecolor=CURRENT,
                edgecolor="none",
                alpha=0.9,
            )
        )
        axis.text(
            x + 0.101,
            y + 0.012,
            f"{100 * float(weight):.0f}%",
            ha="right",
            va="bottom",
            fontsize=6.2,
            color="#805b00",
        )


def _draw_candidate_column(
    axis: plt.Axes,
    x: float,
    candidates: np.ndarray | None,
    *,
    target: int,
    delta: int,
) -> None:
    axis.text(
        x + 0.055,
        0.745,
        f"N+{delta}\nlayer {target}" if candidates is not None else f"N+{delta}\n—",
        ha="center",
        va="center",
        fontsize=9.3,
        color=TEXT if candidates is not None else "#9aa3af",
        fontweight="bold",
    )
    if candidates is None:
        axis.text(
            x + 0.055,
            0.50,
            "outside\nmodel",
            ha="center",
            va="center",
            fontsize=9,
            color="#9aa3af",
        )
        return
    for rank, expert in enumerate(candidates):
        column = rank % 2
        row = rank // 2
        _expert_tile(
            axis,
            x + column * 0.058,
            0.625 - row * 0.092,
            0.052,
            0.066,
            int(expert),
            facecolor=PENDING,
            textcolor="white",
        )


def _delta_two_batch(
    current: int,
    token_index: int,
    route_ids: np.ndarray,
    ranks: np.ndarray,
    pair_lookup: dict[tuple[int, int], int],
) -> tuple[np.ndarray, set[int], set[int]] | None:
    if current < ANIMATION_DELTA:
        return None
    pair = pair_lookup[(current - ANIMATION_DELTA, current)]
    predicted = ranks[token_index, pair, :PRIMARY_K]
    actual = {int(value) for value in route_ids[token_index, current]}
    predicted_set = {int(value) for value in predicted}
    return predicted, actual & predicted_set, actual - predicted_set


def animate_forecast_cone_cache_ledger(
    token_index: int,
    metadata: dict[str, Any],
    route_ids: np.ndarray,
    route_weights: np.ndarray,
    ranks: np.ndarray,
    pair_lookup: dict[tuple[int, int], int],
) -> None:
    """Show new Δ=1..3 forecasts while resolving a causal Δ=2 cache batch."""

    fig, axis = plt.subplots(figsize=(14.0, 7.9))
    frame_layers: list[int] = [0, 0]
    for layer in range(LAYERS):
        frame_layers.append(layer)
        batch = _delta_two_batch(layer, token_index, route_ids, ranks, pair_lookup)
        if batch is not None and batch[2]:
            frame_layers.append(layer)  # Pause on misses.
    frame_layers.extend((LAYERS - 1, LAYERS - 1, LAYERS - 1))

    def draw(current: int) -> None:
        axis.clear()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        _panel(axis, 0.018, 0.17, 0.645, 0.73)
        _panel(axis, 0.685, 0.17, 0.297, 0.73, edgecolor="#9fb3c8")

        axis.text(
            0.04,
            0.862,
            "FORECAST CONE",
            fontsize=12,
            fontweight="bold",
            color=TEXT,
            va="center",
        )
        axis.text(
            0.04,
            0.826,
            f"Observe layer {current}; nominate K=8 experts at each future layer",
            fontsize=9.3,
            color="#4b5563",
            va="center",
        )
        axis.text(
            0.073,
            0.745,
            f"NOW\nlayer {current}",
            ha="center",
            va="center",
            fontsize=9.3,
            color="#805b00",
            fontweight="bold",
        )
        _draw_current_route(
            axis,
            0.021,
            route_ids[token_index, current],
            route_weights[token_index, current],
        )

        forecast_x = (0.205, 0.355, 0.505)
        for x, delta in zip(forecast_x, (1, 2, 3), strict=True):
            target = current + delta
            candidates: np.ndarray | None = None
            if target < LAYERS:
                pair = pair_lookup[(current, target)]
                candidates = ranks[token_index, pair, :PRIMARY_K]
            _draw_candidate_column(
                axis, x, candidates, target=target, delta=delta
            )
            if candidates is not None:
                axis.add_patch(
                    FancyArrowPatch(
                        (0.135, 0.707),
                        (x + 0.047, 0.707),
                        arrowstyle="-|>",
                        mutation_scale=11,
                        linewidth=1.0,
                        color=PENDING,
                        alpha=0.65,
                        connectionstyle=f"arc3,rad={0.08 * (delta - 2)}",
                    )
                )
        axis.text(
            0.342,
            0.235,
            "Future routes remain hidden. Each blue tile is a speculative expert "
            "residency/transfer candidate.",
            ha="center",
            va="center",
            fontsize=8.8,
            color="#4b5563",
        )

        axis.text(
            0.708,
            0.862,
            "CACHE LEDGER",
            fontsize=12,
            fontweight="bold",
            color=TEXT,
            va="center",
        )
        axis.text(
            0.708,
            0.826,
            "Track one causal batch: fixed Δ=2",
            fontsize=9.3,
            color="#4b5563",
            va="center",
        )
        batch = _delta_two_batch(
            current, token_index, route_ids, ranks, pair_lookup
        )
        if batch is None:
            axis.text(
                0.834,
                0.60,
                "WARM-UP",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=UNSCORED,
            )
            axis.text(
                0.834,
                0.53,
                "No layer N−2 forecast\nis due yet.",
                ha="center",
                va="center",
                fontsize=10,
                color="#4b5563",
            )
        else:
            predicted, hits, misses = batch
            axis.text(
                0.834,
                0.774,
                f"issued at L{current - ANIMATION_DELTA}  →  resolved at L{current}",
                ha="center",
                va="center",
                fontsize=9.5,
                color=TEXT,
                fontweight="bold",
            )
            for rank, expert_value in enumerate(predicted):
                expert = int(expert_value)
                column = rank % 2
                row = rank // 2
                is_hit = expert in hits
                _expert_tile(
                    axis,
                    0.716 + column * 0.126,
                    0.66 - row * 0.083,
                    0.112,
                    0.057,
                    expert,
                    facecolor=HIT if is_hit else PREDICTED,
                    textcolor="white" if is_hit else TEXT,
                    suffix="  hit" if is_hit else "",
                )
            miss_text = "  ".join(f"E{expert}" for expert in sorted(misses))
            if misses:
                axis.add_patch(
                    FancyBboxPatch(
                        (0.716, 0.285),
                        0.238,
                        0.055,
                        boxstyle="round,pad=0.004,rounding_size=0.007",
                        facecolor=MISS,
                        edgecolor=MISS,
                    )
                )
                axis.text(
                    0.835,
                    0.312,
                    f"late demand: {miss_text}  MISS",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white",
                    fontweight="bold",
                )
            else:
                axis.text(
                    0.835,
                    0.312,
                    "complete route covered — no late demand",
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    color=HIT,
                    fontweight="bold",
                )

        cumulative = _coverage(
            token_index,
            route_ids,
            ranks,
            pair_lookup,
            delta=ANIMATION_DELTA,
            candidate_count=PRIMARY_K,
            revealed_through=current,
        )
        bar_x, bar_y, bar_w, bar_h = 0.715, 0.215, 0.24, 0.024
        axis.add_patch(
            Rectangle(
                (bar_x, bar_y),
                bar_w,
                bar_h,
                facecolor="#e5e7eb",
                edgecolor="none",
            )
        )
        if cumulative["forecasts"]:
            hit_width = bar_w * float(cumulative["selection_coverage"])
            axis.add_patch(
                Rectangle(
                    (bar_x, bar_y),
                    hit_width,
                    bar_h,
                    facecolor=HIT,
                    edgecolor="none",
                )
            )
            axis.add_patch(
                Rectangle(
                    (bar_x + hit_width, bar_y),
                    bar_w - hit_width,
                    bar_h,
                    facecolor=MISS,
                    edgecolor="none",
                )
            )
            progress = (
                f"cumulative Δ=2: {cumulative['hits']}/"
                f"{cumulative['forecasts'] * TOP_K} demands covered "
                f"({100 * cumulative['selection_coverage']:.1f}%)"
            )
        else:
            progress = "cumulative Δ=2: waiting for first resolution"
        axis.text(
            0.835,
            0.255,
            progress,
            ha="center",
            va="center",
            fontsize=8.2,
            color=TEXT,
        )
        axis.text(
            0.835,
            0.19,
            "Blue = fetched but unused • green = used • red = absent when demanded",
            ha="center",
            va="center",
            fontsize=7.4,
            color="#4b5563",
        )

    animation = FuncAnimation(
        fig, draw, frames=frame_layers, interval=650, repeat=True
    )
    fig.suptitle(
        "Forecast now, reconcile later: one token’s expert-demand trajectory",
        fontsize=16,
        color=TEXT,
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        f"Frozen route heads • {metadata['sample_id']} • sequence position "
        f"{metadata['token_position']} (decode)",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#fff4d6", edgecolor=CURRENT, label="observed route"),
            Patch(facecolor=PENDING, label="pending prediction"),
            Patch(facecolor=HIT, label="covered demand"),
            Patch(facecolor=MISS, label="missed demand"),
            Patch(facecolor=PREDICTED, label="prefetched, unused"),
        ],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=9,
    )
    fig.text(
        0.5,
        0.012,
        "Conceptual cache ledger: candidate membership only; no capacity, eviction, "
        "transfer-completion, or latency simulation.",
        ha="center",
        fontsize=8.3,
        color="#6b7280",
    )
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.91))
    animation.save(
        OUTPUT / "fig4_forecast_cone_cache_ledger.gif",
        writer=PillowWriter(fps=2),
        dpi=105,
    )
    plt.close(fig)


def _write_manifest(
    metadata: dict[str, Any],
    horizon_metrics: list[dict[str, Any]],
    amplification_metrics: list[dict[str, Any]],
) -> None:
    artifacts = {
        str(path): sha256_file(path)
        for path in sorted(OUTPUT.glob("fig*"))
        if path.is_file()
    }
    write_json(
        OUTPUT / "visualization_manifest.json",
        {
            "schema_version": 1,
            "source": {
                "confirmation_run": str(CONFIRMATION),
                "confirmation_artifact_manifest_sha256": sha256_file(
                    CONFIRMATION / "artifact_manifest.json"
                ),
                "predictor_checkpoint": str(CHECKPOINT),
                "predictor_checkpoint_sha256": sha256_file(CHECKPOINT),
                "predictor_config": str(PREDICTOR_CONFIG),
                "predictor_config_sha256": sha256_file(PREDICTOR_CONFIG),
            },
            "semantics": {
                "actual_experts_per_layer": TOP_K,
                "primary_candidate_count": PRIMARY_K,
                "animation_delta": ANIMATION_DELTA,
                "forecast_cone_deltas": [1, 2, 3],
                "cache_ledger_delta": ANIMATION_DELTA,
                "cache_ledger_scope": (
                    "resolves each delta-2 prediction batch independently; overlapping "
                    "delta-1 and delta-3 sets are not treated as one cache policy"
                ),
                "cache_hit_definition": (
                    "actual selected expert belongs to the earlier K-candidate prediction"
                ),
                "cache_miss_definition": (
                    "actual selected expert does not belong to the earlier K-candidate prediction"
                ),
                "caveat": (
                    "cache hit/miss is a demand-coverage visualization; it does not simulate "
                    "capacity, eviction, transfer completion, or measured latency"
                ),
            },
            "representative_token": metadata,
            "horizon_metrics": horizon_metrics,
            "candidate_amplification_metrics": amplification_metrics,
            "artifacts": artifacts,
        },
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with PREDICTOR_CONFIG.open("rb") as handle:
        config = tomllib.load(handle)
    config = {**config, "device": "cpu"}
    tokens, requests = _load_routes(CONFIRMATION)
    decode = [token for token in tokens if token.phase == "decode"]
    data = compact_routes(decode, set(requests), LAYERS)
    source_pairs, _target_pairs = layer_pairs(LAYERS, list(range(1, LAYERS)))
    model = PairwiseRouteHeads(len(source_pairs), feature_width=64)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint["config_sha256"] != sha256_file(PREDICTOR_CONFIG):
        raise ValueError("predictor checkpoint and frozen config disagree")
    model.load_state_dict(checkpoint["state_dict"])
    scores, source_np, target_np = predict_pairwise_scores(
        model, data, checkpoint["feature_mode"], config
    )
    ranks = np.argsort(-scores, axis=2, kind="stable")[:, :, :16].astype(np.uint8)
    hits = _candidate_hits(ranks[:, :, :PRIMARY_K], data.route_ids.numpy(), target_np)
    token_index, metadata = _choose_representative_token(
        hits, source_np, target_np, data
    )
    lookup = _pair_lookup(source_np, target_np)
    horizon_metrics = plot_horizon_map(
        token_index, metadata, data.route_ids.numpy(), ranks, lookup
    )
    animate_lookahead(token_index, metadata, data.route_ids.numpy(), ranks, lookup)
    amplification_metrics = plot_candidate_amplification(
        token_index, metadata, data.route_ids.numpy(), ranks, lookup
    )
    animate_forecast_cone_cache_ledger(
        token_index,
        metadata,
        data.route_ids.numpy(),
        data.route_weights.numpy(),
        ranks,
        lookup,
    )
    _write_manifest(metadata, horizon_metrics, amplification_metrics)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
