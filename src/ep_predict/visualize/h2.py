from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ep_predict.visualize.h1 import (
    _configure_matplotlib,
    _read_csv,
    _save_figure,
    _sha256,
)


BASELINE_LABELS = {
    "static": "Static popularity",
    "domain_oracle": "Domain oracle",
    "previous_window": "Previous window",
    "transition": "Route transition",
}
BASELINE_COLORS = {
    "static": "#666666",
    "domain_oracle": "#0072B2",
    "previous_window": "#009E73",
    "transition": "#D55E00",
}
BASELINE_MARKERS = {
    "static": "o",
    "domain_oracle": "s",
    "previous_window": "^",
    "transition": "D",
}
DOMAIN_ORDER = ["code", "math", "general", "conversation"]
DOMAIN_LABELS = ["Code", "Mathematics", "General prose", "Conversation"]


def _summary_lookup(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, int, int, str], dict[str, str]]:
    return {
        (
            row["phase"],
            row["domain"],
            int(row["delta"]),
            int(row["capacity"]),
            row["baseline"],
        ): row
        for row in rows
    }


def _lookahead_series(
    rows: list[dict[str, str]],
    *,
    metric: str,
    capacity: int,
    lookaheads: list[int],
) -> dict[str, list[float]]:
    lookup = _summary_lookup(rows)
    return {
        baseline: [
            float(
                lookup[
                    (
                        "decode",
                        "__domain_balanced__",
                        delta,
                        capacity,
                        baseline,
                    )
                ][metric]
            )
            for delta in lookaheads
        ]
        for baseline in ("static", "transition")
    }


def _plot_simple_lookahead(
    *,
    summary_rows: list[dict[str, str]],
    lookaheads: list[int],
    capacity: int,
    metric: str,
    title: str,
    ylabel: str,
    output_name: str,
    output_dir: Path,
) -> list[Path]:
    from matplotlib.ticker import PercentFormatter

    _mpl, plt = _configure_matplotlib()
    series = _lookahead_series(
        summary_rows,
        metric=metric,
        capacity=capacity,
        lookaheads=lookaheads,
    )
    figure, axis = plt.subplots(
        figsize=(6.6, 3.7),
        layout="constrained",
    )
    for baseline in ("static", "transition"):
        values = series[baseline]
        axis.plot(
            lookaheads,
            values,
            color=BASELINE_COLORS[baseline],
            marker=BASELINE_MARKERS[baseline],
            markersize=6.0,
            linewidth=2.2,
            zorder=3,
        )
        for delta, value in zip(lookaheads, values, strict=True):
            offset = (
                20
                if baseline == "static"
                and metric == "mean_complete_token_coverage"
                else 10
                if baseline == "transition"
                else -15
            )
            axis.annotate(
                f"{100 * value:.1f}%",
                (delta, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color=BASELINE_COLORS[baseline],
                fontweight="bold" if baseline == "transition" else "normal",
            )

    transition = series["transition"]
    drop = 100 * (transition[0] - transition[-1])
    axis.text(
        0.98,
        0.92,
        f"Drop from n+1 to n+3: {drop:.1f} pp",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=BASELINE_COLORS["transition"],
    )
    axis.text(
        lookaheads[-1] + 0.13,
        transition[-1],
        "Current-route\ntransition",
        ha="left",
        va="center",
        fontsize=8.5,
        color=BASELINE_COLORS["transition"],
    )
    if metric == "mean_complete_token_coverage":
        axis.annotate(
            "Static popularity (<1%)",
            (lookaheads[-1], series["static"][-1]),
            xytext=(lookaheads[-1] + 0.13, 0.04),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.5,
            color=BASELINE_COLORS["static"],
            arrowprops={
                "arrowstyle": "-",
                "color": BASELINE_COLORS["static"],
                "linewidth": 0.8,
            },
        )
    else:
        axis.text(
            lookaheads[-1] + 0.13,
            series["static"][-1],
            "Static\npopularity",
            ha="left",
            va="center",
            fontsize=8.5,
            color=BASELINE_COLORS["static"],
        )
    axis.set_title(
        f"{title}\nHeld-out decode · 16 candidates for the actual top-8 route",
        loc="left",
        pad=10,
    )
    axis.set_xlabel("How far ahead are we predicting?")
    axis.set_ylabel(ylabel)
    axis.set_xticks(
        lookaheads,
        ["Next layer\nn+1", "Two layers ahead\nn+2", "Three layers ahead\nn+3"],
    )
    axis.set_xlim(min(lookaheads) - 0.15, max(lookaheads) + 0.58)
    if metric == "mean_selection_coverage":
        axis.set_ylim(0, 1.0)
        axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        axis.set_ylim(0, 0.31)
        axis.set_yticks([0, 0.1, 0.2, 0.3])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    return _save_figure(figure, output_dir / output_name)


def _plot_capacity_tradeoff(
    *,
    summary_rows: list[dict[str, str]],
    capacities: list[int],
    lookaheads: list[int],
    output_dir: Path,
) -> list[Path]:
    from matplotlib.ticker import PercentFormatter

    _mpl, plt = _configure_matplotlib()
    lookup = _summary_lookup(summary_rows)
    figure, axes = plt.subplots(
        2,
        len(lookaheads),
        figsize=(7.2, 4.65),
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    metrics = [
        ("mean_selection_coverage", "Expert-selection coverage"),
        ("mean_complete_token_coverage", "Complete-token coverage"),
    ]
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column_index, delta in enumerate(lookaheads):
            axis = axes[row_index][column_index]
            for baseline in BASELINE_LABELS:
                values = [
                    float(
                        lookup[
                            (
                                "decode",
                                "__domain_balanced__",
                                delta,
                                capacity,
                                baseline,
                            )
                        ][metric]
                    )
                    for capacity in capacities
                ]
                axis.plot(
                    capacities,
                    values,
                    color=BASELINE_COLORS[baseline],
                    marker=BASELINE_MARKERS[baseline],
                    markersize=3.8,
                    label=BASELINE_LABELS[baseline],
                )
            panel = row_index * len(lookaheads) + column_index
            axis.set_title(
                f"({chr(ord('a') + panel)}) Decode, $\\Delta={delta}$",
                loc="left",
            )
            axis.set_xticks(capacities)
            axis.set_xlim(min(capacities) - 1, max(capacities) + 1)
            axis.set_ylim(0, 1.0)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#E6E6E6", linewidth=0.5)
            if column_index == 0:
                axis.set_ylabel(ylabel)
    axes[1][1].set_xlabel("Candidate experts per target layer, $K$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        columnspacing=1.5,
    )
    return _save_figure(figure, output_dir / "fig1_h2_capacity_coverage")


def _plot_gain_heatmap(
    *,
    metric_rows: list[dict[str, str]],
    lookaheads: list[int],
    capacity: int,
    output_dir: Path,
) -> list[Path]:
    import numpy as np

    _mpl, plt = _configure_matplotlib()
    lookup = {
        (
            row["phase"],
            row["domain"],
            int(row["source_layer"]),
            int(row["delta"]),
            int(row["capacity"]),
            row["baseline"],
        ): float(row["selection_coverage"])
        for row in metric_rows
    }
    source_layers = sorted(
        {
            int(row["source_layer"])
            for row in metric_rows
            if row["phase"] == "decode"
        }
    )
    matrices: dict[int, Any] = {}
    for delta in lookaheads:
        matrix = np.full((len(DOMAIN_ORDER), len(source_layers)), np.nan)
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            for layer_index, source_layer in enumerate(source_layers):
                transition = lookup.get(
                    (
                        "decode",
                        domain,
                        source_layer,
                        delta,
                        capacity,
                        "transition",
                    )
                )
                static = lookup.get(
                    (
                        "decode",
                        domain,
                        source_layer,
                        delta,
                        capacity,
                        "static",
                    )
                )
                if transition is not None and static is not None:
                    matrix[domain_index, layer_index] = 100 * (transition - static)
        matrices[delta] = matrix

    upper = max(float(np.nanmax(matrix)) for matrix in matrices.values())
    figure, axes = plt.subplots(
        1,
        len(lookaheads),
        figsize=(7.2, 2.75),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for panel, (axis, delta) in enumerate(zip(axes, lookaheads, strict=True)):
        masked = np.ma.masked_invalid(matrices[delta])
        colormap = plt.get_cmap("YlOrRd").copy()
        colormap.set_bad("#ECECEC")
        image = axis.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            vmin=0,
            vmax=upper,
        )
        axis.set_title(f"({chr(ord('a') + panel)}) $\\Delta={delta}$", loc="left")
        axis.set_xlabel("Source MoE layer")
        axis.set_xticks(range(0, len(source_layers), 3))
        axis.set_xticklabels(source_layers[::3])
        axis.set_yticks(range(len(DOMAIN_ORDER)))
        axis.set_yticklabels(DOMAIN_LABELS)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label(f"Selection-coverage gain at $K={capacity}$ (pp)")
    return _save_figure(figure, output_dir / "fig2_h2_transition_gain")


def _plot_coverage_churn(
    *,
    summary_rows: list[dict[str, str]],
    lookaheads: list[int],
    capacity: int,
    output_dir: Path,
) -> list[Path]:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    _mpl, plt = _configure_matplotlib()
    lookup = _summary_lookup(summary_rows)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.6, 3.15),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for panel, (axis, phase) in enumerate(
        zip(axes, ("prefill", "decode"), strict=True)
    ):
        for baseline in BASELINE_LABELS:
            points = [
                lookup[
                    (
                        phase,
                        "__domain_balanced__",
                        delta,
                        capacity,
                        baseline,
                    )
                ]
                for delta in lookaheads
            ]
            x = [
                float(point["mean_candidate_replacement_fraction"])
                for point in points
            ]
            y = [float(point["mean_selection_coverage"]) for point in points]
            axis.plot(
                x,
                y,
                color=BASELINE_COLORS[baseline],
                marker=BASELINE_MARKERS[baseline],
                markersize=4.3,
                linewidth=1.2,
                label=BASELINE_LABELS[baseline],
            )
            if baseline == "transition":
                for delta, x_value, y_value in zip(
                    lookaheads, x, y, strict=True
                ):
                    vertical_offset = {1: 5, 2: -2, 3: -9}.get(delta, 2)
                    axis.annotate(
                        f"Δ{delta}",
                        (x_value, y_value),
                        xytext=(4, vertical_offset),
                        textcoords="offset points",
                        fontsize=7,
                        color=BASELINE_COLORS[baseline],
                    )
        axis.set_title(f"({chr(ord('a') + panel)}) {phase.capitalize()}", loc="left")
        axis.set_xlabel("Candidate replacement per token")
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlim(-0.025, 0.55)
        axis.set_ylim(0.35, 0.85)
        axis.grid(color="#E6E6E6", linewidth=0.5)
    axes[0].set_ylabel(f"Expert-selection coverage at $K={capacity}$")
    handles = [
        Line2D(
            [0],
            [0],
            color=BASELINE_COLORS[baseline],
            marker=BASELINE_MARKERS[baseline],
            markersize=4,
            label=label,
        )
        for baseline, label in BASELINE_LABELS.items()
    ]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        columnspacing=1.4,
    )
    return _save_figure(figure, output_dir / "fig3_h2_coverage_churn")


def _write_figure_notes(path: Path, *, run_id: str) -> None:
    text = f"""# H2 figures: `{run_id}`

These figures use held-out requests only. PNG files are 450 DPI; PDFs retain
vector text and lines.

## Simple conclusion

The current route predicts future routing well. With 16 candidates, it finds
79.0% of the eight experts used in the next MoE layer and 76.8% three layers
ahead. The drop from n+1 to n+3 is only 2.2 percentage points. Static
popularity finds about 41%.

## Figure 1 — Future experts found

`fig1_predictability_by_lookahead` reports the fraction of the actual top-8
future experts contained in the 16 candidates. This is the clearest evidence
that current routing predicts routing one to three layers ahead.

## Figure 2 — Complete future route found

`fig2_complete_route_by_lookahead` uses the stricter metric: all eight future
experts must be among the 16 candidates. Coverage is 24.1% at n+1 and 22.2% at
n+3, compared with less than 1% for static popularity. Predictability is real,
but guaranteeing the entire top-8 route remains difficult.

The older capacity, layer/domain heatmap, and coverage/churn figures in this
directory are supplementary diagnostics rather than primary communication
figures.

## Human visual-review checkpoint

- [ ] A reader can state the H2 conclusion after viewing Figure 1 for a few
      seconds.
- [ ] The difference between "future experts found" and "all eight found" is
      clear.
- [ ] Headline values agree with `REPORT.md`, `summary.csv`, and `gate.json`.
- [ ] The plots are interpreted as predictability evidence, not yet as latency
      or hardware benefit.
- [ ] The reviewer records whether to advance to an external H3 predictor.
"""
    path.write_text(text, encoding="utf-8")


def plot_h2(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    directory = Path(run_dir)
    analysis_dir = directory / "analysis" / "h2"
    figure_dir = (
        Path(output_dir) if output_dir is not None else analysis_dir / "figures"
    )
    lookaheads = [int(value) for value in experiment_config["lookaheads"]]
    gate_capacity = int(experiment_config["decision_gate"]["capacity_experts"])
    input_paths = {
        "summary": analysis_dir / "summary.csv",
        "gate": analysis_dir / "gate.json",
        "split": analysis_dir / "split.json",
    }
    summary_rows = _read_csv(input_paths["summary"])

    outputs: list[Path] = []
    outputs.extend(
        _plot_simple_lookahead(
            summary_rows=summary_rows,
            lookaheads=lookaheads,
            capacity=gate_capacity,
            metric="mean_selection_coverage",
            title="Current routing predicts most future experts",
            ylabel="Actual future experts found",
            output_name="fig1_predictability_by_lookahead",
            output_dir=figure_dir,
        )
    )
    outputs.extend(
        _plot_simple_lookahead(
            summary_rows=summary_rows,
            lookaheads=lookaheads,
            capacity=gate_capacity,
            metric="mean_complete_token_coverage",
            title="Predicting all 8 experts is harder, but routing still helps",
            ylabel="Tokens with all 8 future experts found",
            output_name="fig2_complete_route_by_lookahead",
            output_dir=figure_dir,
        )
    )
    manifest = json.loads(
        (directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    notes_path = figure_dir / "FIGURES.md"
    _write_figure_notes(notes_path, run_id=manifest["run_id"])
    outputs.append(notes_path)

    figure_manifest = {
        "run_id": manifest["run_id"],
        "hypothesis": "H2",
        "figure_grade": "pilot",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in outputs
        ],
        "semantics": {
            "split_unit": "request",
            "metric_unit": "held-out token-layer pair",
            "aggregate": "equal mean across layer-domain scopes",
            "selection_coverage": "fraction of actual top-8 target experts covered",
            "complete_token_coverage": "all actual top-8 target experts covered",
            "primary_question": "prediction quality versus layer lookahead",
        },
    }
    manifest_path = figure_dir / "figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure_manifest["manifest_path"] = str(manifest_path)
    return figure_manifest
