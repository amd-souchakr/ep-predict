from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


BLUE = "#3266A8"
ORANGE = "#D97732"
GRID = "#D9DEE7"
TEXT = "#20242B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#7A828E",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "legend.frameon": False,
        }
    )


def _save(figure: Any, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "ep-predict extended-horizon visualization"},
    )
    return [png, pdf]


def _mean_series(
    summaries: list[dict[str, str]],
    *,
    metric: str,
) -> dict[str, list[float]]:
    result = {"transition": [], "linear": []}
    for delta in range(1, 16):
        for policy in result:
            matches = [
                row
                for row in summaries
                if row["phase"] == "decode"
                and row["domain"] == "__domain_balanced__"
                and int(row["capacity"]) == 16
                and int(row["delta"]) == delta
                and row["baseline"] == policy
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"missing extended summary for {policy}, delta={delta}"
                )
            result[policy].append(100 * float(matches[0][metric]))
    return result


def _source_layer_points(
    metrics: list[dict[str, str]],
    *,
    metric: str,
) -> dict[tuple[str, int], list[float]]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in metrics:
        if (
            row["phase"] != "decode"
            or int(row["capacity"]) != 16
            or row["baseline"] not in {"transition", "linear"}
        ):
            continue
        grouped[
            (
                row["baseline"],
                int(row["delta"]),
                int(row["source_layer"]),
            )
        ].append(100 * float(row[metric]))
    result: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (policy, delta, _source), values in sorted(grouped.items()):
        result[(policy, delta)].append(statistics.fmean(values))
    return result


def _plot_horizon(
    summaries: list[dict[str, str]],
    metrics: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    deltas = list(range(1, 16))
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.76, wspace=0.22)
    panels = (
        (
            "mean_selection_coverage",
            "selection_coverage",
            "Future experts found",
            "Selection coverage (%)",
        ),
        (
            "mean_complete_token_coverage",
            "complete_token_coverage",
            "Entire top-8 route found",
            "Complete-token coverage (%)",
        ),
    )
    for axis, (summary_metric, scope_metric, title, ylabel) in zip(
        axes, panels, strict=True
    ):
        means = _mean_series(summaries, metric=summary_metric)
        points = _source_layer_points(metrics, metric=scope_metric)
        all_values: list[float] = []
        for policy, color, offset in (
            ("transition", BLUE, -0.07),
            ("linear", ORANGE, 0.07),
        ):
            for delta in deltas:
                values = points[(policy, delta)]
                all_values.extend(values)
                axis.scatter(
                    [delta + offset] * len(values),
                    values,
                    s=13,
                    alpha=0.20,
                    color=color,
                    edgecolors="none",
                    zorder=1,
                )
            axis.plot(
                deltas,
                means[policy],
                marker="o",
                markersize=4.5,
                linewidth=2.2,
                color=color,
                label=(
                    "Transition table" if policy == "transition" else "Linear sidecar"
                ),
                zorder=3,
            )
        lower = max(0, math.floor((min(all_values) - 5) / 5) * 5)
        upper = min(100, math.ceil((max(all_values) + 5) / 5) * 5)
        axis.set_ylim(lower, upper)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            deltas,
            [f"{delta}\n{16 - delta}" for delta in deltas],
        )
        axis.set_xlabel("Lookahead Δ\nEligible source layers")
        axis.grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].legend(loc="best")
    figure.suptitle(
        "Prediction remains informative across the full 16-layer model",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.90,
        "Decode, K=16. Faint points are source-layer means; lines average all "
        "eligible layer-domain scopes.",
        fontsize=9.5,
        color="#555E6B",
    )
    return _save(figure, output_dir / "fig1_extended_horizon_coverage")


def _gain_matrix(
    metrics: list[dict[str, str]],
    *,
    metric: str,
) -> Any:
    import numpy as np

    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in metrics:
        if (
            row["phase"] != "decode"
            or int(row["capacity"]) != 16
            or row["baseline"] not in {"transition", "linear"}
        ):
            continue
        grouped[
            (
                int(row["source_layer"]),
                int(row["target_layer"]),
                row["baseline"],
            )
        ].append(float(row[metric]))
    matrix = np.full((16, 16), np.nan, dtype=float)
    for source in range(16):
        for target in range(source + 1, 16):
            linear = grouped[(source, target, "linear")]
            transition = grouped[(source, target, "transition")]
            if not linear or not transition:
                raise ValueError(f"missing source-target scope {source}->{target}")
            matrix[source, target] = 100 * (
                statistics.fmean(linear) - statistics.fmean(transition)
            )
    return matrix


def _plot_heatmaps(
    metrics: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    matrices = (
        (
            _gain_matrix(metrics, metric="selection_coverage"),
            "Future-expert selection gain",
        ),
        (
            _gain_matrix(metrics, metric="complete_token_coverage"),
            "Complete top-8 gain",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    figure.subplots_adjust(left=0.07, right=0.94, bottom=0.12, top=0.78, wspace=0.26)
    colormap = plt.get_cmap("RdBu_r").with_extremes(bad="#F1F3F5")
    for axis, (matrix, title) in zip(axes, matrices, strict=True):
        finite = np.abs(matrix[np.isfinite(matrix)])
        limit = max(5.0, math.ceil(float(finite.max()) / 5) * 5)
        image = axis.imshow(
            matrix,
            origin="upper",
            cmap=colormap,
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Future target layer")
        axis.set_ylabel("Source layer")
        axis.set_xticks(range(16))
        axis.set_yticks(range(16))
        axis.tick_params(labelsize=7.5)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label("Linear − transition (percentage points)")
    figure.suptitle(
        "Hidden-state gains depend strongly on the source-target layer pair",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.90,
        "Decode, K=16, domain-balanced. Gray cells are invalid "
        "(target is not later than source).",
        fontsize=9.5,
        color="#555E6B",
    )
    return _save(figure, output_dir / "fig2_source_target_gain_heatmap")


def plot_extended_horizon(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    run = Path(run_dir)
    output_name = str(
        experiment_config.get("analysis", {}).get(
            "output_name", "h23_extended_horizon"
        )
    )
    analysis_dir = run / "analysis" / output_name
    summary_path = analysis_dir / "summary.csv"
    metrics_path = analysis_dir / "metrics.csv"
    gate_path = analysis_dir / "gate.json"
    for path in (summary_path, metrics_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(f"run extended analysis before plotting: {path}")
    destination = (
        Path(output_dir) if output_dir is not None else analysis_dir / "figures"
    )
    destination.mkdir(parents=True, exist_ok=True)
    summaries = _read_csv(summary_path)
    metrics = _read_csv(metrics_path)

    outputs: list[Path] = []
    outputs.extend(_plot_horizon(summaries, metrics, destination))
    plt.close("all")
    outputs.extend(_plot_heatmaps(metrics, destination))
    plt.close("all")

    selection = _mean_series(summaries, metric="mean_selection_coverage")
    complete = _mean_series(summaries, metric="mean_complete_token_coverage")
    note = destination / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# Extended-horizon figure review",
                "",
                "This is post-hoc descriptive evidence and does not alter the "
                "formal H2/H3 gates.",
                "",
                "## Automated headline",
                "",
                "At decode K=16, transition selection coverage changes from "
                f"{selection['transition'][0]:.1f}% at Δ=1 to "
                f"{selection['transition'][-1]:.1f}% at Δ=15, while linear "
                f"changes from {selection['linear'][0]:.1f}% to "
                f"{selection['linear'][-1]:.1f}%. Complete-token coverage "
                f"changes from {complete['transition'][0]:.1f}% to "
                f"{complete['transition'][-1]:.1f}% for transition and "
                f"{complete['linear'][0]:.1f}% to "
                f"{complete['linear'][-1]:.1f}% for linear.",
                "",
                "The Δ=15 point contains only layer 0→15; use the heatmap before "
                "attributing the aggregate trend to distance alone.",
                "",
                "## Human review checklist",
                "",
                "- [ ] Horizon means are interpreted with the changing number "
                "of eligible source layers.",
                "- [ ] Heatmap cells agree with source/target layer semantics.",
                "- [ ] Selection and complete-token coverage are not conflated.",
                "- [ ] Long-horizon outliers and layer clusters are recorded.",
                "- [ ] One next action is recorded before H4.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(note)
    manifest = {
        "analysis": "h23_extended_horizon",
        "status": "post_hoc_descriptive",
        "inputs": {
            str(path): _sha256(path)
            for path in (summary_path, metrics_path, gate_path)
        },
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
