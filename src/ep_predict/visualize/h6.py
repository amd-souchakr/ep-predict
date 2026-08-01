from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


TEXT = "#20242B"
GRID = "#D9DEE7"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.8,
            "axes.edgecolor": "#7A828E",
            "axes.linewidth": 0.8,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
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
        metadata={"Creator": "ep-predict H6 residency replay"},
    )
    return [png, pdf]


def _heatmap_values(
    rows: list[dict[str, str]],
    *,
    phase: str,
    capacity: int,
) -> tuple[Any, dict[tuple[int, int], str]]:
    import numpy as np

    scoped: dict[
        tuple[str, int, int], dict[str, dict[str, str]]
    ] = {}
    for row in rows:
        if row["phase"] != phase or int(row["capacity"]) != capacity:
            continue
        key = (
            row["domain"],
            int(row["source_layer"]),
            int(row["delta"]),
        )
        scoped.setdefault(key, {})[row["policy"]] = row

    values: dict[tuple[int, int], list[float]] = {}
    winners: dict[tuple[int, int], list[str]] = {}
    for (_domain, source, delta), policies in scoped.items():
        required = {"static", "domain", "lru", "transition", "linear"}
        if not required <= set(policies):
            continue
        reference = max(
            float(policies[name]["complete_resident_set_hit_coverage"])
            for name in ("static", "domain", "lru")
        )
        guided = {
            name: float(
                policies[name]["complete_resident_set_hit_coverage"]
            )
            for name in ("transition", "linear")
        }
        winner = max(guided, key=guided.get)
        values.setdefault((source, delta), []).append(guided[winner] - reference)
        winners.setdefault((source, delta), []).append(winner)

    matrix = np.full((15, 15), np.nan, dtype=np.float64)
    labels: dict[tuple[int, int], str] = {}
    for (source, delta), samples in values.items():
        matrix[source, delta - 1] = 100 * float(np.mean(samples))
        votes = winners[(source, delta)]
        labels[(source, delta)] = (
            "L" if votes.count("linear") >= votes.count("transition") else "T"
        )
    return matrix, labels


def _plot_gain_heatmap(
    *,
    rows: list[dict[str, str]],
    output: Path,
    capacities: list[int],
    primary_delta: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    matrices = [
        _heatmap_values(rows, phase="decode", capacity=capacity)[0]
        for capacity in capacities
    ]
    finite = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix in matrices]
    )
    limit = max(2.0, float(np.quantile(np.abs(finite), 0.98)))
    figure, axes = plt.subplots(
        1,
        len(capacities),
        figsize=(12.4, 4.7),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.065, right=0.91, bottom=0.17, top=0.80, wspace=0.08
    )
    image = None
    for axis, capacity, matrix in zip(
        axes, capacities, matrices, strict=True
    ):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        axis.axvline(
            primary_delta - 1,
            color="#252A31",
            linewidth=0.9,
            linestyle="--",
            alpha=0.75,
        )
        axis.set_title(f"Fast tier K={capacity}", loc="left", fontweight="bold")
        axis.set_xlabel("Lookahead Δ (layers)")
        axis.set_xticks([0, 2, 5, 8, 11, 14])
        axis.set_xticklabels([1, 3, 6, 9, 12, 15])
        axis.set_yticks([0, 3, 6, 9, 12, 14])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Source MoE layer")
    color_axis = figure.add_axes([0.925, 0.23, 0.014, 0.47])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(
        "Best guided gain over\nbest static/domain/LRU (pp)",
        rotation=90,
        labelpad=10,
    )
    figure.suptitle(
        "Where prediction-guided residency changes complete-set hits",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.065,
        0.895,
        "Held-out decode · complete resident-set hit coverage · "
        "one demanded-miss insertion allowed per wave",
        fontsize=9.7,
        color="#59616D",
    )
    figure.text(
        0.065,
        0.045,
        "Blue: prediction helps. Red: a simple baseline is better. "
        f"Dashed line: preregistered Δ={primary_delta}. "
        "Blank cells have no valid target layer.",
        fontsize=8.7,
        color="#59616D",
    )
    return _save(figure, output / "fig1_h6_residency_gain_heatmap")


def plot_h6(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    analysis = Path(str(experiment_config["output_dir"]))
    output = Path(output_dir) if output_dir else analysis / "figures"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = analysis / "scope_metrics.csv"
    gate_path = analysis / "gate.json"
    summary_path = analysis / "summary.csv"
    rows = _read_csv(metrics_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    capacities = [int(value) for value in experiment_config["replay"]["capacities"]]
    outputs = _plot_gain_heatmap(
        rows=rows,
        output=output,
        capacities=capacities,
        primary_delta=int(gate["primary_scope"]["lookahead"]),
    )
    plt.close("all")

    input_paths = [metrics_path, summary_path, gate_path]
    manifest = {
        "hypothesis": "H6",
        "human_review_complete": False,
        "figure_semantics": (
            "Domain-balanced complete resident-set hit gain of the better "
            "transition/linear residency policy over the strongest matched "
            "static/domain/LRU baseline."
        ),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in input_paths
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in outputs
        ],
    }
    write_json(output / "figure_manifest.json", manifest)
    review = [
        "# H6 figure review",
        "",
        "- [ ] Confirm phase, capacity, source-layer, and lookahead semantics.",
        "- [ ] Confirm each cell compares against the strongest matched "
        "static/domain/LRU baseline.",
        "- [ ] Confirm the dashed Δ=3 column matches the frozen gate.",
        "- [ ] Inspect whether positive cells form a broad layer/horizon regime "
        "or isolated artifacts.",
        "- [ ] Compare the visual pattern with `gate.json` and `REPORT.md`.",
        "- [ ] Record accept/reject interpretation and one next action.",
        "",
        "**Human review complete:** no",
        "",
    ]
    (output / "FIGURES.md").write_text("\n".join(review), encoding="utf-8")
    manifest["outputs"].extend(
        [
            {
                "path": str(output / "figure_manifest.json"),
                "sha256": _sha256(output / "figure_manifest.json"),
            },
            {
                "path": str(output / "FIGURES.md"),
                "sha256": _sha256(output / "FIGURES.md"),
            },
        ]
    )
    return manifest
