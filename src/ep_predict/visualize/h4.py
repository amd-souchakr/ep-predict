from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


BLUE = "#3266A8"
ORANGE = "#D97732"
GREEN = "#2A8C72"
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
            "axes.titlesize": 11.5,
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
        metadata={"Creator": "ep-predict H4 visualization"},
    )
    return [png, pdf]


def _lookup(
    rows: list[dict[str, str]],
    *,
    capacity: int,
    lookahead: int,
    bandwidth: float,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if int(row["capacity"]) == capacity
        and int(row["lookahead"]) == lookahead
        and float(row["bandwidth_scale"]) == bandwidth
    ]
    if len(matches) != 1:
        raise ValueError(
            f"missing H4 cell K={capacity}, Δ={lookahead}, BW={bandwidth}"
        )
    return matches[0]


def _plot_heatmap(
    rows: list[dict[str, str]],
    capacities: list[int],
    lookaheads: list[int],
    bandwidths: list[float],
    output: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(
        1,
        len(capacities),
        figsize=(4.0 * len(capacities), 4.55),
        sharex=True,
        sharey=True,
    )
    if len(capacities) == 1:
        axes = [axes]
    figure.subplots_adjust(
        left=0.11, right=0.91, bottom=0.21, top=0.72, wspace=0.12
    )
    image = None
    for axis, capacity in zip(axes, capacities, strict=True):
        matrix = np.array(
            [
                [
                    100
                    * float(
                        _lookup(
                            rows,
                            capacity=capacity,
                            lookahead=delta,
                            bandwidth=bandwidth,
                        )["oracle_stall_reduction"]
                    )
                    for delta in lookaheads
                ]
                for bandwidth in bandwidths
            ]
        )
        image = axis.imshow(
            matrix,
            origin="lower",
            vmin=0,
            vmax=100,
            cmap="YlGnBu",
            aspect="auto",
            interpolation="nearest",
        )
        resident_fraction = 100 * capacity / 64
        axis.set_title(
            f"Keep {capacity} experts on GPU\n({resident_fraction:g}% per layer)",
            fontweight="bold",
        )
        axis.set_xticks(range(len(lookaheads)), lookaheads)
        bandwidth_labels = {
            0.5: "Half measured",
            1.0: "Measured",
            2.0: "Double measured",
        }
        axis.set_yticks(
            range(len(bandwidths)),
            [
                bandwidth_labels.get(bandwidth, f"{bandwidth:g}× measured")
                for bandwidth in bandwidths
            ],
        )
        axis.set_xlabel("Advance notice (layers before use)")
        for row_index, bandwidth in enumerate(bandwidths):
            for column_index, delta in enumerate(lookaheads):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= 58 else TEXT,
                )
        if capacity == 16 and 1.0 in bandwidths:
            from matplotlib.patches import Rectangle

            primary_columns = [
                index for index, delta in enumerate(lookaheads) if delta <= 3
            ]
            if primary_columns:
                row_index = bandwidths.index(1.0)
                start = min(primary_columns)
                end = max(primary_columns)
                axis.add_patch(
                    Rectangle(
                        (start - 0.5, row_index - 0.5),
                        end - start + 1,
                        1,
                        fill=False,
                        edgecolor=TEXT,
                        linewidth=2.4,
                    )
                )
    axes[0].set_ylabel("Host-to-GPU copy speed")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Transfer waiting removed (%)")
    figure.suptitle(
        "Faster copies, more GPU capacity, and earlier notice all reduce waiting",
        x=0.03,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.03,
        0.85,
        "Each cell is the percentage of transfer-induced waiting removed with "
        "perfect knowledge of future expert use. Higher is better.",
        fontsize=9.5,
        color="#555E6B",
    )
    figure.text(
        0.03,
        0.07,
        "Black outline: the formal decision region (16 resident experts, "
        "measured copy speed, 1–3 layers of notice).",
        fontsize=9,
        color="#555E6B",
    )
    return _save(figure, output / "fig1_h4_oracle_feasibility_heatmap")


def _plot_curve(
    rows: list[dict[str, str]],
    capacities: list[int],
    lookaheads: list[int],
    output: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.24, top=0.70, wspace=0.16
    )
    metrics = (
        (
            "deadline_feasible_cold_fraction",
            "Needed expert data arriving on time",
        ),
        ("oracle_stall_reduction", "Transfer waiting removed"),
    )
    styles = tuple(
        zip(capacities, (ORANGE, BLUE, GREEN), ("o", "s", "^"), strict=True)
    )
    handles = []
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for capacity, color, marker in styles:
            values = [
                100
                * float(
                    _lookup(
                        rows,
                        capacity=capacity,
                        lookahead=delta,
                        bandwidth=1.0,
                    )[metric]
                )
                for delta in lookaheads
            ]
            (line,) = axis.plot(
                lookaheads,
                values,
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=5.5,
                label=(
                    f"{capacity} experts on GPU "
                    f"({100 * capacity / 64:g}% per layer)"
                ),
            )
            if axis is axes[0]:
                handles.append(line)
        axis.axhline(
            50,
            color="#596273",
            linestyle="--",
            linewidth=1.3,
        )
        axis.axvspan(1, 3, color="#DCE7F5", alpha=0.45, zorder=0)
        axis.text(
            1.08,
            52.5,
            "Required: 50%",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#596273",
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(lookaheads)
        axis.set_ylim(0, 103)
        axis.set_xlabel("Advance notice (layers before use)")
        axis.grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].set_ylabel("Percent (%)")
    figure.suptitle(
        "At the measured link, more notice and more GPU capacity both help",
        x=0.03,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.03,
        0.84,
        "Left: does required data arrive before use?  Right: how much waiting "
        "does advance notice eliminate?  Shading marks the 1–3 layer test range.",
        fontsize=9.5,
        color="#555E6B",
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=len(capacities),
        frameon=False,
    )
    figure.text(
        0.03,
        0.015,
        "All points use the measured MI355X copy rate. Longer notice covers "
        "fewer destination layers; 15 layers means only the first-to-last pair.",
        fontsize=8.7,
        color="#555E6B",
    )
    return _save(figure, output / "fig2_h4_stall_reduction_curve")


def plot_h4(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    analysis = Path(experiment_config["output_dir"])
    metrics_path = analysis / "oracle_metrics.csv"
    gate_path = analysis / "gate.json"
    measurement_path = analysis / "measurement.json"
    for path in (metrics_path, gate_path, measurement_path):
        if not path.is_file():
            raise FileNotFoundError(f"run H4 analysis before plotting: {path}")
    destination = (
        Path(output_dir) if output_dir is not None else analysis / "figures"
    )
    destination.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(metrics_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    simulation = experiment_config["simulation"]
    capacities = [int(value) for value in simulation["capacities"]]
    lookaheads = [int(value) for value in simulation["lookaheads"]]
    bandwidths = [float(value) for value in simulation["bandwidth_scales"]]

    outputs: list[Path] = []
    outputs.extend(
        _plot_heatmap(rows, capacities, lookaheads, bandwidths, destination)
    )
    plt.close("all")
    outputs.extend(_plot_curve(rows, capacities, lookaheads, destination))
    plt.close("all")

    best = gate["best_primary_row"]
    note = destination / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# H4 figure review",
                "",
                "## Automated headline",
                "",
                f"Formal decision: `{gate['decision']}`. With 16 experts kept "
                f"on GPU per layer and the measured copy speed, the best "
                f"tested advance notice is {best['lookahead']} layers: "
                f"{100 * best['deadline_feasible_cold_fraction']:.1f}% "
                "of required cold data arrives on time and "
                f"{100 * best['oracle_stall_reduction']:.1f}% of transfer "
                "waiting is removed.",
                "",
                "## Human review checklist",
                "",
                "- [ ] The grid reads as copy speed × advance notice × experts "
                "kept on GPU without requiring H4 terminology.",
                "- [ ] The black-outlined 16-expert, measured-speed, 1–3 layer "
                "cells agree with `gate.json`.",
                "- [ ] The two-panel chart keeps on-time data distinct from "
                "waiting removed.",
                "- [ ] Every capacity and lookahead remains visible in both "
                "measured-link trends.",
                "- [ ] The effective-average layer-time approximation and "
                "single-copy-engine limitation are accepted.",
                "- [ ] The testbed forward time is not presented as an inherent "
                "MI355X hardware characteristic.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(note)
    manifest = {
        "analysis": "h4",
        "decision": gate["decision"],
        "inputs": {
            str(path): _sha256(path)
            for path in (metrics_path, gate_path, measurement_path)
        },
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
