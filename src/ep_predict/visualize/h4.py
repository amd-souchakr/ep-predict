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
        figsize=(3.75 * len(capacities), 4.15),
        sharex=True,
        sharey=True,
    )
    if len(capacities) == 1:
        axes = [axes]
    figure.subplots_adjust(
        left=0.08, right=0.93, bottom=0.18, top=0.75, wspace=0.12
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
        axis.set_title(f"Fast tier K={capacity}", fontweight="bold")
        axis.set_xticks(range(len(lookaheads)), lookaheads)
        axis.set_yticks(
            range(len(bandwidths)),
            [f"{bandwidth:g}×" for bandwidth in bandwidths],
        )
        axis.set_xlabel("Same-token lookahead Δ")
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
    axes[0].set_ylabel("Effective H2D bandwidth")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Oracle cold-stall reduction (%)")
    figure.suptitle(
        "Perfect knowledge exposes the whole-expert feasibility region",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.87,
        "Decode, one serialized copy engine, exact 12 MiB experts. "
        "Numbers are reduction versus reactive loading.",
        fontsize=9.5,
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

    figure, axis = plt.subplots(figsize=(7.6, 4.5))
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.17, top=0.76)
    for capacity, color, marker in zip(
        capacities,
        (ORANGE, BLUE, GREEN),
        ("o", "s", "^"),
        strict=True,
    ):
        values = [
            100
            * float(
                _lookup(
                    rows,
                    capacity=capacity,
                    lookahead=delta,
                    bandwidth=1.0,
                )["oracle_stall_reduction"]
            )
            for delta in lookaheads
        ]
        axis.plot(
            lookaheads,
            values,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=5,
            label=f"K={capacity}",
        )
    axis.axhline(
        50,
        color="#596273",
        linestyle="--",
        linewidth=1.2,
        label="Primary 50% gate",
    )
    axis.axvspan(1, 3, color="#DCE7F5", alpha=0.45, zorder=0)
    axis.text(
        2,
        4,
        "Primary horizons",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#596273",
    )
    axis.set_xticks(lookaheads)
    axis.set_ylim(0, 103)
    axis.set_xlabel("Same-token lookahead Δ")
    axis.set_ylabel("Oracle cold-stall reduction (%)")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.legend(loc="lower right", ncol=2)
    figure.suptitle(
        "More residency and lookahead jointly reduce cold-expert stall",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.88,
        "Measured host-to-device rate; synchronous decode waves. "
        "Long horizons cover fewer target layers (Δ=15 is L0→L15 only) and "
        "do not alter the Δ=1–3 gate.",
        fontsize=9.5,
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
                f"Formal decision: `{gate['decision']}`. The best frozen "
                f"K=16, measured-bandwidth short horizon is Δ="
                f"{best['lookahead']}: "
                f"{100 * best['deadline_feasible_cold_fraction']:.1f}% "
                "deadline-feasible cold bytes and "
                f"{100 * best['oracle_stall_reduction']:.1f}% oracle stall "
                "reduction.",
                "",
                "## Human review checklist",
                "",
                "- [ ] Heatmap axes, bandwidth multipliers, capacities, and "
                "12 MiB semantics are correct.",
                "- [ ] The K=16, Δ=1–3 cells agree with `gate.json`.",
                "- [ ] Deadline-feasible bytes are not conflated with resident "
                "hit bytes.",
                "- [ ] Any saturation, capacity threshold, or non-monotonic "
                "lookahead behavior is recorded.",
                "- [ ] The effective-average layer-time approximation and "
                "single-copy-engine limitation are accepted.",
                "- [ ] One next action is recorded before H5.",
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
