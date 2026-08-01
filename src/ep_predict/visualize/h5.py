from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


TEXT = "#20242B"
GRID = "#D9DEE7"
CAPACITY_COLORS = {8: "#D97732", 16: "#3266A8", 32: "#2A8C72"}
CELL_COLORS = {
    "short_physics_control": "#8657A6",
    "short_boundary": "#2A8C72",
    "long_linear": "#3266A8",
    "prediction_control": "#D97732",
}


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
            "font.size": 9.3,
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.8,
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
        metadata={"Creator": "ep-predict H5 first-order analysis"},
    )
    return [png, pdf]


def _profit_mask(
    x: Any,
    y: Any,
    *,
    amplification: float,
    min_stall: float,
    min_recovery: float,
    max_amplification: float,
) -> Any:
    import numpy as np

    oracle = np.minimum(1.0, x)
    benefit = np.minimum(y, x / amplification)
    return (
        (benefit >= min_stall)
        & (benefit / oracle >= min_recovery)
        & (amplification <= max_amplification)
    )


def _plot_phase_diagram(
    *,
    output: Path,
    policy_rows: list[dict[str, str]],
    gate: dict[str, Any],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    min_stall = float(gate["min_modeled_stall_reduction"])
    min_recovery = float(gate["min_oracle_recovery"])
    max_amplification = float(gate["max_predicted_to_useful_bytes"])
    amplifications = [1.0, 2.0, 4.0]
    figure, axes = plt.subplots(
        1, 4, figsize=(13.4, 4.7), sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1.15]},
    )
    figure.subplots_adjust(
        left=0.065, right=0.99, bottom=0.24, top=0.74, wspace=0.15
    )
    x_values = np.geomspace(0.08, 30.0, 450)
    y_values = np.linspace(0.0, 1.0, 350)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    cmap = ListedColormap(["#F1E6E3", "#DDEFE4"])
    for axis, amplification in zip(axes[:3], amplifications, strict=True):
        mask = _profit_mask(
            x_grid,
            y_grid,
            amplification=amplification,
            min_stall=min_stall,
            min_recovery=min_recovery,
            max_amplification=max_amplification,
        )
        axis.pcolormesh(
            x_values,
            100 * y_values,
            mask.astype(int),
            cmap=cmap,
            shading="auto",
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        axis.contour(
            x_grid,
            100 * y_grid,
            mask.astype(float),
            levels=[0.5],
            colors=["#356B4D"],
            linewidths=1.3,
        ) if mask.any() and (~mask).any() else None
        axis.axhline(25, color="#6A7280", linestyle=":", linewidth=0.9)
        axis.set_title(
            f"A = {amplification:g}×"
            + ("  (traffic gate fails)" if amplification > max_amplification else ""),
            loc="left",
            fontweight="bold",
        )
        axis.text(
            0.11,
            93,
            "Analytically\nprofitable" if mask.any() else "No profitable\nwindow",
            fontsize=8.2,
            color="#356B4D" if mask.any() else "#8A4C45",
            va="top",
        )
        axis.set_xlabel("Cold-service headroom H")

    actual = axes[3]
    actual.set_facecolor("#F5F6F8")
    grouped_actual: dict[str, list[dict[str, str]]] = {}
    for row in policy_rows:
        grouped_actual.setdefault(row["cell"], []).append(row)
        headroom = float(row["cold_service_headroom"])
        coverage = 100 * float(row["complete_cold_set_coverage"])
        policy = row["policy"]
        marker = "o" if policy == "linear" else "s"
        passed = row["profitable"].lower() == "true"
        actual.scatter(
            headroom,
            coverage,
            s=56,
            marker=marker,
            facecolor=CELL_COLORS[row["cell"]] if passed else "white",
            edgecolor=CELL_COLORS[row["cell"]],
            linewidth=1.6,
            zorder=3,
        )
    label_offsets = {
        "short_physics_control": (-58, -3),
        "short_boundary": (-22, 20),
        "long_linear": (7, -5),
        "prediction_control": (7, -7),
    }
    for cell, rows in grouped_actual.items():
        x_value = float(rows[0]["cold_service_headroom"])
        y_value = max(100 * float(row["complete_cold_set_coverage"]) for row in rows)
        amplifications = [
            float(row["candidate_transfer_amplification"]) for row in rows
        ]
        label = (
            f"K={rows[0]['capacity']}, Δ={rows[0]['lookahead']}\n"
            f"A={min(amplifications):.1f}–{max(amplifications):.1f}×"
        )
        actual.annotate(
            label,
            (x_value, y_value),
            xytext=label_offsets[cell],
            textcoords="offset points",
            fontsize=7.3,
            color=CELL_COLORS[cell],
        )
    actual.axhline(25, color="#6A7280", linestyle=":", linewidth=0.9)
    actual.set_title("Existing policies", loc="left", fontweight="bold")
    actual.set_xlabel("Cold-service headroom H")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(0.08, 30)
        axis.set_ylim(0, 100)
        axis.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8, 16])
        axis.set_xticklabels([".1", ".25", ".5", "1", "2", "4", "8", "16"])
        axis.grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].set_ylabel("Complete cold-set coverage C (%)")
    policy_handles = [
        Line2D(
            [0], [0], marker="s", linestyle="none", color="#596273",
            markerfacecolor="white", label="Transition",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", color="#596273",
            markerfacecolor="white", label="Linear",
        ),
    ]
    figure.legend(
        handles=policy_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        ncol=2,
        fontsize=8.4,
    )
    figure.suptitle(
        "A profitable window exists—but the current candidate streams over-transfer",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.7,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.875,
        "Trace-driven cold demand; first-order proportional service. "
        "Green satisfies ≥25% stall reduction, ≥50% oracle recovery, and A≤2×. "
        "This is an analytical screen, not measured speedup.",
        fontsize=9.2,
        color="#555E6B",
    )
    return _save(figure, output / "fig1_h5_profitability_phase_diagram")


def _plot_inverse(
    *,
    output: Path,
    inverse_rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    amplifications = [1.0, 2.0]
    bandwidths = [0.5, 1.0, 2.0]
    line_styles = {0.5: ":", 1.0: "-", 2.0: "--"}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharey=True)
    figure.subplots_adjust(
        left=0.08, right=0.985, bottom=0.25, top=0.73, wspace=0.14
    )
    for axis, amplification in zip(axes, amplifications, strict=True):
        for capacity in (8, 16, 32):
            for bandwidth in bandwidths:
                selected = sorted(
                    (
                        row
                        for row in inverse_rows
                        if int(row["capacity"]) == capacity
                        and float(row["bandwidth_scale"]) == bandwidth
                        and float(row["candidate_transfer_amplification"])
                        == amplification
                    ),
                    key=lambda row: int(row["lookahead"]),
                )
                x = [int(row["lookahead"]) for row in selected]
                y = [
                    (
                        100 * float(row["minimum_complete_cold_set_coverage"])
                        if row["minimum_complete_cold_set_coverage"]
                        else float("nan")
                    )
                    for row in selected
                ]
                axis.plot(
                    x,
                    y,
                    color=CAPACITY_COLORS[capacity],
                    linestyle=line_styles[bandwidth],
                    linewidth=1.65,
                )
        axis.set_xlim(1, 15)
        axis.set_ylim(20, 54)
        axis.set_xticks([1, 3, 6, 9, 12, 15])
        axis.set_xlabel("Lookahead Δ (MoE layers)")
        axis.set_title(
            f"Candidate transfer amplification A={amplification:g}×",
            loc="left",
            fontweight="bold",
        )
        axis.grid(color=GRID, linewidth=0.7)
    axes[0].set_ylabel("Minimum complete cold-set coverage (%)")
    capacity_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=f"K={capacity}")
        for capacity, color in CAPACITY_COLORS.items()
    ]
    bandwidth_handles = [
        Line2D(
            [0], [0], color="#596273", linestyle=line_styles[bandwidth],
            linewidth=1.7, label=f"{bandwidth:g}× BW",
        )
        for bandwidth in bandwidths
    ]
    figure.legend(
        handles=capacity_handles + bandwidth_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        fontsize=8.4,
    )
    figure.suptitle(
        "Longer lookahead turns impossible cells into a 25–50% coverage target",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.7,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.865,
        "Missing line segments are empty windows: even 100% complete prediction "
        "cannot cross the frozen benefit gate at that physical headroom.",
        fontsize=9.2,
        color="#555E6B",
    )
    return _save(figure, output / "fig2_h5_inverse_prediction_requirement")


def plot_h5(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    _style()
    analysis = Path(experiment_config["output_dir"])
    inputs = [
        analysis / "h5_design_points.csv",
        analysis / "h5_inverse_requirements.csv",
        analysis / "h5_policy_placement.csv",
        analysis / "summary.json",
        analysis / "gate.json",
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"analyze H5 before plotting: {path}")
    output = Path(output_dir) if output_dir else analysis / "figures"
    output.mkdir(parents=True, exist_ok=True)
    inverse = _read_csv(analysis / "h5_inverse_requirements.csv")
    policies = _read_csv(analysis / "h5_policy_placement.csv")
    gate = json.loads((analysis / "gate.json").read_text(encoding="utf-8"))
    outputs = []
    outputs.extend(
        _plot_phase_diagram(
            output=output,
            policy_rows=policies,
            gate=gate["thresholds"],
        )
    )
    outputs.extend(_plot_inverse(output=output, inverse_rows=inverse))
    note = output / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# H5 figure review",
                "",
                "The figures are trace-driven first-order analytical screens. "
                "They are not end-to-end latency measurements.",
                "",
                "## Human review checklist",
                "",
                "- [ ] Complete cold-set coverage excludes already-hot waves "
                "and is not confused with selection coverage.",
                "- [ ] A is transferred candidate bytes per useful predicted "
                "cold byte after resident filtering.",
                "- [ ] The green region is read as analytically profitable "
                "under the frozen proxy, not demonstrated speedup.",
                "- [ ] Empty inverse-curve segments are recognized as physical "
                "failure, not missing data.",
                "- [ ] Actual policy values agree with "
                "`h5_policy_placement.csv`.",
                "- [ ] One next action is recorded before H6 or any new setup.",
                "",
                "## One next action",
                "",
                "Pending researcher review.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(note)
    manifest = {
        "analysis": "h5_first_order_codesign",
        "evidence_grade": "trace_driven_analytical_pilot",
        "inputs": {str(path): _sha256(path) for path in inputs},
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
