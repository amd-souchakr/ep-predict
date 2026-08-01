from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


TEXT = "#20242B"
MUTED = "#5E6875"
GRID = "#D9DEE7"
BLUE = "#3266A8"
GREEN = "#2A8C72"
ORANGE = "#D97732"
PURPLE = "#8657A6"


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
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.7,
            "axes.edgecolor": "#7A828E",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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
        metadata={"Creator": "ep-predict AX architecture analysis"},
    )
    return [png, pdf]


def _phase_map(
    output: Path,
    phase_rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    amplifications = [1.0, 2.0, 4.0]
    x_values = np.geomspace(0.08, 64.0, 500)
    y_values = np.linspace(0.50, 1.0, 400)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    cmap = ListedColormap(["#F3E5E2", "#F5E7C8", "#DDEFE4"])
    figure, axes = plt.subplots(1, 3, figsize=(11.1, 4.25), sharey=True)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.76, wspace=0.12)

    marker_capacities = {8: "o", 16: "s", 32: "^"}
    for axis, amplification in zip(axes, amplifications, strict=True):
        effective = x_grid / amplification
        oracle = np.minimum(1.0, x_grid)
        benefit = y_grid * np.minimum(1.0, effective)
        recovery = benefit / oracle
        profitable = (benefit >= 0.25) & (recovery >= 0.50)
        slo = profitable & (y_grid >= 0.99) & (effective >= 1.25)
        category = profitable.astype(int) + slo.astype(int)
        axis.pcolormesh(
            x_values,
            100 * y_values,
            category,
            cmap=cmap,
            shading="auto",
            vmin=0,
            vmax=2,
            rasterized=True,
        )
        axis.contour(
            x_grid,
            100 * y_grid,
            effective,
            levels=[1.0],
            colors=["#8B6A2B"],
            linestyles=["--"],
            linewidths=1.0,
        )
        axis.axhline(99, color="#356B4D", linestyle=":", linewidth=1.0)
        markers = [
            row
            for row in phase_rows
            if float(row["amplification"]) == amplification
            and float(row["coverage"]) == 0.99
            and abs(float(row["bandwidth_gbps"]) - 24.1354255944) < 0.01
            and int(row["lookahead_layers"]) in {3, 9}
        ]
        for row in markers:
            capacity = int(row["capacity_experts_per_layer"])
            delta = int(row["lookahead_layers"])
            x = float(row["raw_cold_service_headroom"])
            axis.scatter(
                x,
                99,
                marker=marker_capacities[capacity],
                s=48,
                facecolor="white" if delta == 3 else TEXT,
                edgecolor=TEXT,
                linewidth=1.1,
                zorder=3,
            )
        axis.set_xscale("log")
        axis.set_xlim(0.08, 64)
        axis.set_ylim(50, 100.2)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.set_title(f"Amplification A = {amplification:g}×", loc="left", fontweight="bold")
        axis.set_xlabel("Raw cold-service headroom H")
    axes[0].set_ylabel("Complete cold-set coverage C (%)")
    figure.suptitle(
        "Prediction reliability and transfer headroom define the profitable region",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.865,
        "Green: P99-SLO candidate (C≥99%, H/A≥1.25). "
        "Amber: mean benefit but tail risk. Red: <25% stall reduction or <50% oracle recovery.",
        fontsize=9.0,
        color=MUTED,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor=TEXT, label="K=8"),
        Line2D([0], [0], marker="s", color="none", markeredgecolor=TEXT, label="K=16"),
        Line2D([0], [0], marker="^", color="none", markeredgecolor=TEXT, label="K=32"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=TEXT, markerfacecolor="white", label="Δ=3"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=TEXT, markerfacecolor=TEXT, label="Δ=9"),
    ]
    figure.legend(
        handles=handles,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        fontsize=8.3,
    )
    figure.text(
        0.985,
        0.025,
        "Markers: measured PCIe + trace demand; y=99% is assumed",
        ha="right",
        fontsize=7.5,
        color=MUTED,
    )
    return _save(figure, output / "fig1_profitability_phase_map")


def _pareto(
    output: Path,
    rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.20, top=0.77)
    styles = {
        "reactive_offload": (ORANGE, "o", "--", "Reactive offload"),
        "predictive_C99_A1.5": (BLUE, "s", "-", "Predictive C=99%, A=1.5×"),
        "predictive_C999_A1.25": (GREEN, "^", "-", "Predictive C=99.9%, A=1.25×"),
        "oracle_offload": (PURPLE, "D", ":", "Oracle C=100%, A=1×"),
    }
    for key, (color, marker, linestyle, label) in styles.items():
        if key == "reactive_offload":
            selected = [row for row in rows if row["policy"] == key]
        elif key == "oracle_offload":
            selected = [row for row in rows if row["policy"] == key]
        else:
            selected = [row for row in rows if row["quality_profile"] == key]
        selected.sort(key=lambda row: float(row["fast_tier_expert_gib"]))
        axis.plot(
            [float(row["fast_tier_expert_gib"]) for row in selected],
            [float(row["modeled_p99_tpot_ms"]) for row in selected],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=6,
            label=label,
        )
        if "predictive" in key:
            for row in selected:
                axis.annotate(
                    f"Δ={row['selected_lookahead']}",
                    (
                        float(row["fast_tier_expert_gib"]),
                        float(row["modeled_p99_tpot_ms"]),
                    ),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7.3,
                    color=color,
                )
    resident = next(row for row in rows if row["policy"] == "all_resident_reference")
    axis.scatter(
        float(resident["fast_tier_expert_gib"]),
        float(resident["modeled_p99_tpot_ms"]),
        marker="*",
        s=115,
        color=TEXT,
        label="All-resident measured reference",
        zorder=4,
    )
    axis.grid(color=GRID, linewidth=0.7)
    axis.set_xlabel("HBM capacity used for expert weights (GiB)")
    axis.set_ylabel("Modeled P99 decode TPOT (ms)")
    axis.legend(loc="upper right", fontsize=8.0)
    figure.suptitle(
        "Better routing reliability converts HBM capacity into lower offload tail",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.865,
        "Measured PCIe anchor; trace-derived LRU cold demand; correlated wave misses. "
        "Offload is compared with reactive offload, not claimed faster than all-HBM.",
        fontsize=8.8,
        color=MUTED,
    )
    return _save(figure, output / "fig2_memory_p99_pareto")


def _inverse(
    output: Path,
    rows: list[dict[str, str]],
    measured_startup_us: float,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.45), sharey=True)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.76, wspace=0.14)
    capacity_colors = {8: ORANGE, 16: BLUE, 32: GREEN}
    for capacity in (8, 16, 32):
        for amplification, linestyle in ((1.0, "-"), (2.0, "--")):
            selected = [
                row
                for row in rows
                if row["demand_source"] == "trace_derived_olmoe"
                and int(row["capacity_experts_per_layer"]) == capacity
                and float(row["object_size_mib"]) == 12.0
                and float(row["amplification"]) == amplification
                and abs(float(row["startup_latency_us"]) - measured_startup_us) < 0.1
                and int(row["transfer_concurrency"]) == 1
            ]
            selected.sort(key=lambda row: int(row["lookahead_layers"]))
            axes[0].plot(
                [int(row["lookahead_layers"]) for row in selected],
                [float(row["minimum_bandwidth_gbps"]) for row in selected],
                color=capacity_colors[capacity],
                linestyle=linestyle,
                marker="o",
                markersize=3.8,
                linewidth=1.5,
            )
    object_colors = {12.0: PURPLE, 4.0: BLUE, 1.0: GREEN, 0.25: ORANGE}
    for object_mib in (12.0, 4.0, 1.0, 0.25):
        selected = [
            row
            for row in rows
            if row["demand_source"] == "normalized_sensitivity"
            and float(row["unique_cold_objects_per_wave"]) == 2.0
            and float(row["object_size_mib"]) == object_mib
            and float(row["amplification"]) == 1.5
            and abs(float(row["startup_latency_us"]) - measured_startup_us) < 0.1
            and int(row["transfer_concurrency"]) == 1
        ]
        selected.sort(key=lambda row: int(row["lookahead_layers"]))
        axes[1].plot(
            [int(row["lookahead_layers"]) for row in selected],
            [float(row["minimum_bandwidth_gbps"]) for row in selected],
            color=object_colors[object_mib],
            marker="o",
            markersize=3.8,
            linewidth=1.5,
            label=f"{object_mib:g} MiB",
        )
    for axis in axes:
        axis.axhline(24.1354, color=TEXT, linestyle=":", linewidth=1.1)
        axis.set_yscale("log")
        axis.set_xticks([1, 2, 3, 6, 9, 12, 15])
        axis.grid(color=GRID, linewidth=0.7)
        axis.set_xlabel("Lookahead Δ (MoE layers)")
    axes[0].set_ylabel("Minimum interconnect bandwidth (GB/s)")
    axes[0].set_title("Trace-derived whole experts", loc="left", fontweight="bold")
    axes[1].set_title("Normalized U=2, A=1.5×", loc="left", fontweight="bold")
    axes[1].legend(title="Transfer object", fontsize=8.0, title_fontsize=8.2)
    capacity_handles = [
        Line2D([0], [0], color=capacity_colors[k], lw=1.7, label=f"K={k}")
        for k in (8, 16, 32)
    ] + [
        Line2D([0], [0], color=TEXT, lw=1.5, linestyle="-", label="A=1×"),
        Line2D([0], [0], color=TEXT, lw=1.5, linestyle="--", label="A=2×"),
    ]
    axes[0].legend(handles=capacity_handles, ncol=2, fontsize=7.8)
    figure.suptitle(
        "Lookahead buys bandwidth; amplification and object size spend it",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.865,
        "Inverse first-order bound including measured 2.8 μs startup. "
        "Dotted line is measured PCIe bandwidth; lower curves are easier to serve.",
        fontsize=8.9,
        color=MUTED,
    )
    return _save(figure, output / "fig3_inverse_bandwidth_lookahead")


def plot_architecture(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    _style()
    analysis = Path(experiment_config["output_dir"])
    inputs = [
        analysis / "ax1_envelope.csv",
        analysis / "ax1_pareto.csv",
        analysis / "ax1_predictor_integrity.csv",
        analysis / "ax1_queue_sensitivity.csv",
        analysis / "ax2_inverse_bounds.csv",
        analysis / "ax2_phase_points.csv",
        analysis / "ax3_staging.csv",
        analysis / "summary.json",
        analysis / "evidence_ledger.json",
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"run architecture analysis before plotting: {path}")
    output = Path(output_dir) if output_dir else analysis / "figures"
    output.mkdir(parents=True, exist_ok=True)
    phase = _read_csv(analysis / "ax2_phase_points.csv")
    pareto = _read_csv(analysis / "ax1_pareto.csv")
    inverse = _read_csv(analysis / "ax2_inverse_bounds.csv")
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    startup = float(summary["measured_inputs"]["h2d_startup_latency_us"])

    outputs: list[Path] = []
    outputs.extend(_phase_map(output, phase))
    outputs.extend(_pareto(output, pareto))
    outputs.extend(_inverse(output, inverse, startup))
    note = output / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# AX architecture figures: human review",
                "",
                "These figures combine measured anchors, trace-derived demand, "
                "assumed future-router quality, and hypothetical hardware. They "
                "are analytical projections, not measured speedups.",
                "",
                "## Review checklist",
                "",
                "- [ ] The phase-map axes are complete cold-set coverage and "
                "raw service headroom; amplification is applied within each panel.",
                "- [ ] Green is read as an SLO candidate, not a demonstrated system.",
                "- [ ] The Pareto comparison is predictive versus reactive offload "
                "on the same measured PCIe hierarchy.",
                "- [ ] The all-resident point is a capacity/performance reference, "
                "not the baseline that predictive CPU offload must beat.",
                "- [ ] The inverse curve is a necessary first-order bandwidth "
                "bound; queue and reliability tails remain separate constraints.",
                "- [ ] AX3's 192/384 MiB whole-expert double-buffer bounds are "
                "checked against `ax3_staging.csv`.",
                "- [ ] One architectural point is selected before any optional "
                "live asynchronous calibration.",
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
        "analysis": "assumption_driven_architecture_exploration",
        "evidence_grade": summary["evidence_grade"],
        "inputs": {str(path): _sha256(path) for path in inputs},
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
