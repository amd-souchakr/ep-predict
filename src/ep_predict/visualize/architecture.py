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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
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


def _queue_tail(
    output: Path,
    rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.0, 5.1))
    figure.subplots_adjust(left=0.13, right=0.80, bottom=0.18, top=0.72)
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    label_offsets = [-10, 0, 10, 0]
    for color, row, y_offset in zip(
        colors, rows, label_offsets, strict=True
    ):
        values = [
            float(row["wave_local_p99_stall_ms"]),
            float(row["p99_queue_replay_stall_ms"]),
        ]
        axis.plot(
            [0, 1],
            values,
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
        )
        label = (
            f"K={row['capacity_experts_per_layer']}, "
            f"{row['lookahead_layers']} ahead, "
            f"C={100 * float(row['coverage']):.1f}%, "
            f"A={float(row['amplification']):.1f}×, "
            f"{float(row['bandwidth_gbps']):.0f} GB/s"
        )
        axis.annotate(
            f"{values[1]:.1f} ms  {label}",
            (1, values[1]),
            xytext=(7, y_offset),
            textcoords="offset points",
            va="center",
            fontsize=8.0,
            color=color,
        )
    axis.set_xticks([0, 1], ["Wave-local estimate", "Trace-ordered queue"])
    axis.set_xlim(-0.12, 1.65)
    axis.set_ylim(0, 72)
    axis.set_ylabel("Delay in the worst 1% of waves")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    figure.suptitle(
        "Trace ordering raises the P99 stall in all four boundary checks",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.4,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "Each line holds capacity, warning distance, predictor quality, traffic, "
        "and bandwidth fixed. The rise is caused by bursty candidates sharing "
        "one FCFS transfer queue.",
        fontsize=9.1,
        color=MUTED,
    )
    return _save(figure, output / "fig1_profitability_phase_map")


def _memory_equivalence(
    output: Path,
    rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    reactive = sorted(
        (row for row in rows if row["policy"] == "reactive_offload"),
        key=lambda row: float(row["fast_tier_expert_gib"]),
    )
    predictive = sorted(
        (
            row
            for row in rows
            if row["quality_profile"] == "predictive_C99_A1.5"
        ),
        key=lambda row: float(row["fast_tier_expert_gib"]),
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.19, top=0.72)
    for selected, color, label, marker in (
        (reactive, ORANGE, "Reactive offload", "o"),
        (
            predictive,
            BLUE,
            "Assumed predictor: 99% coverage, 1.5× traffic",
            "s",
        ),
    ):
        axis.plot(
            [float(row["fast_tier_expert_gib"]) for row in selected],
            [float(row["modeled_p99_tpot_ms"]) for row in selected],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            label=label,
        )
        for row in selected:
            axis.annotate(
                f"K={row['capacity_experts_per_layer']}",
                (
                    float(row["fast_tier_expert_gib"]),
                    float(row["modeled_p99_tpot_ms"]),
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
    axis.set_xlim(1.1, 6.4)
    axis.set_ylim(25, 77)
    axis.set_xlabel("Local memory used for expert weights (GiB)")
    axis.set_ylabel("Modeled P99 decode time (ms)")
    axis.grid(color=GRID, linewidth=0.7)
    axis.legend(loc="upper right", fontsize=8.5)
    figure.suptitle(
        "The full capacity sweep shows how assumed prediction trades memory for tail latency",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.835,
        "Projected on the same measured PCIe hierarchy. The 1.5 GiB predictive "
        "point reaches 48.0 ms versus 50.6 ms for reactive offload at 6 GiB; "
        "all three capacities remain visible.",
        fontsize=9.1,
        color=MUTED,
    )
    return _save(figure, output / "fig2_memory_p99_pareto")


def _bandwidth_warning(
    output: Path,
    rows: list[dict[str, str]],
    measured_startup_us: float,
    measured_bandwidth_gbps: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    selected = sorted(
        (
            row
            for row in rows
            if row["demand_source"] == "trace_derived_olmoe"
            and int(row["capacity_experts_per_layer"]) == 16
            and float(row["object_size_mib"]) == 12.0
            and float(row["amplification"]) == 1.0
            and abs(float(row["startup_latency_us"]) - measured_startup_us)
            < 0.1
            and int(row["transfer_concurrency"]) == 1
        ),
        key=lambda row: int(row["lookahead_layers"]),
    )
    deltas = [int(row["lookahead_layers"]) for row in selected]
    values = [float(row["minimum_bandwidth_gbps"]) for row in selected]

    figure, axis = plt.subplots(figsize=(8.0, 4.7))
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.20, top=0.72)
    axis.plot(
        deltas,
        values,
        color=BLUE,
        marker="o",
        markersize=5,
        linewidth=2.2,
    )
    axis.axhline(
        measured_bandwidth_gbps,
        color=TEXT,
        linestyle="--",
        linewidth=1.2,
    )
    axis.text(
        14.8,
        measured_bandwidth_gbps + 1.4,
        f"measured link: {measured_bandwidth_gbps:.1f} GB/s",
        ha="right",
        fontsize=8.6,
        color=MUTED,
    )
    for delta in (1, 3, 6, 9):
        value = values[deltas.index(delta)]
        axis.annotate(
            f"{value:.1f}",
            (delta, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    axis.set_ylim(0, max(values) + 12)
    axis.set_xlim(1, 15)
    axis.set_xticks([1, 3, 6, 9, 12, 15])
    axis.set_xlabel("Layers of advance warning")
    axis.set_ylabel("Minimum average link bandwidth (GB/s)")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    figure.suptitle(
        "The full inverse curve shows how advance warning buys bandwidth",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.1,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "First-order K=16 whole-expert bound with no unnecessary traffic. "
        "Queue bursts and prediction misses remain separate tail constraints.",
        fontsize=9.1,
        color=MUTED,
    )
    return _save(figure, output / "fig3_inverse_bandwidth_lookahead")


def plot_architecture(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

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
            raise FileNotFoundError(
                f"run architecture analysis before plotting: {path}"
            )
    output = Path(output_dir) if output_dir else analysis / "figures"
    output.mkdir(parents=True, exist_ok=True)
    pareto = _read_csv(analysis / "ax1_pareto.csv")
    queue = _read_csv(analysis / "ax1_queue_sensitivity.csv")
    inverse = _read_csv(analysis / "ax2_inverse_bounds.csv")
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    startup = float(summary["measured_inputs"]["h2d_startup_latency_us"])
    measured_bandwidth = float(
        summary["measured_inputs"]["h2d_bandwidth_gbps"]
    )

    outputs: list[Path] = []
    outputs.extend(_queue_tail(output, queue))
    outputs.extend(_memory_equivalence(output, pareto))
    outputs.extend(
        _bandwidth_warning(
            output,
            inverse,
            startup,
            measured_bandwidth,
        )
    )
    plt.close("all")

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
                "- [ ] Figure 1 includes all four selected queue-sensitivity "
                "checks and holds each line's scenario fixed.",
                "- [ ] Figure 2 compares predictive and reactive offload on the "
                "same hierarchy and does not claim equivalence to all-resident "
                "execution.",
                "- [ ] Figure 2 includes all K=8/16/32 capacity points; its 99% "
                "coverage and 1.5× traffic are assumed future-predictor properties.",
                "- [ ] Figure 3 is a necessary first-order average-bandwidth "
                "bound; queue and reliability tails remain separate.",
                "- [ ] AX3's 192/384 MiB whole-expert double-buffer bounds remain "
                "recorded in `ax3_staging.csv` and `REPORT.md`.",
                "",
                "## One next action",
                "",
                "Review the queue-sensitive capacity point before selecting any "
                "live asynchronous calibration.",
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
        "figure_semantics": (
            "All queue-sensitivity checks, the full matched capacity sweep, "
            "and the complete warning-distance bandwidth curve."
        ),
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
