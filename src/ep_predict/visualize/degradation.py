from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json
from ep_predict.visualize.architecture import (
    BLUE,
    GREEN,
    GRID,
    MUTED,
    ORANGE,
    PURPLE,
    TEXT,
    _save,
    _style,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _best_mass_priority(
    rows: list[dict[str, str]],
    *,
    capacity: int,
    bandwidth: float,
) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if int(row["capacity_experts_per_layer"]) == capacity
        and abs(float(row["bandwidth_gbps"]) - bandwidth) < 0.01
        and row["importance_order"] == "mass_priority_oracle"
        and abs(float(row["requested_complete_coverage"]) - 0.99) < 1e-9
        and abs(float(row["requested_amplification"]) - 1.5) < 1e-9
    ]
    if not candidates:
        raise ValueError(
            f"no AX4 mass-priority boundary row for K={capacity}, "
            f"bandwidth={bandwidth:g}"
        )
    return min(
        candidates,
        key=lambda row: (
            float(row["p99_missing_routed_mass"]),
            float(row["full_fallback_wave_fraction"]),
        ),
    )


def _bandwidth_missing(
    output: Path,
    rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    bandwidths = [24.1354255944, 64.0, 128.0, 256.0]
    figure, axis = plt.subplots(figsize=(8.0, 4.7))
    figure.subplots_adjust(left=0.15, right=0.97, bottom=0.20, top=0.72)
    colors = {8: ORANGE, 16: BLUE, 32: GREEN}
    for capacity in (8, 16, 32):
        selected = [
            _best_mass_priority(
                rows,
                capacity=capacity,
                bandwidth=value,
            )
            for value in bandwidths
        ]
        missing = [
            100 * float(row["p99_missing_routed_mass"]) for row in selected
        ]
        axis.plot(
            bandwidths,
            missing,
            color=colors[capacity],
            marker="o",
            markersize=7,
            linewidth=2.2,
            label=f"{capacity} experts local",
        )
        for bandwidth, value in zip(
            bandwidths, missing, strict=True
        ):
            if bandwidth == 256.0 and capacity != 32:
                continue
            label = "all ≈0%" if bandwidth == 256.0 else f"{value:.1f}%"
            y_offset = {8: 7, 16: -13, 32: 7}[capacity]
            axis.annotate(
                label,
                (bandwidth, value),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=7.8,
                color=colors[capacity],
            )
    axis.axhline(20, color=TEXT, linestyle="--", linewidth=1.1)
    axis.text(
        252,
        22,
        "future quality contract: at most 20%",
        ha="right",
        fontsize=8.5,
        color=MUTED,
    )
    axis.set_ylim(0, 112)
    axis.set_xlim(18, 264)
    axis.set_xticks(bandwidths, ["24\n(current)", "64", "128", "256"])
    axis.set_xlabel("Cold-memory link bandwidth (GB/s)")
    axis.set_ylabel(
        "Selected expert contribution unavailable\nin the worst 1% of waves"
    )
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.legend(loc="upper right")
    figure.suptitle(
        "The full bandwidth–residency sweep reveals the deadline boundary",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "All 12 trace-ordered mass-priority boundary points under an assumed "
        "99%-coverage, 1.5×-traffic predictor. Language quality under erasure "
        "is not measured.",
        fontsize=9.0,
        color=MUTED,
    )
    return _save(figure, output / "fig1_deadline_quality_latency_frontier")


def _rare_tail(
    output: Path,
    rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    passing = []
    for capacity in (8, 16, 32):
        candidates = [
            row
            for row in rows
            if int(row["capacity_experts_per_layer"]) == capacity
            and row["importance_order"] == "mass_priority_oracle"
            and row["gate_pass"].lower() == "true"
        ]
        passing.append(
            min(candidates, key=lambda row: float(row["bandwidth_gbps"]))
        )
    labels = [
        f"K={row['capacity_experts_per_layer']}\n"
        f"{float(row['bandwidth_gbps']):.0f} GB/s"
        for row in passing
    ]
    degraded = [100 * float(row["degraded_wave_fraction"]) for row in passing]

    figure, axis = plt.subplots(figsize=(8.0, 4.7))
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.20, top=0.70)
    bars = axis.bar(
        labels,
        degraded,
        color=[ORANGE, BLUE, GREEN],
        width=0.58,
    )
    for bar, value, row in zip(bars, degraded, passing, strict=True):
        worst = 100 * float(row["worst_missing_routed_mass"])
        p99 = 100 * float(row["p99_missing_routed_mass"])
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.06,
            f"{value:.2f}% waves\nP99 {p99:.1f}% · worst {worst:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axis.set_ylim(0, max(degraded) + 0.55)
    axis.set_ylabel("Waves with any unavailable contribution")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    figure.suptitle(
        "Passing points still differ in how often and how severely they degrade",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.3,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.81,
        "The lowest-bandwidth formal pass at each resident capacity. P99 alone "
        "can hide sub-percent degraded waves, so each bar also reports the "
        "worst observed missing contribution.",
        fontsize=9.1,
        color=MUTED,
    )
    return _save(
        figure, output / "fig2_capacity_throughput_degradation_pareto"
    )


def _minimum_bandwidth(
    output: Path,
    rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    bandwidths = [24.1354255944, 64.0, 128.0, 256.0]
    capacities = [8, 16, 32]
    grid = np.zeros((3, 4), dtype=int)
    labels: list[list[str]] = []
    for row_index, capacity in enumerate(capacities):
        row_labels = []
        for column_index, bandwidth in enumerate(bandwidths):
            row = _best_mass_priority(
                rows,
                capacity=capacity,
                bandwidth=bandwidth,
            )
            passed = row["gate_pass"].lower() == "true"
            grid[row_index, column_index] = int(passed)
            missing = 100 * float(row["p99_missing_routed_mass"])
            row_labels.append(
                ("PASS" if passed else "fail") + f"\n{missing:.1f}% missing"
            )
        labels.append(row_labels)

    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    figure.subplots_adjust(left=0.16, right=0.97, bottom=0.20, top=0.72)
    axis.imshow(
        grid,
        aspect="auto",
        cmap=ListedColormap(["#F3E5E2", "#DDEFE4"]),
        vmin=0,
        vmax=1,
    )
    for row_index in range(3):
        for column_index in range(4):
            axis.text(
                column_index,
                row_index,
                labels[row_index][column_index],
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=9,
                color=TEXT,
            )
    axis.set_xticks(
        range(4), ["24\n(current)", "64", "128", "256"]
    )
    axis.set_yticks(range(3), [f"{capacity} local" for capacity in capacities])
    axis.set_xlabel("Cold-memory link bandwidth (GB/s)")
    axis.set_ylabel("Experts kept locally per layer")
    axis.tick_params(length=0)
    figure.suptitle(
        "The formal gate passes only in three high-bandwidth cells",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "Each cell uses the best frozen lookahead/slack at assumed 99% coverage "
        "and 1.5× traffic. Labels show P99 unavailable contribution, but the "
        "pass also requires latency, fallback, domain, and layer-band breadth.",
        fontsize=9.1,
        color=MUTED,
    )
    return _save(figure, output / "fig3_deadline_hardware_phase_map")


def plot_deadline_degradation(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    analysis = Path(experiment_config["output_dir"])
    inputs = [
        analysis / "weight_semantics_integrity.csv",
        analysis / "deadline_envelope.csv",
        analysis / "deadline_fcfs_candidates.csv",
        analysis / "deadline_fcfs_scope_metrics.csv",
        analysis / "degradation_policy_bounds.csv",
        analysis / "deadline_physical_bounds.csv",
        analysis / "large_sparse_model_projection.csv",
        analysis / "summary.json",
        analysis / "evidence_ledger.json",
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(
                f"run AX4 deadline analysis before plotting: {path}"
            )
    output = Path(output_dir) if output_dir else analysis / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fcfs = _read_csv(analysis / "deadline_fcfs_candidates.csv")
    outputs: list[Path] = []
    outputs.extend(_bandwidth_missing(output, fcfs))
    outputs.extend(_rare_tail(output, fcfs))
    outputs.extend(_minimum_bandwidth(output, fcfs))
    plt.close("all")

    review = output / "FIGURES.md"
    review.write_text(
        "\n".join(
            [
                "# AX4 deadline-degradation figures: human review",
                "",
                "These figures combine measured timing anchors, trace-derived "
                "routes and weights, an assumed future predictor, an assumed "
                "erasure-robustness contract, and hypothetical hardware.",
                "",
                "## Review checklist",
                "",
                "- [ ] Unavailable contribution is normalized within the "
                "selected top-8 and is not current OLMoE's raw probability mass.",
                "- [ ] Every deadline point has exactly zero post-commit "
                "transfer wait.",
                "- [ ] Figure 1 contains all 12 K×bandwidth mass-priority "
                "boundary points.",
                "- [ ] Figure 2 compares the lowest-bandwidth formal pass at "
                "each capacity and pairs P99 with incidence and worst case.",
                "- [ ] Figure 3 applies the full formal gate, not only the 20% "
                "missing-contribution condition.",
                "- [ ] Assumed predictor and unvalidated language-quality "
                "robustness remain explicit.",
                "- [ ] A training or new-model run remains permission-gated.",
                "",
                "## One next action",
                "",
                "Review whether the 128–256 GB/s boundary is a credible future "
                "training target.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(review)
    manifest = {
        "analysis": "AX4_deadline_bounded_graceful_expert_degradation",
        "evidence_grade": (
            "trace_calibrated_assumption_driven_architecture_projection"
        ),
        "inputs": {str(path): _sha256(path) for path in inputs},
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
        "figure_semantics": (
            "Full bandwidth/residency boundary, incidence and worst case for "
            "formal passes, and the complete formal-gate outcome matrix."
        ),
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
