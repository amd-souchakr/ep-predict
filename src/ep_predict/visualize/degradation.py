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


def _quality_latency(
    output: Path,
    rows: list[dict[str, str]],
    summary: dict[str, Any],
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.19, top=0.76)
    base = float(summary["frozen_latency_prediction"]["all_local_anchor_ms"])
    gate_y = 1.5 * base
    axis.axvspan(0, 20, color="#E1F0E7", alpha=0.8, zorder=0)
    axis.axhspan(0, gate_y, color="#E1F0E7", alpha=0.35, zorder=0)
    axis.axvline(20, color=GREEN, linestyle=":", linewidth=1.1)
    axis.axhline(gate_y, color=GREEN, linestyle=":", linewidth=1.1)

    colors = {8: ORANGE, 16: BLUE, 32: GREEN}
    markers = {
        "mass_priority_oracle": "o",
        "random_within_route": "s",
        "mass_adversarial": "^",
    }
    for row in rows:
        capacity = int(row["capacity_experts_per_layer"])
        x = 100 * float(row["p99_missing_routed_mass"])
        y = float(row["bounded_p99_tpot_ms"])
        axis.scatter(
            x,
            y,
            color=colors[capacity],
            marker=markers[row["importance_order"]],
            s=58,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        if row["gate_pass"] == "True":
            axis.annotate(
                f"{float(row['bandwidth_gbps']):g} GB/s",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.8,
                color=colors[capacity],
            )
    # Exact baselines have zero erasure. Deduplicate values by K.
    for capacity in (8, 16, 32):
        row = next(
            value
            for value in rows
            if int(value["capacity_experts_per_layer"]) == capacity
        )
        axis.scatter(
            0,
            float(row["reactive_p99_tpot_ms"]),
            color=colors[capacity],
            marker="x",
            s=62,
            linewidth=1.7,
            zorder=4,
        )
    exact_wait = min(
        rows,
        key=lambda row: abs(int(row["capacity_experts_per_layer"]) - 16),
    )
    axis.scatter(
        0,
        float(exact_wait["exact_wait_wave_local_p99_tpot_ms"]),
        facecolor="white",
        edgecolor=PURPLE,
        marker="D",
        s=58,
        linewidth=1.2,
        zorder=4,
    )
    axis.scatter(
        0,
        base,
        marker="*",
        s=125,
        color=TEXT,
        zorder=5,
    )
    axis.text(
        20.7,
        gate_y - 0.7,
        "Preregistered architecture gate",
        color=GREEN,
        fontsize=8.2,
        va="top",
    )
    axis.set_xlim(-2, max(45, max(100 * float(row["p99_missing_routed_mass"]) for row in rows) + 5))
    axis.set_ylim(8, max(80, max(float(row["reactive_p99_tpot_ms"]) for row in rows) + 5))
    axis.set_xlabel("P99 missing normalized routed mass (%)")
    axis.set_ylabel("Modeled P99 decode TPOT (ms)")
    axis.grid(color=GRID, linewidth=0.7)
    handles = [
        Line2D([0], [0], color=colors[k], marker="o", linestyle="none", label=f"K={k}")
        for k in (8, 16, 32)
    ] + [
        Line2D([0], [0], color=TEXT, marker=markers[name], linestyle="none", label=label)
        for name, label in (
            ("mass_priority_oracle", "Mass-priority oracle"),
            ("random_within_route", "Random order"),
            ("mass_adversarial", "Mass-adversarial"),
        )
    ] + [
        Line2D([0], [0], color=TEXT, marker="x", linestyle="none", label="Reactive exact"),
        Line2D([0], [0], color=TEXT, marker="*", linestyle="none", label="All-local"),
    ]
    axis.legend(handles=handles, ncol=2, fontsize=7.7, loc="upper right")
    figure.suptitle(
        "Hard commit exchanges cold-transfer tail latency for missing expert mass",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.86,
        "FCFS trace replay. Deadline points never wait after commit; "
        "the green box is a training target, not demonstrated quality.",
        fontsize=8.8,
        color=MUTED,
    )
    return _save(figure, output / "fig1_deadline_quality_latency_frontier")


def _capacity_pareto(
    output: Path,
    rows: list[dict[str, str]],
    summary: dict[str, Any],
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.76)
    oracle_candidates = [
        row for row in rows if row["importance_order"] == "mass_priority_oracle"
    ]
    oracle = [
        min(
            (
                row
                for row in oracle_candidates
                if int(row["capacity_experts_per_layer"]) == capacity
            ),
            key=lambda row: (
                float(row["p99_missing_routed_mass"]),
                float(row["bandwidth_gbps"]),
            ),
        )
        for capacity in (8, 16, 32)
    ]
    oracle.sort(key=lambda row: float(row["fast_tier_expert_gib"]))
    x = [float(row["fast_tier_expert_gib"]) for row in oracle]
    y = [float(row["bounded_batch1_tokens_per_second"]) for row in oracle]
    mass = [100 * float(row["p99_missing_routed_mass"]) for row in oracle]
    points = axis.scatter(
        x,
        y,
        c=mass,
        cmap="viridis_r",
        vmin=0,
        vmax=max(20, max(mass)),
        marker="o",
        s=82,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Deadline, mass-priority",
    )
    reactive = [
        (
            float(row["fast_tier_expert_gib"]),
            1000.0 / float(row["reactive_p99_tpot_ms"]),
        )
        for row in oracle
    ]
    axis.plot(
        [value[0] for value in reactive],
        [value[1] for value in reactive],
        color=ORANGE,
        marker="x",
        linestyle="--",
        linewidth=1.5,
        label="Reactive exact offload",
    )
    base = float(summary["frozen_latency_prediction"]["all_local_anchor_ms"])
    all_expert_gib = 12.0
    axis.scatter(
        all_expert_gib,
        1000.0 / base,
        color=TEXT,
        marker="*",
        s=125,
        zorder=5,
        label="All-local measured anchor",
    )
    for row in oracle:
        axis.annotate(
            f"K={row['capacity_experts_per_layer']}\n{float(row['bandwidth_gbps']):g} GB/s",
            (
                float(row["fast_tier_expert_gib"]),
                float(row["bounded_batch1_tokens_per_second"]),
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.0,
        )
    colorbar = figure.colorbar(points, ax=axis, pad=0.015)
    colorbar.set_label("P99 missing mass (%)")
    axis.set_xlabel("HBM capacity for routed expert weights (GiB)")
    axis.set_ylabel("Projected batch-1 throughput (tokens/s)")
    axis.set_xlim(0.7, 12.8)
    axis.grid(color=GRID, linewidth=0.7)
    axis.legend(fontsize=8.0, loc="lower right")
    figure.suptitle(
        "Deadline erasure exposes a capacity–throughput–degradation Pareto",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.86,
        "Current OLMoE geometry: 12 GiB total routed experts. "
        "Deadline throughput assumes a fixed 10% local fallback/commit allowance.",
        fontsize=8.7,
        color=MUTED,
    )
    return _save(figure, output / "fig2_capacity_throughput_degradation_pareto")


def _phase_map(
    output: Path,
    fcfs_rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    bandwidths = sorted(
        {
            float(row["bandwidth_gbps"])
            for row in fcfs_rows
            if row["importance_order"] == "mass_priority_oracle"
        }
    )
    tolerances = [0.0, 0.05, 0.10, 0.20, 0.40]
    cmap = ListedColormap(["#E9D9D7", "#F3E6C5", "#DDEFE4", "#C9E4D5"])
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 4.4), sharey=True)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.75, wspace=0.10)
    for axis, capacity in zip(axes, (8, 16, 32), strict=True):
        grid = np.zeros((len(tolerances), len(bandwidths)), dtype=int)
        annotations: list[tuple[int, int, str]] = []
        for x_index, bandwidth in enumerate(bandwidths):
            candidates = [
                row
                for row in fcfs_rows
                if int(row["capacity_experts_per_layer"]) == capacity
                and abs(float(row["bandwidth_gbps"]) - bandwidth) < 1e-6
                and row["importance_order"] == "mass_priority_oracle"
            ]
            best = min(
                candidates,
                key=lambda row: (
                    float(row["p99_missing_routed_mass"]),
                    float(row["full_fallback_wave_fraction"]),
                ),
            )
            missing = float(best["p99_missing_routed_mass"])
            fallback = float(best["full_fallback_wave_fraction"])
            annotations.append(
                (x_index, len(tolerances) - 1, f"{100 * missing:.0f}%")
            )
            for y_index, tolerance in enumerate(tolerances):
                if fallback > 0.01:
                    category = 0  # fallback dominated
                elif missing <= 1e-12:
                    category = 3  # exact
                elif missing <= tolerance:
                    category = 2  # graceful
                else:
                    category = 1  # contract infeasible
                grid[y_index, x_index] = category
        axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        for x_index, y_index, text in annotations:
            axis.text(
                x_index,
                y_index,
                text,
                ha="center",
                va="center",
                fontsize=7.7,
                color=TEXT,
                fontweight="bold",
            )
        axis.set_xticks(range(len(bandwidths)))
        axis.set_xticklabels([f"{value:g}" for value in bandwidths])
        axis.set_yticks(range(len(tolerances)))
        axis.set_yticklabels([f"{100 * value:.0f}" for value in tolerances])
        axis.set_xlabel("Cold-tier bandwidth (GB/s)")
        axis.set_title(f"K={capacity} resident/layer", loc="left", fontweight="bold")
        axis.tick_params(length=0)
    axes[0].set_ylabel("Tolerated P99 missing mass (%)")
    handles = [
        Patch(facecolor="#C9E4D5", label="Exact"),
        Patch(facecolor="#DDEFE4", label="Graceful contract"),
        Patch(facecolor="#F3E6C5", label="Mass exceeds contract"),
        Patch(facecolor="#E9D9D7", label="Fallback dominated"),
    ]
    figure.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        fontsize=8.0,
    )
    figure.suptitle(
        "Bandwidth and tolerated erasure define the deadline-elastic regime",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "C=99%, A=1.5× mass-priority FCFS boundary; best frozen lookahead/slack per cell. "
        "Trace-ordered FCFS; top-row labels show the attained P99 missing mass.",
        fontsize=8.7,
        color=MUTED,
    )
    return _save(figure, output / "fig3_deadline_hardware_phase_map")


def plot_deadline_degradation(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
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
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    outputs: list[Path] = []
    outputs.extend(_quality_latency(output, fcfs, summary))
    outputs.extend(_capacity_pareto(output, fcfs, summary))
    outputs.extend(_phase_map(output, fcfs))
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
                "- [ ] Normalized-within-top-8 missing mass is not mistaken for "
                "current OLMoE's raw selected probability mass.",
                "- [ ] Every deadline point has exactly zero post-commit transfer wait.",
                "- [ ] The green gate region is read as a future training target, "
                "not demonstrated language quality.",
                "- [ ] Reactive offload—not all-HBM—is the performance baseline.",
                "- [ ] FCFS candidate values match `deadline_fcfs_candidates.csv`.",
                "- [ ] The capacity figure includes the fallback-plane assumption "
                "when interpreting larger-model projections.",
                "- [ ] The phase map uses the selected C=99%, A=1.5× "
                "mass-priority FCFS boundary and does not imply current predictor quality.",
                "- [ ] A training or new-model run remains permission-gated.",
                "",
                "## One next action",
                "",
                "Pending researcher review.",
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
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
