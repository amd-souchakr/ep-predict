from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


TEXT = "#20242B"
GRID = "#D9DEE7"
POLICY_COLORS = {"transition": "#3266A8", "linear": "#2A8C72"}
CLASS_COLORS = {"useful_cold": "#D85C41", "useless": "#66758A"}


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
            "font.size": 9.5,
            "axes.titlesize": 11,
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
        metadata={"Creator": "ep-predict H5 admission analysis"},
    )
    return [png, pdf]


def _frontier_plot(
    *,
    output: Path,
    frontier: list[dict[str, str]],
    best: list[dict[str, str]],
    raw_policy: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 2, figsize=(10.7, 4.8), sharey=True)
    figure.subplots_adjust(
        left=0.08, right=0.985, bottom=0.23, top=0.72, wspace=0.14
    )
    for axis, delta in zip(axes, (3, 9), strict=True):
        axis.axvspan(1.0, 2.0, color="#E1EFE6", alpha=0.9)
        axis.axvline(2.0, color="#596273", linestyle="--", linewidth=1.1)
        axis.axhline(50.0, color="#596273", linestyle=":", linewidth=1.1)
        for policy in ("transition", "linear"):
            selected = sorted(
                (
                    row
                    for row in frontier
                    if int(row["lookahead"]) == delta
                    and row["policy"] == policy
                    and float(row["candidate_transfer_amplification"]) < 10
                ),
                key=lambda row: float(row["standardized_score_threshold"]),
            )
            axis.plot(
                [float(row["candidate_transfer_amplification"]) for row in selected],
                [100 * float(row["complete_cold_set_coverage"]) for row in selected],
                color=POLICY_COLORS[policy],
                linewidth=2.0,
                label=policy,
            )
            reference = min(
                (
                    row
                    for row in selected
                    if float(row["complete_cold_set_coverage"]) >= 0.50
                ),
                key=lambda row: float(row["candidate_transfer_amplification"]),
            )
            reference_x = float(reference["candidate_transfer_amplification"])
            reference_y = 100 * float(reference["complete_cold_set_coverage"])
            axis.scatter(
                reference_x,
                reference_y,
                s=46,
                marker="s",
                facecolor="white",
                edgecolor=POLICY_COLORS[policy],
                linewidth=1.5,
                zorder=4,
            )
            axis.annotate(
                f"{reference_x:.1f}×",
                (reference_x, reference_y),
                xytext=(4, 4 if policy == "linear" else -12),
                textcoords="offset points",
                fontsize=7.3,
                color=POLICY_COLORS[policy],
            )
            operating = next(
                row
                for row in best
                if int(row["lookahead"]) == delta and row["policy"] == policy
            )
            x_value = float(operating["candidate_transfer_amplification"])
            y_value = 100 * float(operating["complete_cold_set_coverage"])
            within = operating["within_2x_window"].lower() == "true"
            axis.scatter(
                x_value,
                y_value,
                s=56,
                color=POLICY_COLORS[policy],
                marker="o" if within else "D",
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            axis.annotate(
                (
                    f"{y_value:.0f}% @ {x_value:.1f}×"
                    if within
                    else f"no 2× crossing\nmin {x_value:.1f}×"
                ),
                (x_value, y_value),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=7.4,
                color=POLICY_COLORS[policy],
            )
            raw = next(
                row
                for row in raw_policy
                if int(row["capacity"]) == 32
                and int(row["lookahead"]) == delta
                and row["policy"] == policy
            )
            axis.scatter(
                float(raw["candidate_transfer_amplification"]),
                100 * float(raw["complete_cold_set_coverage"]),
                s=50,
                marker="x",
                color=POLICY_COLORS[policy],
                linewidth=1.7,
                zorder=3,
            )
        axis.set_xlim(1.0, 7.25)
        axis.set_ylim(0, 100)
        axis.set_xlabel("Transferred candidates / useful cold experts")
        axis.set_title(
            f"K=32 resident tier, lookahead Δ={delta}",
            loc="left",
            fontweight="bold",
        )
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.text(
            1.08,
            93,
            "≤2× traffic",
            fontsize=8.2,
            color="#356B4D",
            va="top",
        )
    axes[0].set_ylabel("Complete cold-set coverage (%)")
    handles = [
        Line2D([0], [0], color=POLICY_COLORS["transition"], lw=2, label="Transition"),
        Line2D([0], [0], color=POLICY_COLORS["linear"], lw=2, label="Linear"),
        Line2D(
            [0], [0], marker="x", linestyle="none", color="#596273",
            label="Unfiltered K=32",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", color="#596273",
            label="Best threshold at A≤2×",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", color="#596273",
            label="Minimum A; no 2× crossing",
        ),
        Line2D(
            [0], [0], marker="s", markerfacecolor="white", linestyle="none",
            color="#596273", label="Minimum A at C≥50%",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=8.5,
    )
    figure.suptitle(
        "Score thresholding cuts traffic—but reveals the coverage price",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.7,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "Held-out decode; thresholds act on all 64 expert scores after K=32 "
        "resident filtering. Green marks the H5 traffic budget; 50% is the "
        "saturated-headroom coverage reference.",
        fontsize=9.2,
        color="#555E6B",
    )
    return _save(figure, output / "fig1_admission_frontier")


def _separation_plot(
    *,
    output: Path,
    histograms: list[dict[str, str]],
    separation: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        2, 2, figsize=(10.7, 6.8), sharex=True, sharey="row"
    )
    figure.subplots_adjust(
        left=0.08, right=0.985, bottom=0.16, top=0.76, wspace=0.12, hspace=0.25
    )
    for row_index, policy in enumerate(("transition", "linear")):
        for column_index, delta in enumerate((3, 9)):
            axis = axes[row_index][column_index]
            for candidate_class in ("useless", "useful_cold"):
                selected = sorted(
                    (
                        row
                        for row in histograms
                        if row["policy"] == policy
                        and int(row["lookahead"]) == delta
                        and row["candidate_class"] == candidate_class
                    ),
                    key=lambda row: float(row["bin_center"]),
                )
                x = [float(row["bin_center"]) for row in selected]
                y = [float(row["density"]) for row in selected]
                axis.plot(
                    x,
                    y,
                    color=CLASS_COLORS[candidate_class],
                    linewidth=1.8,
                )
                axis.fill_between(
                    x,
                    y,
                    color=CLASS_COLORS[candidate_class],
                    alpha=0.13,
                )
            stats = next(
                row
                for row in separation
                if row["policy"] == policy and int(row["lookahead"]) == delta
            )
            axis.text(
                0.97,
                0.92,
                f"AUROC {float(stats['useful_vs_useless_auroc']):.3f}\n"
                f"JS {float(stats['score_js_divergence_bits']):.3f} bits",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.3,
                color="#555E6B",
            )
            axis.set_xlim(-2.5, 4.5)
            axis.grid(axis="y", color=GRID, linewidth=0.7)
            axis.set_title(
                f"{policy.capitalize()}, Δ={delta}",
                loc="left",
                fontweight="bold",
            )
            if column_index == 0:
                axis.set_ylabel("Density")
            if row_index == 1:
                axis.set_xlabel(
                    "Within-wave standardized expert score\n"
                    "(linear logit / transition score)"
                )
    handles = [
        Line2D(
            [0], [0], color=CLASS_COLORS["useful_cold"], lw=2,
            label="Actually demanded cold expert",
        ),
        Line2D(
            [0], [0], color=CLASS_COLORS["useless"], lw=2,
            label="Useless nonresident expert",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        fontsize=8.7,
    )
    figure.suptitle(
        "Useful cold experts score higher, but the distributions still overlap",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.7,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.87,
        "Every distribution scores all nonresident expert IDs in each held-out "
        "cold wave. Per-wave standardization removes head-specific logit scale; "
        "JS quantifies the class-conditional score divergence (1 bit maximum).",
        fontsize=9.2,
        color="#555E6B",
    )
    return _save(figure, output / "fig2_expert_score_separation")


def plot_admission(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    _style()
    analysis = Path(experiment_config["output_dir"])
    inputs = [
        analysis / "admission_frontier.csv",
        analysis / "best_at_2x.csv",
        analysis / "score_histograms.csv",
        analysis / "score_separation.csv",
        analysis / "boundary_at_reference_coverage.csv",
        analysis / "summary.json",
        Path(experiment_config["h5_analysis"]) / "h5_policy_placement.csv",
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"analyze admission before plotting: {path}")
    output = Path(output_dir) if output_dir else analysis / "figures"
    output.mkdir(parents=True, exist_ok=True)
    frontier = _read_csv(analysis / "admission_frontier.csv")
    best = _read_csv(analysis / "best_at_2x.csv")
    histograms = _read_csv(analysis / "score_histograms.csv")
    separation = _read_csv(analysis / "score_separation.csv")
    raw = _read_csv(Path(experiment_config["h5_analysis"]) / "h5_policy_placement.csv")
    outputs: list[Path] = []
    outputs.extend(
        _frontier_plot(
            output=output,
            frontier=frontier,
            best=best,
            raw_policy=raw,
        )
    )
    outputs.extend(
        _separation_plot(
            output=output,
            histograms=histograms,
            separation=separation,
        )
    )
    note = output / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# H5 admission figure review",
                "",
                "## Human review checklist",
                "",
                "- [ ] The threshold curve uses held-out labels only for "
                "evaluation, not threshold fitting.",
                "- [ ] K=32 means resident capacity; thresholded candidate "
                "count is allowed to vary by wave.",
                "- [ ] A≤2× is read together with complete cold-set coverage.",
                "- [ ] Standardized scores are recognized as within-wave "
                "linear logits or transition scores, not calibrated "
                "probabilities.",
                "- [ ] AUROC is treated as descriptive separation, not the "
                "hardware decision metric.",
                "- [ ] One next action is recorded.",
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
        "analysis": "h5_admission_separation",
        "evidence_grade": "held_out_trace_driven_analytical_pilot",
        "inputs": {str(path): _sha256(path) for path in inputs},
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
