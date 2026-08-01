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
USEFUL = "#2A8C72"
WASTED = "#D97732"
BLUE = "#3266A8"


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
        metadata={"Creator": "ep-predict H5 admission analysis"},
    )
    return [png, pdf]


def _plot_coverage_at_budget(
    *,
    output: Path,
    frontier: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = {"transition": "#6B7280", "linear": BLUE}
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    figure.subplots_adjust(
        left=0.10, right=0.98, bottom=0.25, top=0.72, wspace=0.12
    )
    for axis, lookahead in zip(axes, (3, 9), strict=True):
        axis.axvspan(1, 2, color="#DDEFE4", alpha=0.85)
        axis.axvline(2, color=TEXT, linestyle="--", linewidth=1.0)
        axis.axhline(50, color=TEXT, linestyle=":", linewidth=1.0)
        for policy in ("transition", "linear"):
            selected = sorted(
                (
                    row
                    for row in frontier
                    if int(row["lookahead"]) == lookahead
                    and row["policy"] == policy
                    and 1 <= float(row["candidate_transfer_amplification"]) <= 7
                ),
                key=lambda row: float(row["candidate_transfer_amplification"]),
            )
            frontier_x: list[float] = []
            frontier_y: list[float] = []
            best_coverage = -1.0
            for row in selected:
                coverage = 100 * float(row["complete_cold_set_coverage"])
                if coverage <= best_coverage + 1e-9:
                    continue
                frontier_x.append(
                    float(row["candidate_transfer_amplification"])
                )
                frontier_y.append(coverage)
                best_coverage = coverage
            axis.plot(
                frontier_x,
                frontier_y,
                color=colors[policy],
                linewidth=2.2,
                label=policy.capitalize(),
            )
        axis.set_xlim(1, 7)
        axis.set_ylim(0, 88)
        axis.set_xlabel("Copies made per expert actually needed")
        axis.set_title(
            f"{lookahead} layers ahead",
            loc="left",
            fontweight="bold",
        )
        axis.grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].set_ylabel("Cold requests whose missing experts are all found")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
    )
    axes[0].text(
        1.12,
        82,
        "acceptable\ntraffic",
        fontsize=8.2,
        color=USEFUL,
        va="top",
    )
    figure.suptitle(
        "Filtering exposes the full tradeoff between traffic and protected requests",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "Observed threshold sweep with 32 experts resident. Green allows at "
        "most one unnecessary copy per useful copy; the dotted line marks half "
        "of cold requests protected.",
        fontsize=9.2,
        color=MUTED,
    )
    return _save(figure, output / "fig1_admission_frontier")


def _plot_useful_share_at_half_coverage(
    *,
    output: Path,
    boundary: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    selected = sorted(
        (row for row in boundary if row["policy"] == "linear"),
        key=lambda row: int(row["lookahead"]),
    )
    labels = [f"{row['lookahead']} layers ahead" for row in selected]
    useful = [100 * float(row["candidate_precision"]) for row in selected]
    wasted = [100 - value for value in useful]
    positions = list(range(len(selected)))

    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    figure.subplots_adjust(left=0.21, right=0.97, bottom=0.20, top=0.72)
    axis.barh(positions, useful, color=USEFUL, height=0.56)
    axis.barh(
        positions,
        wasted,
        left=useful,
        color=WASTED,
        height=0.56,
    )
    for position, good, bad in zip(positions, useful, wasted, strict=True):
        axis.text(
            good / 2,
            position,
            f"{good:.0f}% useful",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        axis.text(
            good + bad / 2,
            position,
            f"{bad:.0f}% wasted",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of admitted transfers")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=USEFUL, label="Useful"),
            plt.Rectangle((0, 0), 1, 1, color=WASTED, label="Unnecessary"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
    )
    figure.suptitle(
        "Even after filtering, two of every three transfers are wasted",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "Observed operating points chosen to protect at least half of cold "
        "requests. This exposes the rare-useful-candidate problem directly.",
        fontsize=9.2,
        color=MUTED,
    )
    return _save(figure, output / "fig2_expert_score_separation")


def plot_admission(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

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
    boundary = _read_csv(analysis / "boundary_at_reference_coverage.csv")

    outputs: list[Path] = []
    outputs.extend(
        _plot_coverage_at_budget(output=output, frontier=frontier)
    )
    outputs.extend(
        _plot_useful_share_at_half_coverage(
            output=output,
            boundary=boundary,
        )
    )
    plt.close("all")

    note = output / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# H5 admission figure review",
                "",
                "## Human review checklist",
                "",
                "- [ ] Figure 1 shows the full held-out threshold frontier for "
                "both unchanged policies and both frozen lookaheads.",
                "- [ ] The green region fixes the prior H5 traffic budget at at "
                "most two total copies per useful copy.",
                "- [ ] Figure 2 fixes complete cold-request coverage at at least "
                "50% and reports useful versus unnecessary admitted transfers.",
                "- [ ] The useful-transfer share is not confused with AUROC or "
                "the 7–8% unfiltered useful base rate.",
                "- [ ] Headline values agree with `best_at_2x.csv` and "
                "`boundary_at_reference_coverage.csv`.",
                "",
                "## One next action",
                "",
                "Use the figures to review why ranking separation does not "
                "produce an affordable admission policy.",
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
        "figure_semantics": (
            "Full threshold traffic/coverage frontiers and useful-transfer "
            "share when half of cold requests are protected."
        ),
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
