from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


COLORS = {8: "#D97732", 16: "#3266A8", 32: "#2A8C72"}
TEXT = "#20242B"
GRID = "#D9DEE7"


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
        metadata={"Creator": "ep-predict co-design regime map"},
    )
    return [png, pdf]


def plot_codesign_map(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style()
    settings = experiment_config["codesign_map"]
    analysis = Path(settings["output_dir"])
    points_path = analysis / "codesign_points.csv"
    summary_path = analysis / "summary.json"
    for path in (points_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"analyze co-design map before plotting: {path}")
    destination = (
        Path(output_dir) if output_dir is not None else analysis / "figures"
    )
    destination.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(points_path)
    policies = [str(policy) for policy in settings["policies"]]
    capacities = [int(value) for value in experiment_config["simulation"]["capacities"]]
    complete_threshold = 100 * float(settings["min_complete_route_coverage"])
    headroom_threshold = float(settings["physical_headroom_ratio"])

    figure, axes = plt.subplots(1, len(policies), figsize=(11.4, 4.9), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    figure.subplots_adjust(
        left=0.08, right=0.985, bottom=0.25, top=0.74, wspace=0.14
    )
    x_min, x_max = 0.2, 13.0
    y_min, y_max = 0.0, 72.0
    y_split = complete_threshold / y_max
    for axis, policy in zip(axes, policies, strict=True):
        axis.axvspan(
            x_min,
            headroom_threshold,
            ymin=0,
            ymax=y_split,
            color="#F3E8E5",
        )
        axis.axvspan(
            x_min,
            headroom_threshold,
            ymin=y_split,
            ymax=1,
            color="#F5DDD8",
        )
        axis.axvspan(
            headroom_threshold,
            x_max,
            ymin=0,
            ymax=y_split,
            color="#F1ECD9",
        )
        axis.axvspan(
            headroom_threshold,
            x_max,
            ymin=y_split,
            ymax=1,
            color="#E1EFE6",
        )
        axis.axvline(
            headroom_threshold,
            color="#596273",
            linestyle="--",
            linewidth=1.1,
        )
        axis.axhline(
            complete_threshold,
            color="#596273",
            linestyle="--",
            linewidth=1.1,
        )
        for capacity in capacities:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["policy"] == policy
                    and int(row["capacity"]) == capacity
                ),
                key=lambda row: int(row["lookahead"]),
            )
            x = [float(row["cold_service_headroom_ratio"]) for row in selected]
            y = [100 * float(row["complete_route_coverage"]) for row in selected]
            axis.plot(x, y, color=COLORS[capacity], linewidth=1.6, alpha=0.8)
            for row, x_value, y_value in zip(selected, x, y, strict=True):
                oracle_pass = row["oracle_pass"].lower() == "true"
                axis.scatter(
                    x_value,
                    y_value,
                    s=52,
                    marker="o",
                    facecolor=COLORS[capacity] if oracle_pass else "white",
                    edgecolor=COLORS[capacity],
                    linewidth=1.7,
                    zorder=3,
                )
                if int(row["lookahead"]) in {1, 3, 9, 15}:
                    axis.annotate(
                        f"Δ{row['lookahead']}",
                        (x_value, y_value),
                        xytext=(3, 5),
                        textcoords="offset points",
                        fontsize=7.2,
                        color=COLORS[capacity],
                    )
        axis.set_xscale("log")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks([0.25, 0.5, 1, 2, 4, 8, 12])
        axis.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "12"])
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.set_xlabel(
            "Cold-service headroom\n"
            r"$\Delta T_{\mathrm{layer}}/"
            r"(\bar N_{\mathrm{cold}}T_{\mathrm{copy}})$"
        )
        title = "Transition table" if policy == "transition" else "Linear sidecar"
        axis.set_title(title, loc="left", fontweight="bold")
        axis.text(
            0.28,
            67,
            "Transfer-limited",
            fontsize=8.5,
            color="#8A4C45",
        )
        axis.text(
            1.2,
            67,
            "Candidate co-design region",
            fontsize=8.5,
            color="#356B4D",
        )
        axis.text(
            0.34,
            10,
            "Both limited",
            fontsize=8.5,
            color="#7B625E",
        )
        axis.text(
            1.2,
            10,
            "Prediction-limited",
            fontsize=8.5,
            color="#596273",
        )
    axes[0].set_ylabel("Complete top-8 prediction coverage (%)")
    capacity_handles = [
        Line2D([0], [0], color=COLORS[capacity], marker="o", label=f"K={capacity}")
        for capacity in capacities
    ]
    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#596273",
            markerfacecolor="#596273",
            linestyle="none",
            label="Oracle passes",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#596273",
            markerfacecolor="white",
            linestyle="none",
            label="Oracle fails",
        ),
    ]
    figure.legend(
        handles=capacity_handles + state_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=8.5,
    )
    figure.suptitle(
        "Prediction is actionable only where cold-path service has headroom",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.88,
        "Decode at measured H2D rate; K couples resident and candidate budgets. "
        "Filled points pass both oracle thresholds; upper-right is not profit proof.",
        fontsize=9.5,
        color="#555E6B",
    )
    outputs = _save(figure, destination / "fig1_codesign_regime_map")
    plt.close(figure)
    note = destination / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# H4/H2/H3 co-design map review",
                "",
                "This is post-hoc descriptive synthesis. It does not change "
                "the H3 or H4 formal gates.",
                "",
                "## Reading the map",
                "",
                "- Headroom below 1 means mean serialized cold-transfer work "
                "exceeds nominal lead time.",
                "- K is deliberately coupled across per-layer fast-tier "
                "capacity and prediction candidate budget for this screening "
                "slice; it is not a policy replay.",
                "- Complete-route coverage below 50% leaves prediction as the "
                "dominant limitation.",
                "- Filled markers independently pass the trace-driven oracle "
                "on-time-byte and stall-reduction thresholds.",
                "- The upper-right region is only eligible for a policy replay; "
                "profitability requires measured overlap and learned/oracle "
                "recovery.",
                "",
                "## Human review checklist",
                "",
                "- [ ] The headroom ratio and complete-route metric are "
                "interpreted independently.",
                "- [ ] Open points above headroom 1 are recognized as tail or "
                "queue failures, not contradictions.",
                "- [ ] Candidate-region points are not called profitable.",
                "- [ ] Changing eligible target-layer count at long Δ is "
                "retained as a limitation.",
                "- [ ] One next action is recorded.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(note)
    manifest = {
        "analysis": "h4_codesign_map",
        "status": "post_hoc_descriptive",
        "inputs": {
            str(path): _sha256(path) for path in (points_path, summary_path)
        },
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
