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
CAPACITY_COLORS = {8: "#D97732", 16: "#3266A8", 32: "#2A8C72"}


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
        metadata={"Creator": "ep-predict H5 first-order analysis"},
    )
    return [png, pdf]


def _plot_transfer_waste(
    *,
    output: Path,
    policy_rows: list[dict[str, str]],
    max_amplification: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    selected = list(policy_rows)
    selected.sort(
        key=lambda row: (
            -int(row["capacity"]),
            int(row["lookahead"]),
            0 if row["policy"] == "transition" else 1,
        )
    )
    labels = [
        f"{row['capacity']} local · {row['lookahead']} ahead · "
        f"{row['policy'].capitalize()}"
        for row in selected
    ]
    totals = [
        float(row["candidate_transfer_amplification"]) for row in selected
    ]
    wasted = [value - 1.0 for value in totals]

    figure, axis = plt.subplots(figsize=(8.8, 6.0))
    figure.subplots_adjust(left=0.29, right=0.96, bottom=0.15, top=0.76)
    positions = list(range(len(selected)))
    axis.barh(positions, [1.0] * len(selected), color=USEFUL, height=0.58)
    axis.barh(
        positions,
        wasted,
        left=[1.0] * len(selected),
        color=WASTED,
        height=0.58,
    )
    axis.axvline(
        max_amplification,
        color=TEXT,
        linestyle="--",
        linewidth=1.2,
    )
    axis.text(
        max_amplification + 0.06,
        len(selected) - 0.45,
        "traffic budget",
        fontsize=8.5,
        color=MUTED,
        va="top",
    )
    for position, total, false_copies, row in zip(
        positions, totals, wasted, selected, strict=True
    ):
        axis.text(
            total + 0.08,
            position,
            f"{total:.1f} copies · "
            f"{100 * float(row['complete_cold_set_coverage']):.0f}% requests protected",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
        axis.text(
            1.0 + false_copies / 2,
            position,
            f"{false_copies:.1f} unnecessary",
            ha="center",
            va="center",
            fontsize=8.3,
            color="white",
            fontweight="bold",
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(totals) + 2.4)
    axis.set_xlabel("Experts copied for each expert actually needed")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    figure.suptitle(
        "Every preregistered policy stream exceeds the traffic budget",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.4,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.865,
        "All eight held-out H5 placements: 32 local experts at 1, 3, and 9 "
        "layers ahead, plus the 16-local, 9-layer control. Labels also report "
        "the cold requests whose missing experts are all predicted.",
        fontsize=9.2,
        color=MUTED,
    )
    return _save(figure, output / "fig1_h5_profitability_phase_diagram")


def _plot_earliest_warning(
    *,
    output: Path,
    inverse_rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    figure.subplots_adjust(left=0.15, right=0.97, bottom=0.19, top=0.72)
    for capacity in (8, 16, 32):
        selected = sorted(
            (
                row
                for row in inverse_rows
                if int(row["capacity"]) == capacity
                and float(row["bandwidth_scale"]) == 1.0
                and float(row["candidate_transfer_amplification"]) == 2.0
            ),
            key=lambda row: int(row["lookahead"]),
        )
        x = [int(row["lookahead"]) for row in selected]
        y = [
            (
                100 * float(row["minimum_complete_cold_set_coverage"])
                if row["inverse_window_exists"].lower() == "true"
                else float("nan")
            )
            for row in selected
        ]
        axis.plot(
            x,
            y,
            color=CAPACITY_COLORS[capacity],
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            label=f"{capacity} experts local",
        )
    axis.axhline(50, color=TEXT, linestyle=":", linewidth=1.0)
    axis.set_xlim(1, 15)
    axis.set_ylim(20, 54)
    axis.set_xticks([1, 3, 6, 9, 12, 15])
    axis.set_ylabel("Minimum cold requests that must be fully predicted")
    axis.set_xlabel("Layers of advance warning")
    axis.grid(color=GRID, linewidth=0.7)
    axis.legend(loc="lower right")
    figure.suptitle(
        "The full lookahead sweep shows where the design window first opens",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.4,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        "All 15 lookaheads at the measured link rate, allowing one unnecessary "
        "copy per useful copy. Missing segments are physically infeasible, not "
        "missing observations.",
        fontsize=9.2,
        color=MUTED,
    )
    return _save(figure, output / "fig2_h5_inverse_prediction_requirement")


def plot_h5(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

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
    max_amplification = float(
        gate["thresholds"]["max_predicted_to_useful_bytes"]
    )

    outputs: list[Path] = []
    outputs.extend(
        _plot_transfer_waste(
            output=output,
            policy_rows=policies,
            max_amplification=max_amplification,
        )
    )
    outputs.extend(_plot_earliest_warning(output=output, inverse_rows=inverse))
    plt.close("all")

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
                "- [ ] Figure 1 includes all eight preregistered H5 policy "
                "placements and counts only nonresident candidate transfers.",
                "- [ ] The dashed 2× budget means at most one unnecessary copy "
                "per useful copy.",
                "- [ ] Figure 2 includes all 15 lookaheads at measured bandwidth "
                "and the frozen 2× traffic budget.",
                "- [ ] Missing inverse-curve segments indicate a physically "
                "empty analytical window, not missing data.",
                "- [ ] Headline values agree with `h5_policy_placement.csv` and "
                "`h5_inverse_requirements.csv`.",
                "",
                "## One next action",
                "",
                "Use the figures to review the existing H5 decision; do not "
                "change the frozen gate.",
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
        "figure_semantics": (
            "All preregistered policy transfer streams and the full lookahead "
            "inverse-requirement sweep at the frozen traffic budget."
        ),
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
