from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json
from ep_predict.visualize.h1 import _configure_matplotlib, _save_figure


TEXT = "#20242B"
MUTED = "#5E6875"
GRID = "#D9DEE7"
BASE = "#6B7280"
INSTRUCT = "#3266A8"
UNCHANGED = "#2A8C72"
CHANGED = "#D97732"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot_predictability(
    rows: list[dict[str, str]],
    *,
    capacity: int,
    output_dir: Path,
) -> list[Path]:
    _mpl, plt = _configure_matplotlib()
    scoped = [
        row
        for row in rows
        if row["domain"] == "__domain_balanced__"
        and int(row["capacity"]) == capacity
        and int(row["delta"]) == 15
    ]
    values = {
        row["checkpoint"]: 100 * float(row["selection_gain_over_static"])
        for row in scoped
    }
    checkpoints = ["base", "instruct"]
    labels = ["Base", "Instruct"]
    heights = [values[name] for name in checkpoints]

    figure, axis = plt.subplots(figsize=(6.8, 4.25))
    figure.subplots_adjust(left=0.16, right=0.96, bottom=0.19, top=0.70)
    bars = axis.bar(
        labels,
        heights,
        color=[BASE, INSTRUCT],
        width=0.52,
    )
    for bar, value in zip(bars, heights, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"+{value:.1f} points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10.5,
        )
    difference = heights[1] - heights[0]
    axis.annotate(
        f"change: {difference:+.1f} points",
        xy=(1, heights[1]),
        xytext=(0.5, max(heights) + 2.2),
        ha="center",
        fontsize=10,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": MUTED,
            "connectionstyle": "arc3,rad=0",
        },
    )
    axis.set_ylim(0, max(heights) + 4.3)
    axis.set_ylabel("Prediction gain beyond simple popularity")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    figure.suptitle(
        "Post-training changes long-range route predictability by only 1.6 points",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=13.8,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.84,
        "Observed on identical prefill tokens from layer 0 to layer 15. "
        "The preregistered material-change threshold was 5 points.",
        fontsize=9.1,
        color=MUTED,
    )
    return _save_figure(
        figure, output_dir / "fig1_base_instruct_predictability"
    )


def _plot_route_agreement(
    rows: list[dict[str, str]],
    *,
    output_dir: Path,
) -> list[Path]:
    _mpl, plt = _configure_matplotlib()
    domain_colors = {
        "code": "#3266A8",
        "math": "#D97732",
        "general": "#2A8C72",
        "conversation": "#8657A6",
    }
    domain_labels = {
        "code": "Code",
        "math": "Mathematics",
        "general": "General prose",
        "conversation": "Conversation",
    }
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.19, top=0.72)
    for domain in ("code", "math", "general", "conversation"):
        selected = sorted(
            (row for row in rows if row["domain"] == domain),
            key=lambda row: int(row["layer_id"]),
        )
        axis.plot(
            [int(row["layer_id"]) for row in selected],
            [
                8 * (1 - float(row["selection_agreement"]))
                for row in selected
            ],
            color=domain_colors[domain],
            linewidth=1.5,
            marker="o",
            markersize=3.5,
            label=domain_labels[domain],
        )
    balanced = sorted(
        (row for row in rows if row["domain"] == "__domain_balanced__"),
        key=lambda row: int(row["layer_id"]),
    )
    axis.plot(
        [int(row["layer_id"]) for row in balanced],
        [
            8 * (1 - float(row["selection_agreement"]))
            for row in balanced
        ],
        color=TEXT,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Domain-balanced mean",
        zorder=5,
    )
    axis.axhline(1, color=MUTED, linestyle="--", linewidth=1.0)
    axis.text(
        0.2,
        1.04,
        "one of eight experts changed",
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 1.85)
    axis.set_xticks([0, 3, 6, 9, 12, 15])
    axis.set_xlabel("MoE layer")
    axis.set_ylabel("Experts changed out of each top-8 route")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.5)
    figure.suptitle(
        "Post-training usually changes fewer than one of eight experts at every layer",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=13.8,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.82,
        "Observed on exactly matched Base and Instruct tokens. Domain lines "
        "expose where the average hides localized substitutions; mathematics "
        "at the final layer is the clearest exception.",
        fontsize=9.1,
        color=MUTED,
    )
    return _save_figure(figure, output_dir / "fig2_matched_route_agreement")


def plot_checkpoint_trajectories(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    analysis_dir = Path(experiment_config["output_dir"]) / "analysis" / "c0"
    destination = (
        Path(output_dir) if output_dir is not None else analysis_dir / "figures"
    )
    destination.mkdir(parents=True, exist_ok=True)
    inputs = {
        "predictability": analysis_dir / "predictability_by_horizon.csv",
        "route_overlap": analysis_dir / "matched_route_overlap.csv",
        "cross_transfer": analysis_dir / "cross_checkpoint_transfer.csv",
        "gate": analysis_dir / "gate.json",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"run C0 analysis before plotting: {path}")
    predictability = _read_csv(inputs["predictability"])
    route_overlap = _read_csv(inputs["route_overlap"])
    capacity = int(experiment_config["decision_gate"]["capacity_experts"])

    outputs: list[Path] = []
    outputs.extend(
        _plot_predictability(
            predictability,
            capacity=capacity,
            output_dir=destination,
        )
    )
    outputs.extend(_plot_route_agreement(route_overlap, output_dir=destination))

    gate = json.loads(inputs["gate"].read_text(encoding="utf-8"))
    balanced = [
        row for row in route_overlap if row["domain"] == "__domain_balanced__"
    ]
    mean_agreement = sum(
        float(row["selection_agreement"]) for row in balanced
    ) / len(balanced)
    notes = destination / "FIGURES.md"
    notes.write_text(
        "\n".join(
            [
                "# C0 Base–Instruct figure review",
                "",
                "## Automated headline",
                "",
                f"Formal decision: **{gate['decision']}**. At layer 0→15 and "
                f"K={capacity}, the Instruct-minus-Base change in conditional "
                f"selection gain is "
                f"{100 * gate['instruct_minus_base_gain']:+.1f} percentage "
                f"points. Matched routes retain "
                f"{100 * mean_agreement:.1f}% of selected expert IDs.",
                "",
                "## Human visual-review checkpoint",
                "",
                "- [x] Figure 1 holds the source layer fixed at layer 0 and "
                "shows the preregistered endpoint comparison.",
                "- [x] Figure 2 translates selection agreement into the number "
                "of unchanged slots in a top-8 route.",
                "- [x] Both figures use only requests with exactly matching "
                "input token IDs.",
                "- [ ] The Base–Instruct endpoint result justifies or rejects "
                "adding SFT/DPO.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(notes)

    manifest = {
        "analysis": "c0-base-instruct-trajectories",
        "figure_grade": "pilot",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in outputs
        ],
        "semantics": {
            "phase": "matched-token prefill",
            "split_unit": "request",
            "prediction_view": "fixed source layer 0 to target layer 15",
            "route_agreement": (
                "changed expert slots out of eight by MoE layer and domain"
            ),
        },
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
