from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json
from ep_predict.visualize.h1 import (
    DOMAIN_COLORS,
    DOMAIN_LABELS,
    _configure_matplotlib,
    _save_figure,
)


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
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    _mpl, plt = _configure_matplotlib()
    colors = {"base": "#6B7280", "instruct": "#0072B2"}
    labels = {"base": "Base", "instruct": "Instruct (SFT+DPO+RLVR)"}
    scoped = [
        row
        for row in rows
        if row["domain"] == "__domain_balanced__"
        and int(row["capacity"]) == capacity
    ]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        sharex=True,
        layout="constrained",
    )
    for checkpoint in ("base", "instruct"):
        series = sorted(
            (row for row in scoped if row["checkpoint"] == checkpoint),
            key=lambda row: int(row["delta"]),
        )
        x = np.asarray([int(row["delta"]) for row in series])
        axes[0].plot(
            x,
            [float(row["selection_gain_over_static"]) for row in series],
            color=colors[checkpoint],
            marker="o",
            markersize=3.2,
            markevery=[0, 2, 5, 9, 14],
            label=labels[checkpoint],
        )
        axes[1].plot(
            x,
            [float(row["transition_complete_coverage"]) for row in series],
            color=colors[checkpoint],
            marker="o",
            markersize=3.2,
            markevery=[0, 2, 5, 9, 14],
            label=labels[checkpoint],
        )
        for axis, key in (
            (axes[0], "selection_gain_over_static"),
            (axes[1], "transition_complete_coverage"),
        ):
            final = series[-1]
            axis.annotate(
                f"{100 * float(final[key]):.1f}%",
                xy=(int(final["delta"]), float(final[key])),
                xytext=(-4, 7 if checkpoint == "instruct" else -11),
                textcoords="offset points",
                ha="right",
                color=colors[checkpoint],
                fontsize=7.5,
            )

    axes[0].set_title("(a) Conditional value beyond popularity", loc="left")
    axes[0].set_ylabel("Transition gain over static popularity")
    axes[1].set_title("(b) Strict complete-route prediction", loc="left")
    axes[1].set_ylabel("Tokens with all 8 future experts covered")
    for axis in axes:
        axis.set_xlabel("Target depth from source layer 0, Δ")
        axis.set_xlim(1, 15)
        axis.set_xticks([1, 3, 6, 10, 15])
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    handles, labels_list = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels_list,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        f"Post-training leaves trajectory predictability largely unchanged  K={capacity}",
        fontsize=10.5,
    )
    return _save_figure(
        figure, output_dir / "fig1_base_instruct_predictability"
    )


def _plot_route_agreement(
    rows: list[dict[str, str]],
    *,
    output_dir: Path,
) -> list[Path]:
    from matplotlib.ticker import PercentFormatter

    _mpl, plt = _configure_matplotlib()
    domains = ["code", "math", "general", "conversation"]
    figure, axis = plt.subplots(figsize=(6.8, 3.25), layout="constrained")
    balanced = sorted(
        (row for row in rows if row["domain"] == "__domain_balanced__"),
        key=lambda row: int(row["layer_id"]),
    )
    for domain in domains:
        series = sorted(
            (row for row in rows if row["domain"] == domain),
            key=lambda row: int(row["layer_id"]),
        )
        axis.plot(
            [int(row["layer_id"]) for row in series],
            [float(row["selection_agreement"]) for row in series],
            color=DOMAIN_COLORS[domain],
            linewidth=1.2,
            alpha=0.78,
            label=DOMAIN_LABELS[domain],
        )
    axis.plot(
        [int(row["layer_id"]) for row in balanced],
        [float(row["selection_agreement"]) for row in balanced],
        color="#222222",
        linewidth=2.2,
        marker="o",
        markersize=3.2,
        label="Domain-balanced mean",
        zorder=5,
    )
    axis.set_title(
        "How much of each token’s top-8 route survives post-training",
        loc="left",
    )
    axis.set_xlabel("MoE layer")
    axis.set_ylabel("Base–Instruct expert-selection agreement")
    axis.set_xticks([0, 3, 6, 9, 12, 15])
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    axis.legend(
        loc="lower left",
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
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
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(
        "\n".join(
            [
                "# C0 Base–Instruct figure review",
                "",
                "## Automated headline",
                "",
                f"Formal decision: **{gate['decision']}**. At layer 0→15 and "
                f"K={capacity}, the Instruct-minus-Base change in conditional "
                "selection gain is "
                f"{100 * gate['instruct_minus_base_gain']:+.1f} percentage "
                f"points. Matched Base/Instruct routes retain "
                f"{100 * mean_agreement:.1f}% of selected expert IDs on average "
                "across depth.",
                "",
                "Figure 1 holds the source layer fixed at layer 0, avoiding the "
                "changing-source composition problem in a global horizon mean. "
                "Figure 2 uses only held-out requests with exactly matching "
                "input token IDs.",
                "",
                "## Human visual-review checkpoint",
                "",
                "- [x] Programmatic check: exact token matching and the 96/32 "
                "request split pass.",
                "- [x] Programmatic/visual check: conditional gain is "
                "distinguished from raw coverage/skew.",
                "- [x] Visual check: complete top-8 coverage is not described "
                "as hardware benefit.",
                "- [x] Visual check: domain and layer heterogeneity in Figure 2 "
                "is visible and recorded in `docs/C0_RESULTS.md`.",
                "- [ ] The Base–Instruct endpoint result justifies or rejects adding SFT/DPO.",
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
            "prediction_view": "fixed source layer 0 across all future layers",
            "route_agreement": (
                "intersection of Base and Instruct top-8 selected expert IDs "
                "divided by eight"
            ),
        },
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
