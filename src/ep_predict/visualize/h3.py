from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


BLUE = "#3266A8"
ORANGE = "#D97732"
GREEN = "#2F855A"
PURPLE = "#8056A6"
GRID = "#D9DEE7"
TEXT = "#20242B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_figure(figure: Any, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(
        png,
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "ep-predict H3 scripted visualization"},
    )
    return [png, pdf]


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
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
            "figure.dpi": 120,
        }
    )


def _domain_balanced(
    rows: list[dict[str, str]],
    *,
    metric: str,
) -> dict[str, list[float]]:
    result = {"transition": [], "linear": []}
    for delta in (1, 2, 3):
        for policy in result:
            matches = [
                row
                for row in rows
                if row["phase"] == "decode"
                and row["domain"] == "__domain_balanced__"
                and int(row["capacity"]) == 16
                and int(row["delta"]) == delta
                and row["baseline"] == policy
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one summary row for {policy}, delta={delta}"
                )
            result[policy].append(100 * float(matches[0][metric]))
    return result


def _plot_lookahead(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    selection = _domain_balanced(rows, metric="mean_selection_coverage")
    complete = _domain_balanced(rows, metric="mean_complete_token_coverage")
    x = [1, 2, 3]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)
    for axis, values, title, ylabel, limits in (
        (
            axes[0],
            selection,
            "Future experts found",
            "Selection coverage (%)",
            (70, 83),
        ),
        (
            axes[1],
            complete,
            "Entire top-8 route found",
            "Complete-token coverage (%)",
            (15, 33),
        ),
    ):
        axis.plot(
            x,
            values["transition"],
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=BLUE,
            label="Transition table",
        )
        axis.plot(
            x,
            values["linear"],
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=ORANGE,
            label="Linear sidecar",
        )
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel("Predicted layer")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, ["n+1", "n+2", "n+3"])
        axis.set_ylim(*limits)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        for policy, color in (("transition", BLUE), ("linear", ORANGE)):
            for xi, yi in zip(x, values[policy], strict=True):
                axis.annotate(
                    f"{yi:.1f}",
                    (xi, yi),
                    xytext=(0, 7 if policy == "linear" else -12),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=8.5,
                )
    axes[0].legend(loc="lower left")
    figure.suptitle(
        "H3: the linear sidecar does not decisively beat route transitions",
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    figure.text(
        0.02,
        0.93,
        "Decode, K=16, domain-balanced held-out requests",
        fontsize=9.5,
        color="#555E6B",
    )
    return _save_figure(figure, output_dir / "fig1_h3_lookahead_comparison")


def _plot_domain_gains(
    metric_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {
        (
            row["domain"],
            int(row["source_layer"]),
            row["baseline"],
        ): row
        for row in metric_rows
        if row["phase"] == "decode"
        and int(row["capacity"]) == 16
        and int(row["delta"]) == 1
        and row["baseline"] in {"transition", "linear"}
    }
    domains = ["code", "conversation", "general", "math"]
    selection: list[float] = []
    complete: list[float] = []
    for domain in domains:
        source_layers = sorted(
            layer
            for row_domain, layer, policy in lookup
            if row_domain == domain and policy == "linear"
        )
        selection.append(
            100
            * np.mean(
                [
                    float(lookup[(domain, layer, "linear")]["selection_coverage"])
                    - float(
                        lookup[(domain, layer, "transition")][
                            "selection_coverage"
                        ]
                    )
                    for layer in source_layers
                ]
            )
        )
        complete.append(
            100
            * np.mean(
                [
                    float(
                        lookup[(domain, layer, "linear")][
                            "complete_token_coverage"
                        ]
                    )
                    - float(
                        lookup[(domain, layer, "transition")][
                            "complete_token_coverage"
                        ]
                    )
                    for layer in source_layers
                ]
            )
        )

    positions = np.arange(len(domains))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.6, 4.0))
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.78)
    selection_bars = axis.bar(
        positions - width / 2,
        selection,
        width,
        color=GREEN,
        label="Selection coverage gain",
    )
    complete_bars = axis.bar(
        positions + width / 2,
        complete,
        width,
        color=PURPLE,
        label="Complete-token gain",
    )
    axis.axhline(0, color="#6B7280", linewidth=1.0)
    axis.set_xticks(
        positions,
        ["Code", "Conversation", "General prose", "Mathematics"],
    )
    axis.set_ylabel("Linear gain over transition (percentage points)")
    figure.suptitle(
        "Benefits are domain-dependent, not broadly reliable",
        x=0.03,
        y=0.98,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    figure.text(
        0.03,
        0.90,
        "Primary gate: decode, K=16, n+1; positive is better",
        fontsize=9.5,
        color="#555E6B",
    )
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.legend(loc="upper center", ncol=2)
    axis.set_ylim(min(-5, min(selection + complete) - 2), max(15, max(complete) + 2))
    for bars in (selection_bars, complete_bars):
        for bar in bars:
            value = float(bar.get_height())
            axis.annotate(
                f"{value:+.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if value >= 0 else -11),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )
    return _save_figure(figure, output_dir / "fig2_h3_domain_consistency")


def plot_h3(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    run = Path(run_dir)
    analysis_dir = run / "analysis" / "h3"
    summary_path = analysis_dir / "summary.csv"
    metrics_path = analysis_dir / "metrics.csv"
    gate_path = analysis_dir / "gate.json"
    for path in (summary_path, metrics_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(f"run analyze-h3 before plotting: {path}")
    destination = (
        Path(output_dir) if output_dir is not None else analysis_dir / "figures"
    )
    destination.mkdir(parents=True, exist_ok=True)
    summaries = _read_csv(summary_path)
    metrics = _read_csv(metrics_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    outputs: list[Path] = []
    outputs.extend(_plot_lookahead(summaries, destination))
    plt.close("all")
    outputs.extend(_plot_domain_gains(metrics, destination))
    plt.close("all")

    figures_note = destination / "FIGURES.md"
    figures_note.write_text(
        "\n".join(
            [
                "# H3 figure review",
                "",
                "## Automated conclusion",
                "",
                f"Decision: **{gate['decision']}**.",
                "",
                "At the primary decode K=16, n+1 gate, the linear sidecar "
                f"changes selection coverage by "
                f"{100 * gate['mean_selection_coverage_gain']:+.1f} percentage "
                "points and complete-token coverage by "
                f"{100 * gate['mean_complete_token_coverage_gain']:+.1f} "
                "points versus the transition table. It does not satisfy the "
                "preregistered consistency and selection-gain conditions.",
                "",
                "## Human review checklist",
                "",
                "- [ ] Axes, units, aggregation, and baselines are correct.",
                "- [ ] Headline values agree with `gate.json` and `summary.csv`.",
                "- [ ] Domain heterogeneity and candidate churn are considered.",
                "- [ ] The reviewer accepts or challenges the automated H3 decision.",
                "- [ ] One next action is recorded before H4 starts.",
                "",
                "Recommended next action: use the H2 transition policy in the "
                "minimum H4 hardware-feasibility study; do not tune the predictor.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(figures_note)
    manifest = {
        "hypothesis": "H3",
        "decision": gate["decision"],
        "inputs": {
            str(path): _sha256(path)
            for path in (summary_path, metrics_path, gate_path)
        },
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
