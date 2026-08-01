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
POLICY_COLORS = {
    "oracle": "#2A8C72",
    "lru": "#6B7280",
    "linear": "#3266A8",
    "transition": "#D97732",
}


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
        metadata={"Creator": "ep-predict H6 residency replay"},
    )
    return [png, pdf]


def _plot_primary_comparison(
    *,
    rows: list[dict[str, str]],
    output: Path,
    phase: str,
    capacity: int,
    lookahead: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    selected = {
        row["policy"]: row
        for row in rows
        if row["phase"] == phase
        and row["domain"] == "__domain_balanced__"
        and int(row["capacity"]) == capacity
        and int(row["delta"]) == lookahead
        and row["policy"] in POLICY_COLORS
    }
    missing = set(POLICY_COLORS) - set(selected)
    if missing:
        raise ValueError(
            "H6 summary is missing primary comparison rows for "
            + ", ".join(sorted(missing))
        )

    policies = ["oracle", "lru", "linear", "transition"]
    cold_values = [
        100 * float(selected[policy]["mean_residual_cold_expert_fraction"])
        for policy in policies
    ]
    reuse_values = [
        100 * float(selected[policy]["mean_useful_movement_fraction"])
        for policy in policies
    ]
    labels = {
        "oracle": "Perfect next-use policy",
        "lru": "Ordinary LRU",
        "linear": "Linear predictor",
        "transition": "Transition predictor",
    }

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), sharey=True)
    figure.subplots_adjust(
        left=0.22, right=0.98, bottom=0.17, top=0.72, wspace=0.18
    )
    positions = list(range(len(policies)))
    bars = axes[0].barh(
        positions,
        cold_values,
        color=[POLICY_COLORS[policy] for policy in policies],
        height=0.58,
    )
    lru_value = cold_values[policies.index("lru")]
    axes[0].axvline(lru_value, color=TEXT, linestyle="--", linewidth=1.1)
    axes[0].text(
        lru_value + 0.6,
        -0.45,
        "LRU reference",
        fontsize=8.5,
        color=MUTED,
        va="top",
    )
    for bar, value in zip(bars, cold_values, strict=True):
        axes[0].text(
            value + 0.7,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontweight="bold",
            fontsize=10.5,
        )
    reuse_bars = axes[1].barh(
        positions,
        reuse_values,
        color=[POLICY_COLORS[policy] for policy in policies],
        height=0.58,
    )
    for bar, value in zip(reuse_bars, reuse_values, strict=True):
        axes[1].text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
    axes[0].set_yticks(positions, [labels[policy] for policy in policies])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(cold_values) + 8)
    axes[1].set_xlim(0, 105)
    axes[0].set_xlabel("Demand still missing\n(lower is better)")
    axes[1].set_xlabel("Insertions reused before eviction\n(higher is better)")
    axes[0].set_title(
        "Outcome: remaining cold demand", loc="left", fontweight="bold"
    )
    axes[1].set_title(
        "Mechanism: useful cache insertions", loc="left", fontweight="bold"
    )
    for axis in axes:
        axis.grid(axis="x", color=GRID, linewidth=0.7)
    figure.suptitle(
        "The predictors make worse retention choices than ordinary LRU",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.855,
        f"Observed trace replay at {phase}, {capacity} experts resident, "
        f"{lookahead} layers ahead, and the same one-move-per-wave budget. "
        "The right panel explains the left: predictor-guided insertions are "
        "less likely than LRU insertions to earn a later hit.",
        fontsize=9.1,
        color=MUTED,
    )
    return _save(figure, output / "fig1_h6_residency_gain_heatmap")


def plot_h6(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    analysis = Path(str(experiment_config["output_dir"]))
    output = Path(output_dir) if output_dir else analysis / "figures"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = analysis / "scope_metrics.csv"
    gate_path = analysis / "gate.json"
    summary_path = analysis / "summary.csv"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    primary = gate["primary_scope"]
    summary_rows = _read_csv(summary_path)
    outputs = _plot_primary_comparison(
        rows=summary_rows,
        output=output,
        phase=str(primary["phase"]),
        capacity=int(primary["capacity_experts"]),
        lookahead=int(primary["lookahead"]),
    )
    plt.close("all")

    review_path = output / "FIGURES.md"
    review_path.write_text(
        "\n".join(
            [
                "# H6 figure review",
                "",
                "- [ ] All policies use the same resident capacity and runtime "
                "movement budget.",
                "- [ ] Residual cold demand is read as lower-is-better.",
                "- [ ] Useful insertion fraction means the admitted expert earns "
                "a later hit before eviction.",
                "- [ ] LRU is the direct operational comparator.",
                "- [ ] The oracle is a policy ceiling, not an implementable "
                "predictor.",
                "- [ ] Headline values agree with `summary.csv`, `gate.json`, "
                "and `REPORT.md`.",
                "- [ ] The negative gate remains unchanged.",
                "",
                "**Human review complete:** no",
                "",
            ]
        ),
        encoding="utf-8",
    )

    input_paths = [metrics_path, summary_path, gate_path]
    output_paths = outputs + [review_path]
    manifest = {
        "hypothesis": "H6",
        "human_review_complete": False,
        "figure_semantics": (
            "Domain-balanced residual cold demand and insertion reuse at the "
            "frozen primary scope for learned policies, LRU, and an equal-budget "
            "next-use oracle."
        ),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in input_paths
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in output_paths
        ],
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest
