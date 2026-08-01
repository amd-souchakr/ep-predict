from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DOMAIN_ORDER = ["__all__", "code", "math", "general", "conversation"]
DOMAIN_LABELS = {
    "__all__": "Mixed",
    "code": "Code",
    "math": "Mathematics",
    "general": "General prose",
    "conversation": "Conversation",
}
DOMAIN_COLORS = {
    "__all__": "#333333",
    "code": "#0072B2",
    "math": "#D55E00",
    "general": "#009E73",
    "conversation": "#CC79A7",
}
PHASES = ["prefill", "decode"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing H1 analysis table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capacity_curves(
    rows: list[dict[str, str]],
    *,
    layers: list[int],
    num_experts: int,
) -> dict[tuple[str, str], list[float]]:
    """Mean cumulative selection coverage over layer-local popularity ranks."""
    probabilities: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = (row["phase"], row["domain"], int(row["layer_id"]))
        probabilities[key][int(row["rank"])] = float(row["probability"])

    curves: dict[tuple[str, str], list[float]] = {}
    for phase in PHASES:
        for domain in DOMAIN_ORDER:
            layer_curves: list[list[float]] = []
            for layer in layers:
                ranked = probabilities.get((phase, domain, layer))
                if not ranked:
                    raise ValueError(
                        f"rank-frequency table lacks {phase}/{domain}/layer {layer}"
                    )
                running = 0.0
                curve: list[float] = []
                for rank in range(1, num_experts + 1):
                    running += ranked.get(rank, 0.0)
                    curve.append(running)
                layer_curves.append(curve)
            curves[(phase, domain)] = [
                sum(layer_curve[index] for layer_curve in layer_curves)
                / len(layer_curves)
                for index in range(num_experts)
            ]
    return curves


def _configure_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "visualization dependencies are missing; run `uv sync --extra viz`"
        ) from exc

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return mpl, plt


def _save_figure(figure: Any, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    figure.savefig(png, dpi=450, bbox_inches="tight")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": output_base.stem,
            "Creator": "ep-predict",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    return [png, pdf]


def _plot_capacity_coverage(
    *,
    rows: list[dict[str, str]],
    layers: list[int],
    num_experts: int,
    expert_bytes: int,
    output_dir: Path,
) -> list[Path]:
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    _mpl, plt = _configure_matplotlib()
    capacities = np.arange(1, num_experts + 1)
    curves = _capacity_curves(
        rows,
        layers=layers,
        num_experts=num_experts,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for panel, (axis, phase) in enumerate(zip(axes, PHASES, strict=True)):
        axis.plot(
            capacities,
            capacities / num_experts,
            color="#9A9A9A",
            linestyle="--",
            linewidth=1.0,
            label="Uniform",
            zorder=1,
        )
        for domain in DOMAIN_ORDER:
            axis.plot(
                capacities,
                curves[(phase, domain)],
                color=DOMAIN_COLORS[domain],
                label=DOMAIN_LABELS[domain],
                zorder=3 if domain == "__all__" else 2,
                linewidth=1.8 if domain == "__all__" else 1.4,
            )
        for capacity in (8, 16, 32):
            axis.axvline(capacity, color="#D5D5D5", linewidth=0.6, zorder=0)
        axis.set_title(f"({chr(ord('a') + panel)}) {phase.capitalize()}", loc="left")
        axis.set_xlabel("Resident experts per layer, $K$")
        axis.set_xlim(1, num_experts)
        axis.set_ylim(0, 1.01)
        axis.set_xticks([1, 8, 16, 32, 48, 64])
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5)

        bytes_per_capacity = expert_bytes * len(layers)
        gib_per_capacity = bytes_per_capacity / (1024**3)
        secondary = axis.secondary_xaxis(
            "top",
            functions=(
                lambda value: value * gib_per_capacity,
                lambda value: value / gib_per_capacity,
            ),
        )
        secondary.set_xlabel("Fast-tier capacity across layers (GiB)", labelpad=4)
        secondary.set_xticks([1.5, 3, 6, 12])
        secondary.tick_params(length=2.5)

    axes[0].set_ylabel("Cumulative expert-selection coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=6,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    return _save_figure(figure, output_dir / "fig1_capacity_coverage")


def _plot_skew_stability(
    *,
    popularity_rows: list[dict[str, str]],
    window_rows: list[dict[str, str]],
    experiment_config: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    _mpl, plt = _configure_matplotlib()
    gate = experiment_config["decision_gate"]
    capacity = int(gate["capacity_experts"])
    window_size = int(gate["window_size"])
    min_lift = float(gate["min_coverage_lift_over_uniform"])
    min_jaccard = float(gate["min_hotset_jaccard"])
    min_ratio = float(gate["min_lagged_oracle_ratio"])
    coverage_key = f"top_{capacity}_coverage"

    stability = {
        (
            row["phase"],
            row["domain"],
            int(row["layer_id"]),
            int(row["window_size"]),
        ): row
        for row in window_rows
    }
    points: dict[str, dict[str, list[tuple[float, float, float, int]]]] = {
        phase: {domain: [] for domain in DOMAIN_ORDER} for phase in PHASES
    }
    for row in popularity_rows:
        phase = row["phase"]
        domain = row["domain"]
        if phase not in PHASES or domain not in DOMAIN_ORDER:
            continue
        layer = int(row["layer_id"])
        window = stability.get((phase, domain, layer, window_size))
        if window is None:
            continue
        uniform = capacity / int(row["num_experts"])
        points[phase][domain].append(
            (
                float(row[coverage_key]) / uniform,
                float(window["mean_jaccard"]),
                float(window["mean_lagged_oracle_ratio"]),
                layer,
            )
        )

    all_x = [
        point[0]
        for phase in PHASES
        for domain in DOMAIN_ORDER
        for point in points[phase][domain]
    ]
    x_min = min(1.4, min(all_x) - 0.1)
    x_max = max(all_x) + 0.25
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for panel, (axis, phase) in enumerate(zip(axes, PHASES, strict=True)):
        axis.add_patch(
            Rectangle(
                (min_lift, min_jaccard),
                x_max - min_lift,
                1.0 - min_jaccard,
                facecolor="#009E73",
                edgecolor="none",
                alpha=0.045,
                zorder=0,
            )
        )
        for domain in DOMAIN_ORDER:
            domain_points = points[phase][domain]
            for lagged_pass in (False, True):
                selected = [
                    point
                    for point in domain_points
                    if (point[2] >= min_ratio) == lagged_pass
                ]
                if not selected:
                    continue
                axis.scatter(
                    [point[0] for point in selected],
                    [point[1] for point in selected],
                    s=25 if domain == "__all__" else 21,
                    facecolors=DOMAIN_COLORS[domain] if lagged_pass else "none",
                    edgecolors=DOMAIN_COLORS[domain],
                    linewidths=0.8,
                    alpha=0.9,
                    zorder=3 if domain == "__all__" else 2,
                )
            if phase == "decode" and domain == "__all__":
                for lift, jaccard, ratio, layer in domain_points:
                    if (
                        lift >= min_lift
                        and jaccard >= min_jaccard
                        and ratio >= min_ratio
                    ):
                        axis.annotate(
                            f"L{layer}",
                            (lift, jaccard),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=7,
                            color=DOMAIN_COLORS[domain],
                        )
        axis.axvline(min_lift, color="#777777", linewidth=0.8, linestyle="--")
        axis.axhline(min_jaccard, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_title(f"({chr(ord('a') + panel)}) {phase.capitalize()}", loc="left")
        axis.set_xlabel(f"Top-{capacity} coverage lift over uniform")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(0, 1.0)
        axis.set_xticks(np.arange(1.5, 4.6, 0.5))
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    axes[0].set_ylabel(f"Consecutive-window top-{capacity} Jaccard")

    domain_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=DOMAIN_COLORS[domain],
            markerfacecolor=DOMAIN_COLORS[domain],
            markersize=4.5,
            label=DOMAIN_LABELS[domain],
        )
        for domain in DOMAIN_ORDER
    ]
    semantic_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="#666666",
            markerfacecolor="#666666",
            markersize=4.5,
            label=f"Lagged/oracle ≥ {min_ratio:.2f}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="#666666",
            markerfacecolor="none",
            markersize=4.5,
            label=f"Lagged/oracle < {min_ratio:.2f}",
        ),
    ]
    figure.legend(
        domain_handles + semantic_handles,
        [handle.get_label() for handle in domain_handles + semantic_handles],
        loc="outside lower center",
        ncol=7,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.4,
    )
    return _save_figure(figure, output_dir / "fig2_skew_stability")


def _plot_domain_shift(
    *,
    domain_rows: list[dict[str, str]],
    within_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    import numpy as np

    _mpl, plt = _configure_matplotlib()
    values: dict[str, tuple[Any, Any]] = {}
    for phase in PHASES:
        within = np.array(
            [
                float(row["split_half_jensen_shannon_divergence"])
                for row in within_rows
                if row["phase"] == phase
            ]
        )
        between = np.array(
            [
                float(row["jensen_shannon_divergence"])
                for row in domain_rows
                if row["phase"] == phase
            ]
        )
        if not len(within) or not len(between):
            raise ValueError(f"JSD tables lack {phase} observations")
        values[phase] = (within, between)

    all_values = np.concatenate(
        [array for phase_values in values.values() for array in phase_values]
    )
    lower = 10 ** (np.floor(np.log10(all_values.min())) - 0.05)
    upper = 10 ** (np.ceil(np.log10(all_values.max())) + 0.05)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(5.8, 3.15),
        sharey=True,
        layout="constrained",
    )
    rng = np.random.default_rng(7)
    colors = ["#8A8A8A", "#D55E00"]
    for panel, (axis, phase) in enumerate(zip(axes, PHASES, strict=True)):
        within, between = values[phase]
        boxplot = axis.boxplot(
            [within, between],
            positions=[0, 1],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            tick_labels=["Within-domain\nsplit halves", "Between\ndomains"],
            medianprops={"color": "#222222", "linewidth": 1.1},
            whiskerprops={"color": "#666666", "linewidth": 0.8},
            capprops={"color": "#666666", "linewidth": 0.8},
        )
        for patch, color in zip(boxplot["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.18)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.0)
        for position, (data, color) in enumerate(
            zip((within, between), colors, strict=True)
        ):
            jitter = rng.uniform(-0.14, 0.14, len(data))
            axis.scatter(
                position + jitter,
                data,
                s=9,
                color=color,
                alpha=0.48,
                edgecolors="none",
                zorder=2,
            )
        ratio = float(between.mean() / within.mean())
        axis.text(
            0.97,
            0.95,
            f"mean ratio = {ratio:.1f}×",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        axis.set_title(f"({chr(ord('a') + panel)}) {phase.capitalize()}", loc="left")
        axis.set_yscale("log")
        axis.set_ylim(lower, upper)
        axis.grid(axis="y", which="both", color="#E6E6E6", linewidth=0.5)
        axis.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Jensen–Shannon divergence")
    return _save_figure(figure, output_dir / "fig3_domain_shift")


def _write_figure_notes(path: Path, *, run_id: str) -> None:
    text = f"""# H1 figures: `{run_id}`

These figures are generated from the immutable H1 metric tables. PNG files are
450 DPI for review; PDFs retain vector text and lines for publication.

## Figure 1 — Fast-tier capacity versus expert-demand coverage

Layer-local expert ranks are computed before averaging across the 16 layers.
The curves therefore never combine expert namespaces. The top axis converts
resident experts per layer to aggregate BF16 capacity using the inspected
12 MiB expert size. Coverage counts expert selections, not fully resident
token routes.

## Figure 2 — Skew–stability operating map

The dashed lines are the preregistered top-8 skew and Jaccard thresholds.
Filled points also pass the 0.80 lagged/oracle threshold. Only L6 and L9 pass
all three conditions for mixed decode. Per-domain points are pilot evidence
because some scopes contain only three or four complete 512-token windows.

## Figure 3 — Domain shift versus sampling drift

Between-domain routing divergence is compared with within-domain split-half
divergence on a log scale. The mean ratios are descriptive pilot statistics,
not confidence intervals.

## Human visual-review checkpoint

- [ ] Axes, units, phase separation, and selection-versus-token semantics are
      clear.
- [ ] Curves and points agree with the machine-readable report.
- [ ] No important outlier, layer regime, saturation point, or domain confound
      is hidden by aggregation.
- [ ] The reviewer records whether the figures support, weaken, or revise the
      experiment interpretation before the next hypothesis starts.
"""
    path.write_text(text, encoding="utf-8")


def plot_h1(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    directory = Path(run_dir)
    analysis_dir = directory / "analysis" / "h1"
    figure_dir = (
        Path(output_dir) if output_dir is not None else analysis_dir / "figures"
    )
    model_report_path = directory / "model_report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    routers = model_report["routers"]
    layers = sorted(int(router["layer_id"]) for router in routers)
    expert_counts = {int(router["num_experts"]) for router in routers}
    expert_sizes = {int(router["expert_bytes_each"]) for router in routers}
    if len(expert_counts) != 1 or len(expert_sizes) != 1:
        raise ValueError(
            "H1 plots currently require the same expert count and size at every layer"
        )
    num_experts = expert_counts.pop()
    expert_bytes = expert_sizes.pop()

    input_paths = {
        "rank_frequency": analysis_dir / "rank_frequency.csv",
        "popularity": analysis_dir / "popularity.csv",
        "window_stability": analysis_dir / "window_stability.csv",
        "domain_comparison": analysis_dir / "domain_comparison.csv",
        "within_domain_stability": analysis_dir / "within_domain_stability.csv",
        "model_report": model_report_path,
    }
    rank_rows = _read_csv(input_paths["rank_frequency"])
    popularity_rows = _read_csv(input_paths["popularity"])
    window_rows = _read_csv(input_paths["window_stability"])
    domain_rows = _read_csv(input_paths["domain_comparison"])
    within_rows = _read_csv(input_paths["within_domain_stability"])

    outputs: list[Path] = []
    outputs.extend(
        _plot_capacity_coverage(
            rows=rank_rows,
            layers=layers,
            num_experts=num_experts,
            expert_bytes=expert_bytes,
            output_dir=figure_dir,
        )
    )
    outputs.extend(
        _plot_skew_stability(
            popularity_rows=popularity_rows,
            window_rows=window_rows,
            experiment_config=experiment_config,
            output_dir=figure_dir,
        )
    )
    outputs.extend(
        _plot_domain_shift(
            domain_rows=domain_rows,
            within_rows=within_rows,
            output_dir=figure_dir,
        )
    )
    notes_path = figure_dir / "FIGURES.md"
    run_manifest = json.loads(
        (directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    _write_figure_notes(notes_path, run_id=run_manifest["run_id"])
    outputs.append(notes_path)

    manifest = {
        "run_id": run_manifest["run_id"],
        "figure_grade": "pilot",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in outputs
        ],
        "semantics": {
            "capacity_curve": "expert-selection coverage",
            "layer_aggregation": "equal mean of layer-local ranked distributions",
            "domain_shift": "descriptive pilot comparison, no confidence interval",
        },
    }
    manifest_path = figure_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
