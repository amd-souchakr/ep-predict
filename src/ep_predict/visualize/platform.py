from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ep_predict.visualize.h1 import (
    _configure_matplotlib,
    _read_csv,
    _save_figure,
    _sha256,
)
from ep_predict.tracing.storage import write_json


def plot_platform_comparison(
    run_dir: str | Path,
    _experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    analysis = run / "analysis" / "platform_comparison"
    layer_path = analysis / "h1_layer_trends.csv"
    horizon_path = analysis / "h2_horizon_trends.csv"
    summary_path = analysis / "summary.json"
    for path in (layer_path, horizon_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"run platform comparison first: {path}")

    layers = _read_csv(layer_path)
    horizons = _read_csv(horizon_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mi_requests = int(summary["scope"]["mi355x_requests"])
    nv_requests = int(summary["scope"]["nvidia_reference_requests"])
    destination = Path(output_dir) if output_dir else analysis / "figures"
    destination.mkdir(parents=True, exist_ok=True)

    _mpl, plt = _configure_matplotlib()
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.2),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        layout="constrained",
    )
    h1_axis, h2_axis = axes[0]
    h1_residual_axis, h2_residual_axis = axes[1]
    layer_ids = [int(row["layer_id"]) for row in layers]
    h1_axis.plot(
        layer_ids,
        [float(row["nvidia_top8_coverage"]) for row in layers],
        marker="o",
        color="#666666",
        label=f"NVIDIA ({nv_requests} requests)",
    )
    h1_axis.plot(
        layer_ids,
        [float(row["mi355x_top8_coverage"]) for row in layers],
        marker="s",
        color="#0072B2",
        linestyle="--",
        label=f"MI355X ({mi_requests} requests)",
    )
    h1_axis.axhline(0.25, color="#B0B0B0", linestyle=":", linewidth=0.9)
    h1_axis.set_title("(a) H1 layerwise skew", loc="left")
    h1_axis.set_ylabel("Top-8 popularity coverage")
    h1_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    h1_axis.legend(frameon=False, loc="best")
    h1_residual_axis.bar(
        layer_ids,
        [float(row["top8_coverage_difference"]) for row in layers],
        color="#0072B2",
        width=0.65,
    )
    h1_residual_axis.axhline(0, color="#555555", linewidth=0.7)
    h1_residual_axis.set_xlabel("MoE layer")
    h1_residual_axis.set_ylabel("MI−NV")
    h1_residual_axis.set_xticks([0, 3, 6, 9, 12, 15])
    h1_residual_axis.yaxis.set_major_formatter(PercentFormatter(1.0))

    deltas = [int(row["delta"]) for row in horizons]
    h2_axis.plot(
        deltas,
        [float(row["nvidia_selection_gain"]) for row in horizons],
        marker="o",
        color="#666666",
        label=f"NVIDIA ({nv_requests} requests)",
    )
    h2_axis.plot(
        deltas,
        [float(row["mi355x_selection_gain"]) for row in horizons],
        marker="s",
        color="#D55E00",
        linestyle="--",
        label=f"MI355X ({mi_requests} requests)",
    )
    h2_axis.axhline(0.03, color="#B0B0B0", linestyle=":", linewidth=0.9)
    h2_axis.set_title("(b) H2 conditional trajectory gain", loc="left")
    h2_axis.set_ylabel("Transition selection gain over static")
    h2_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    h2_axis.legend(frameon=False, loc="best")
    h2_residual_axis.bar(
        deltas,
        [float(row["selection_gain_difference"]) for row in horizons],
        color="#D55E00",
        width=0.65,
    )
    h2_residual_axis.axhline(0, color="#555555", linewidth=0.7)
    h2_residual_axis.set_xlabel("Lookahead, $\\Delta$")
    h2_residual_axis.set_ylabel("MI−NV")
    h2_residual_axis.set_xticks([1, 3, 6, 9, 12, 15])
    h2_residual_axis.yaxis.set_major_formatter(PercentFormatter(1.0))

    outputs = _save_figure(
        figure, destination / "fig1_mi355x_nvidia_derived_trends"
    )
    plt.close(figure)

    notes = destination / "FIGURES.md"
    notes.write_text(
        "\n".join(
            [
                "# MI355X derived-platform figure review",
                "",
                "**Raw-trace interchangeability:** "
                f"`{summary['interchangeability']}`",
                "",
                f"The NVIDIA and MI355X curves summarize {nv_requests} and "
                f"{mi_requests} requests, respectively. Matched-workload and "
                f"split integrity: {summary['scope']['matched_workload_and_split']}.",
                "",
                "## Human review checklist",
                "",
                "- [ ] Layer and horizon axes match the machine-readable tables.",
                "- [ ] The 25% H1 and 3-point H2 reference lines are labeled as historical gates, not parity thresholds.",
                "- [ ] Request counts and matched-scope integrity agree with the summary.",
                "- [ ] The reviewer accepts, narrows, or rejects proceeding to MI355X H4 calibration.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(notes)
    manifest = {
        "run_id": run.name,
        "figure_grade": "descriptive_platform_trend",
        "interchangeability": summary["interchangeability"],
        "inputs": {
            "h1_layer_trends": {"path": str(layer_path), "sha256": _sha256(layer_path)},
            "h2_horizon_trends": {"path": str(horizon_path), "sha256": _sha256(horizon_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        },
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in outputs
        ],
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
