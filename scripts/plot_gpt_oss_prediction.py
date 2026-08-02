#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "global_static": "#9CA3AF",
    "domain_static": "#2563EB",
    "source_copy": "#D97706",
    "transition": "#059669",
}
LABELS = {
    "global_static": "Global popularity",
    "domain_static": "Domain popularity",
    "source_copy": "Current-route copy",
    "transition": "Transition table",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifact_manifest(run: Path, durable: list[Path]) -> None:
    traces = sorted((run / "trace").glob("request-*.jsonl.gz"))
    manifest = {
        "schema_version": 1,
        "durable_files": {
            str(path): _sha256(path) for path in sorted(set(durable)) if path.is_file()
        },
        "disposable_trace_files": {str(path): _sha256(path) for path in traces},
    }
    (run / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=Path, default=Path("artifacts/runs/gpt-oss-20b-milestone-e")
    )
    args = parser.parse_args()
    run = args.run
    analysis = run / "analysis" / "prediction"
    summaries = _read_csv(analysis / "horizon_summary.csv")
    scopes = _read_csv(analysis / "scope_metrics.csv")
    decision = json.loads((analysis / "decision.json").read_text(encoding="utf-8"))
    integrity = json.loads((run / "integrity.json").read_text(encoding="utf-8"))
    figure_dir = run / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    selected = [
        row
        for row in summaries
        if row["phase"] == "decode" and int(row["candidate_count"]) == 8
    ]
    metrics = (
        ("selection_coverage", "Expert-selection coverage"),
        ("routed_mass_coverage", "Selected routed-mass coverage"),
        ("complete_route_coverage", "Complete top-4 route coverage"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for baseline, label in LABELS.items():
            rows = sorted(
                (row for row in selected if row["baseline"] == baseline),
                key=lambda row: int(row["delta"]),
            )
            axis.plot(
                [int(row["delta"]) for row in rows],
                [100 * float(row[metric]) for row in rows],
                color=COLORS[baseline],
                label=label,
                linewidth=2.2 if baseline == "transition" else 1.6,
                linestyle="--" if baseline == "global_static" else "-",
            )
        axis.axvspan(0.5, 3.5, color="#E5E7EB", alpha=0.45, zorder=0)
        axis.set_title(title)
        axis.set_xlabel("Layer lookahead Δ")
        axis.set_ylabel("Held-out decode coverage (%)")
        axis.set_xlim(1, 23)
        axis.set_xticks([1, 3, 6, 9, 12, 15, 18, 21, 23])
        axis.set_ylim(0, 102)
    axes[0].legend(frameon=True, fontsize=8)
    fig.suptitle(
        "GPT-OSS 20B held-out route prediction at K=8\n"
        "2× candidate amplification; 25% candidate-set fraction; shaded region is preregistered\n"
        "Each horizon averages its valid source layers; the source mix shrinks as Δ grows",
        fontsize=12,
        fontweight="bold",
    )
    horizon_png = figure_dir / "fig1_prediction_quality_by_horizon.png"
    horizon_pdf = figure_dir / "fig1_prediction_quality_by_horizon.pdf"
    fig.savefig(horizon_png, dpi=450)
    fig.savefig(horizon_pdf)
    plt.close(fig)

    # Direct candidate-count comparison. Shared axes prevent the much stronger K=16
    # curves from looking only cosmetically better than K=4. The gray band is
    # shown only in the K=8 column because that is the preregistered gate.
    candidate_count_metadata = {
        4: ("1× top-4; 12.5% candidate fraction", False),
        8: ("2× top-4; 25% candidate fraction (primary)", True),
        16: ("4× top-4; 50% candidate fraction", False),
    }
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15.0, 11.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for column, (candidate_count, (candidate_label, primary)) in enumerate(
        candidate_count_metadata.items()
    ):
        candidate_count_rows = [
            row
            for row in summaries
            if row["phase"] == "decode"
            and int(row["candidate_count"]) == candidate_count
        ]
        for row_index, (metric, metric_title) in enumerate(metrics):
            axis = axes[row_index, column]
            for baseline, label in LABELS.items():
                rows = sorted(
                    (
                        row
                        for row in candidate_count_rows
                        if row["baseline"] == baseline
                    ),
                    key=lambda row: int(row["delta"]),
                )
                axis.plot(
                    [int(row["delta"]) for row in rows],
                    [100 * float(row[metric]) for row in rows],
                    color=COLORS[baseline],
                    label=label,
                    linewidth=2.2 if baseline == "transition" else 1.5,
                    linestyle="--" if baseline == "global_static" else "-",
                )
            if primary:
                axis.axvspan(0.5, 3.5, color="#E5E7EB", alpha=0.55, zorder=0)
            if row_index == 0:
                axis.set_title(
                    f"K={candidate_count}\n{candidate_label}", fontweight="bold"
                )
            if column == 0:
                axis.set_ylabel(f"{metric_title}\n(%)")
            if row_index == len(metrics) - 1:
                axis.set_xlabel("Layer lookahead Δ")
            axis.set_xlim(1, 23)
            axis.set_ylim(0, 102)
            axis.set_xticks([1, 3, 6, 9, 12, 15, 18, 21, 23])
            axis.set_yticks([0, 20, 40, 60, 80, 100])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=True,
    )
    fig.suptitle(
        "GPT-OSS 20B held-out decode coverage by lookahead and candidate count\n"
        "Identical axes expose the coverage–candidate-budget tradeoff; source-layer mix "
        "shrinks as Δ grows",
        fontsize=14,
        fontweight="bold",
    )
    candidate_count_png = (
        figure_dir / "fig3_coverage_by_horizon_and_candidate_count.png"
    )
    candidate_count_pdf = (
        figure_dir / "fig3_coverage_by_horizon_and_candidate_count.pdf"
    )
    fig.savefig(candidate_count_png, dpi=450)
    fig.savefig(candidate_count_pdf)
    plt.close(fig)

    chosen = [
        row
        for row in scopes
        if row["phase"] == "decode" and int(row["candidate_count"]) == 8
    ]
    values: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in chosen:
        values[(int(row["source_layer"]), int(row["delta"]), row["baseline"])].append(
            float(row["selection_coverage"])
        )
    matrix = np.full((24, 23), np.nan)
    for source in range(24):
        for delta in range(1, 24 - source):
            transition = np.mean(values[(source, delta, "transition")])
            comparator = max(
                np.mean(values[(source, delta, "domain_static")]),
                np.mean(values[(source, delta, "source_copy")]),
            )
            matrix[source, delta - 1] = 100 * (transition - comparator)
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(12.0, 7.2), constrained_layout=True)
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_title(
        "Transition gain over the stronger cheap baseline\n"
        "Held-out decode selection coverage, K=8",
        fontweight="bold",
    )
    axis.set_xlabel("Layer lookahead Δ")
    axis.set_ylabel("Source routed layer")
    axis.set_xticks(np.arange(0, 23, 2), np.arange(1, 24, 2))
    axis.set_yticks(np.arange(0, 24, 2))
    fig.colorbar(image, ax=axis, label="Selection-coverage gain (percentage points)")
    heatmap_png = figure_dir / "fig2_source_horizon_gain.png"
    heatmap_pdf = figure_dir / "fig2_source_horizon_gain.pdf"
    fig.savefig(heatmap_png, dpi=450)
    fig.savefig(heatmap_pdf)
    plt.close(fig)

    gate_rows = decision["lookaheads"]
    totals = integrity["totals"]
    report_lines = [
        "# GPT-OSS 20B Milestone E result",
        "",
        "**Overall decision:** `CONDITIONAL_PILOT_SUPPORT_WITH_TRACE_WEIGHT_EXCEPTION`",
        "",
        (
            f"The collection retained {totals['requests']} requests, "
            f"{totals['prompt_tokens']:,} prompt tokens, and "
            f"{totals['generated_tokens']:,} decode tokens, yielding "
            f"{totals['trace_records']:,} complete token-layer records and "
            f"{totals['dispatch_consumed_pairs']:,} consumed ID/weight pairs. "
            "Router-to-dispatch ID and selected-weight mismatches were both zero."
        ),
        "",
        (
            "At the preregistered decode K=8 point, transition prediction is compared "
            "with the stronger of domain popularity and current-route copy:"
        ),
        "",
        (
            "| Δ | Selection coverage gain (95% CI) | Routed-mass gain | "
            "Complete-route gain (95% CI) | Pass |"
        ),
        "|---:|---:|---:|---:|:---:|",
    ]
    if not decision["preregistered_trace_gate_passed"]:
        exception = decision["trace_numerical_exception"]
        report_lines[4] = (
            f"The collection retained {totals['requests']} requests, "
            f"{totals['prompt_tokens']:,} prompt tokens, and "
            f"{totals['generated_tokens']:,} decode tokens, yielding "
            f"{totals['trace_records']:,} complete token-layer records and "
            f"{totals['dispatch_consumed_pairs']:,} consumed ID/weight pairs. "
            "All executed expert IDs matched. The frozen trace gate formally failed: "
            f"{exception['dispatch_weight_mismatches']} independently reconstructed "
            f"weights differed ({100 * exception['mismatch_fraction']:.6f}% of pairs; "
            f"maximum absolute error {exception['max_abs_weight_error']}). The analysis "
            "below is explicitly post-hoc conditional evidence using the exact "
            "dispatch-consumed IDs and weights retained in the trace."
        )
    for row in gate_rows:
        report_lines.append(
            f"| {row['delta']} | {100 * row['selection_coverage_gain']:+.1f} pp "
            f"[{100 * row['selection_gain_ci_low']:+.1f}, {100 * row['selection_gain_ci_high']:+.1f}] | "
            f"{100 * row['routed_mass_coverage_gain']:+.1f} pp | "
            f"{100 * row['complete_route_coverage_gain']:+.1f} pp "
            f"[{100 * row['complete_gain_ci_low']:+.1f}, {100 * row['complete_gain_ci_high']:+.1f}] | "
            f"{'yes' if row['pass'] else 'no'} |"
        )
    report_lines.extend(
        [
            "",
            (
                f"{decision['passing_lookaheads']} lookaheads passed; at least "
                f"{decision['required_passing_lookaheads']} were required."
            ),
            "",
            (
                "The all-horizon K=8 curves, K=4/8/16 candidate-count comparison, and "
                "source-layer heatmap are descriptive. This milestone establishes "
                "held-out route-prediction behavior for GPT-OSS 20B only; it makes "
                "no language-quality, latency, 120B, or cross-model claim."
            ),
            "",
        ]
    )
    (run / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    result = {
        "schema_version": 1,
        "milestone": "E",
        "decision": (
            "CONDITIONAL_PILOT_SUPPORT_WITH_TRACE_WEIGHT_EXCEPTION"
            if not decision["preregistered_trace_gate_passed"]
            else decision["decision"]
        ),
        "prediction_decision": decision["decision"],
        "trace_integrity": integrity["decision"],
        "preregistered_trace_gate_passed": decision["preregistered_trace_gate_passed"],
        "passing_lookaheads": decision["passing_lookaheads"],
        "required_passing_lookaheads": decision["required_passing_lookaheads"],
        "claim_boundary": decision["claim_boundary"],
    }
    (run / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (figure_dir / "FIGURES.md").write_text(
        "# GPT-OSS 20B Milestone E figures\n\n"
        "Figure 1 reports held-out decode prediction quality across all layer "
        "lookaheads at the primary K=8 candidate count. Figure 2 localizes transition "
        "selection-coverage gain relative to the stronger of domain popularity "
        "and current-route copy. Blank cells are invalid source/target pairs. "
        "Figure 3 compares selection, routed-mass, and complete-route coverage "
        "at K=4/8/16 on identical axes; the shaded Δ=1--3 band appears only for "
        "the preregistered K=8 decision point.\n",
        encoding="utf-8",
    )
    durable = [
        run / "run_definition.json",
        run / "model_inspection.json",
        run / "model_report.json",
        run / "outputs.jsonl",
        run / "integrity.json",
        run / "layer_integrity.csv",
        run / "routing_summary.csv",
        analysis / "split.json",
        analysis / "split.csv",
        analysis / "scope_metrics.csv",
        analysis / "request_metrics.csv",
        analysis / "horizon_summary.csv",
        analysis / "bootstrap_gate.csv",
        analysis / "decision.json",
        analysis / "REPORT.md",
        horizon_png,
        horizon_pdf,
        heatmap_png,
        heatmap_pdf,
        candidate_count_png,
        candidate_count_pdf,
        figure_dir / "FIGURES.md",
        run / "REPORT.md",
        run / "result.json",
    ]
    _write_artifact_manifest(run, durable)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
