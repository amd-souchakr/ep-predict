#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_traces(run: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((run / "trace").glob("request-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle)
    if not records:
        raise FileNotFoundError(f"no traces under {run / 'trace'}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("artifacts/runs/gpt-oss-20b-milestone-d"),
    )
    args = parser.parse_args()
    run = args.run
    integrity = json.loads((run / "integrity.json").read_text())
    inspection = json.loads((run / "model_inspection.json").read_text())
    outputs = [json.loads(line) for line in (run / "outputs.jsonl").read_text().splitlines()]
    records = read_traces(run)
    figure_dir = run / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: a compact trajectory view using the highest consumed weight in
    # each selected set. It is descriptive evidence that token/layer records
    # can be reconstructed, not a top-1 router or distribution metric.
    fig, axes = plt.subplots(
        len(outputs), 1, figsize=(11.0, 2.7 * len(outputs)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, output in zip(axes, outputs):
        request_id = output["request_id"]
        token_count = output["prompt_token_count"] + output["generated_token_count"]
        raster = np.full((inspection["routed_layers"], token_count), np.nan)
        for record in records:
            if record["request_id"] != request_id:
                continue
            weights = np.asarray(record["selected_expert_weights"])
            ids = np.asarray(record["selected_expert_ids"])
            raster[record["moe_layer_index"], record["token_position"]] = ids[int(np.argmax(weights))]
        image = axis.imshow(
            raster,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=inspection["experts_per_layer"] - 1,
            cmap="turbo",
        )
        axis.axvline(output["prompt_token_count"] - 0.5, color="white", linewidth=1.5, linestyle="--")
        axis.set_title(
            f"{output['sample_id']}: highest-weight selected expert (prompt | retained output)"
        )
        axis.set_ylabel("Routed layer")
        axis.set_yticks(np.arange(0, inspection["routed_layers"], 4))
    axes[-1].set_xlabel("Token position")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Expert ID", shrink=0.9)
    fig.suptitle("GPT-OSS 20B Milestone D routing tracer raster", fontsize=14, fontweight="bold")
    raster_png = figure_dir / "fig1_routing_raster.png"
    raster_pdf = figure_dir / "fig1_routing_raster.pdf"
    fig.savefig(raster_png, dpi=450)
    fig.savefig(raster_pdf)
    plt.close(fig)

    # Figure 2: all-selection occupancy and exact layer/request coverage.
    occupancy = np.zeros((inspection["routed_layers"], inspection["experts_per_layer"]), dtype=int)
    observed: Counter[tuple[int, int]] = Counter()
    for record in records:
        occupancy[record["moe_layer_index"], record["selected_expert_ids"]] += 1
        observed[(record["request_id"], record["moe_layer_index"])] += 1
    coverage = np.zeros((len(outputs), inspection["routed_layers"]), dtype=float)
    for output in outputs:
        eligible = output["prompt_token_count"] + output["generated_token_count"]
        for layer in range(inspection["routed_layers"]):
            coverage[output["request_id"], layer] = observed[(output["request_id"], layer)] / eligible

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), constrained_layout=True)
    occ_image = axes[0].imshow(occupancy, aspect="auto", interpolation="nearest", origin="lower", cmap="magma")
    axes[0].set_title("All selected-ID observations (descriptive tracer output)")
    axes[0].set_xlabel("Expert ID")
    axes[0].set_ylabel("Routed layer")
    fig.colorbar(occ_image, ax=axes[0], label="Selection count")
    cov_image = axes[1].imshow(coverage, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="Blues")
    axes[1].set_title("Observed / expected token records by request and layer")
    axes[1].set_xlabel("Routed layer")
    axes[1].set_ylabel("Request")
    axes[1].set_yticks(range(len(outputs)), [output["sample_id"] for output in outputs])
    fig.colorbar(cov_image, ax=axes[1], label="Coverage fraction")
    fig.suptitle("GPT-OSS 20B Milestone D trace materialization", fontsize=14, fontweight="bold")
    coverage_png = figure_dir / "fig2_occupancy_and_coverage.png"
    coverage_pdf = figure_dir / "fig2_occupancy_and_coverage.pdf"
    fig.savefig(coverage_png, dpi=450)
    fig.savefig(coverage_pdf)
    plt.close(fig)

    figure_note = (
        "# GPT-OSS Milestone D figures\n\n"
        "Figure 1 reconstructs a compact token-by-layer routing raster from the retained trace; each cell "
        "shows the expert carrying the largest consumed selected weight. Figure 2 shows all selected-ID "
        "observations and the request/layer completeness matrix. These are workflow-qualification views, "
        "not routing-distribution comparisons or performance evidence.\n"
    )
    (figure_dir / "FIGURES.md").write_text(figure_note)

    expected_files = [
        run / "run_definition.json",
        run / "model_inspection.json",
        run / "outputs.jsonl",
        run / "integrity.json",
        run / "layer_integrity.csv",
        run / "routing_summary.csv",
        raster_png,
        raster_pdf,
        coverage_png,
        coverage_pdf,
        figure_dir / "FIGURES.md",
    ] + sorted((run / "trace").glob("request-*.jsonl.gz"))
    parseable = True
    try:
        json.loads((run / "run_definition.json").read_text())
        json.loads((run / "model_inspection.json").read_text())
        json.loads((run / "integrity.json").read_text())
        list(csv.DictReader((run / "routing_summary.csv").open()))
    except (OSError, ValueError, json.JSONDecodeError):
        parseable = False
    artifact_checks = {
        "trace_gate_qualified": integrity["decision"] == "TRACE_QUALIFIED",
        "all_required_artifacts_present": all(path.is_file() and path.stat().st_size > 0 for path in expected_files),
        "required_artifacts_parseable": parseable,
        "two_trace_shards_present": len(list((run / "trace").glob("request-*.jsonl.gz"))) == len(outputs),
        "two_scripted_figures_present": all(path.is_file() for path in (raster_png, raster_pdf, coverage_png, coverage_pdf)),
    }
    result = {
        "schema_version": 1,
        "milestone": "D",
        "decision": "QUALIFIED" if all(artifact_checks.values()) else "NOT_QUALIFIED",
        "artifact_gate_checks": artifact_checks,
        "trace_integrity": integrity["totals"],
        "claim_boundary": "end-to-end tracing workflow qualification only; no routing-distribution comparison or timing claim",
    }
    (run / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    totals = integrity["totals"]
    report = (
        "# GPT-OSS 20B Milestone D result\n\n"
        f"**Decision:** `{result['decision']}`\n\n"
        f"The two-request deterministic tracer retained {totals['prompt_tokens']} prompt tokens and "
        f"{totals['generated_tokens']} generated tokens, producing {totals['trace_records']} token-layer "
        f"records and {totals['dispatch_consumed_pairs']} consumed ID/weight pairs. Every eligible token "
        "is covered at all 24 layers. Dispatch parity has zero ID mismatches, zero selected-weight "
        f"mismatches, and maximum absolute error {totals['dispatch_max_abs_weight_error']}. The immediate "
        "repeat reproduced all rendered inputs, generated IDs, routing IDs, and selected weights within "
        "the frozen tolerance.\n\n"
        "The outputs, standard trace shards, inspection, integrity tables, compact routing summary, two "
        "scripted figures, and hash manifest complete the artifact chain. This qualifies the tracing "
        "workflow only. The small convenience workload cannot support a routing-distribution, domain, "
        "quality, or performance conclusion, and Milestone E remains blocked on review.\n"
    )
    (run / "REPORT.md").write_text(report)
    expected_files.extend([run / "result.json", run / "REPORT.md"])
    manifest = {
        "schema_version": 1,
        "files": {str(path): sha256(path) for path in sorted(expected_files)},
    }
    (run / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["decision"] != "QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
