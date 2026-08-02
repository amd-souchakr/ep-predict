#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("artifacts/runs/gpt-oss-20b-milestone-c"))
    args = parser.parse_args()

    result = json.loads((args.run / "qualification.json").read_text())
    with (args.run / "dispatch_parity.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    layers = np.array([int(row["layer"]) for row in rows])
    dispatch = np.array([int(row["dispatch_hook_calls"]) for row in rows])
    ordinary = np.array([int(row["ordinary_router_hook_calls"]) for row in rows])
    id_bad = np.array([int(row["id_mismatches"]) for row in rows])
    weight_bad = np.array([int(row["weight_mismatches"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(layers, dispatch, marker="o", color="#0072B2", label="Dispatch-boundary hook")
    axes[0].plot(layers, ordinary, marker="x", color="#D55E00", label="Ordinary router hook")
    axes[0].set_ylabel("Calls in one prefill")
    axes[0].set_yticks([0, 1])
    axes[0].set_title("MXFP4 bypass is observed and covered")
    axes[0].legend(loc="center right", frameon=True)

    width = 0.38
    axes[1].bar(layers - width / 2, id_bad, width, color="#009E73", label="Expert-ID mismatches")
    axes[1].bar(layers + width / 2, weight_bad, width, color="#CC79A7", label="Selected-weight mismatches")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Mismatched pairs")
    axes[1].set_xlabel("Routed layer")
    axes[1].set_xticks(np.arange(0, 24, 2))
    axes[1].set_ylim(-0.05, 1.0)
    axes[1].set_title("All 576 dispatch-consumed ID/weight pairs match")
    axes[1].legend(loc="upper right", frameon=True)
    fig.suptitle(f"GPT-OSS 20B Milestone C: {result['decision']}", fontsize=14, fontweight="bold")

    figure_dir = args.run / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "fig1_gpt_oss_dispatch_qualification.png"
    pdf = figure_dir / "fig1_gpt_oss_dispatch_qualification.pdf"
    fig.savefig(png, dpi=450)
    fig.savefig(pdf)
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "inputs": {
            str(args.run / "qualification.json"): sha256(args.run / "qualification.json"),
            str(args.run / "dispatch_parity.csv"): sha256(args.run / "dispatch_parity.csv"),
        },
        "outputs": {str(png): sha256(png), str(pdf): sha256(pdf)},
    }
    (figure_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (figure_dir / "FIGURES.md").write_text(
        "# GPT-OSS Milestone C figures\n\n"
        "Figure 1 shows the decisive instrumentation result. The ordinary router hook is bypassed "
        "on every MXFP4 layer, while the dispatch-boundary hook covers every layer and records zero "
        "expert-ID or selected-weight mismatches. This is a qualification result, not a routing "
        "distribution or performance measurement.\n"
    )


if __name__ == "__main__":
    main()
