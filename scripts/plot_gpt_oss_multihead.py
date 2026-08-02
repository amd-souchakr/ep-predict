#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPLORATORY = Path("artifacts/runs/gpt-oss-20b-mtp-head-exploratory")
CONFIRMATION = Path("artifacts/runs/gpt-oss-20b-mtp-head-confirmation")


def learning_curve() -> None:
    rows = list(csv.DictReader((EXPLORATORY / "learning_curve.csv").open()))
    series = {
        "weighted": ("Weighted route", "#55a868"),
        "binary": ("Binary route", "#c44e52"),
        "weighted_binary": ("Weighted + binary", "#4c72b0"),
        "table": ("Transition table", "#8172b3"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for mode, (label, color) in series.items():
        baseline = "transition" if mode == "table" else "pairwise_head"
        selected = [
            row
            for row in rows
            if row["domain"] == "ALL"
            and row["baseline"] == baseline
            and row["feature_mode"] == mode
        ]
        grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
        for row in selected:
            grouped[(int(row["training_requests"]), int(row["fold"]))].append(row)
        x_values = sorted({key[0] for key in grouped})
        for axis, metric in zip(
            axes, ("selection_coverage", "complete_route_coverage"), strict=True
        ):
            means = []
            errors = []
            for requests in x_values:
                folds = [
                    np.mean([float(row[metric]) for row in grouped[(requests, fold)]])
                    for fold in range(4)
                ]
                means.append(100 * np.mean(folds))
                errors.append(100 * np.std(folds, ddof=1))
            axis.errorbar(
                [value * 16 for value in x_values],
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=color,
                label=label,
            )
    axes[0].axhline(82, color="black", linestyle="--", linewidth=0.9)
    axes[1].axhline(50, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_title("Selection coverage")
    axes[1].set_title("Complete-route coverage")
    for axis in axes:
        axis.set_xlabel("Unique decode training tokens")
        axis.set_ylabel("Decode K=8 coverage (%)")
        axis.grid(alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("MTP-style layer-pair route-head learning curve", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.72))
    directory = EXPLORATORY / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "fig1_learning_curve.png", dpi=450, bbox_inches="tight")
    fig.savefig(directory / "fig1_learning_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def confirmation() -> None:
    decision = json.loads(
        (CONFIRMATION / "analysis" / "multihead_confirmation" / "decision.json").read_text()
    )
    rows = decision["lookaheads"]
    deltas = np.asarray([row["delta"] for row in rows])
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis, metric, title in (
        (axes[0], "selection_coverage", "Selection coverage"),
        (axes[1], "complete_route_coverage", "Complete-route coverage"),
    ):
        axis.bar(
            deltas - width,
            [100 * row[f"cheap_{metric}"] for row in rows],
            width,
            label="Strong cheap comparator",
            color="#c7c7c7",
        )
        axis.bar(
            deltas,
            [100 * row[f"transition_{metric}"] for row in rows],
            width,
            label="Transition table",
            color="#8172b3",
        )
        axis.bar(
            deltas + width,
            [100 * row[f"learned_{metric}"] for row in rows],
            width,
            label="Frozen route heads",
            color="#4c72b0",
        )
        axis.set_xticks(deltas)
        axis.set_xlabel("Lookahead Δ")
        axis.set_ylabel("Fresh-request decode K=8 coverage (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("64-request frozen-head confirmation", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.72))
    directory = CONFIRMATION / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "fig1_confirmation.png", dpi=450, bbox_inches="tight")
    fig.savefig(directory / "fig1_confirmation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    learning_curve()
    confirmation()
