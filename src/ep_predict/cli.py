from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ep_predict.config import load_toml


def _inspect(args: argparse.Namespace) -> int:
    from ep_predict.modeling import (
        inspect_loaded_model,
        load_model_and_tokenizer,
        print_model_summary,
    )
    from ep_predict.tracing.storage import write_json

    config = load_toml(args.config)
    model, _tokenizer = load_model_and_tokenizer(config)
    report, _routers = inspect_loaded_model(
        model,
        router_name_contains=config.get("router_name_contains"),
    )
    if args.output:
        write_json(args.output, report)
    print_model_summary(report)
    return 0


def _prepare_dataset(args: argparse.Namespace) -> int:
    from ep_predict.data.standard import materialize_standard_workload

    manifest = materialize_standard_workload(load_toml(args.config))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _collect(args: argparse.Namespace) -> int:
    from ep_predict.collect import collect_run

    manifest = collect_run(
        load_toml(args.model_config),
        load_toml(args.experiment_config),
        limit=args.limit,
    )
    print(json.dumps({"run_id": manifest["run_id"], "state": manifest["state"]}))
    return 0


def _analyze_h1(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h1 import analyze_h1

    summary = analyze_h1(args.run, load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h1(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h1 import plot_h1

    manifest = plot_h1(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_h2(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h2 import analyze_h2

    summary = analyze_h2(args.run, load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h2(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h2 import plot_h2

    manifest = plot_h2(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_h3(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h3 import analyze_h3

    summary = analyze_h3(args.run, load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h3(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h3 import plot_h3

    manifest = plot_h3(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _plot_extended_horizon(args: argparse.Namespace) -> int:
    from ep_predict.visualize.extended_horizon import plot_extended_horizon

    manifest = plot_extended_horizon(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ep-predict",
        description="Hook-based MoE routing research tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="load a model and report router/expert structure"
    )
    inspect_parser.add_argument("--config", required=True, type=Path)
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.set_defaults(function=_inspect)

    dataset_parser = subparsers.add_parser(
        "prepare-dataset",
        help="materialize a revision-pinned standard trace workload",
    )
    dataset_parser.add_argument("--config", required=True, type=Path)
    dataset_parser.set_defaults(function=_prepare_dataset)

    collect_parser = subparsers.add_parser(
        "collect", help="collect resumable request-level routing traces"
    )
    collect_parser.add_argument("--model-config", required=True, type=Path)
    collect_parser.add_argument("--experiment-config", required=True, type=Path)
    collect_parser.add_argument(
        "--limit", type=int, help="run only the first N prompts for a smoke test"
    )
    collect_parser.set_defaults(function=_collect)

    analyze_parser = subparsers.add_parser(
        "analyze-h1", help="compute H1 skew and hot-set stability metrics"
    )
    analyze_parser.add_argument("--run", required=True, type=Path)
    analyze_parser.add_argument("--config", required=True, type=Path)
    analyze_parser.set_defaults(function=_analyze_h1)

    plot_parser = subparsers.add_parser(
        "plot-h1", help="generate publication-style H1 figures"
    )
    plot_parser.add_argument("--run", required=True, type=Path)
    plot_parser.add_argument("--config", required=True, type=Path)
    plot_parser.add_argument(
        "--output",
        type=Path,
        help="figure directory (default: RUN/analysis/h1/figures)",
    )
    plot_parser.set_defaults(function=_plot_h1)

    h2_parser = subparsers.add_parser(
        "analyze-h2",
        help="evaluate held-out routing-conditioned future-expert baselines",
    )
    h2_parser.add_argument("--run", required=True, type=Path)
    h2_parser.add_argument("--config", required=True, type=Path)
    h2_parser.set_defaults(function=_analyze_h2)

    h2_plot_parser = subparsers.add_parser(
        "plot-h2", help="generate publication-style H2 decision figures"
    )
    h2_plot_parser.add_argument("--run", required=True, type=Path)
    h2_plot_parser.add_argument("--config", required=True, type=Path)
    h2_plot_parser.add_argument(
        "--output",
        type=Path,
        help="figure directory (default: RUN/analysis/h2/figures)",
    )
    h2_plot_parser.set_defaults(function=_plot_h2)

    h3_parser = subparsers.add_parser(
        "analyze-h3",
        help="train and evaluate the held-out linear hidden-state predictor",
    )
    h3_parser.add_argument("--run", required=True, type=Path)
    h3_parser.add_argument("--config", required=True, type=Path)
    h3_parser.set_defaults(function=_analyze_h3)

    h3_plot_parser = subparsers.add_parser(
        "plot-h3", help="generate publication-style H3 decision figures"
    )
    h3_plot_parser.add_argument("--run", required=True, type=Path)
    h3_plot_parser.add_argument("--config", required=True, type=Path)
    h3_plot_parser.add_argument(
        "--output",
        type=Path,
        help="figure directory (default: RUN/analysis/h3/figures)",
    )
    h3_plot_parser.set_defaults(function=_plot_h3)

    horizon_plot_parser = subparsers.add_parser(
        "plot-extended-horizon",
        help="plot post-hoc H2/H3 coverage through the final MoE layer",
    )
    horizon_plot_parser.add_argument("--run", required=True, type=Path)
    horizon_plot_parser.add_argument("--config", required=True, type=Path)
    horizon_plot_parser.add_argument("--output", type=Path)
    horizon_plot_parser.set_defaults(function=_plot_extended_horizon)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
