#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from ep_predict.analysis.gpt_oss_multihead import run_confirmation_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiment/gpt-oss-20b-mtp-head-confirmation.toml"
        ),
    )
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    result = run_confirmation_analysis(config, args.config)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
