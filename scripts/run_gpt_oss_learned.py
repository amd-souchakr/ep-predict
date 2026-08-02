#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from ep_predict.analysis.gpt_oss_learned import run_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/gpt-oss-20b-milestone-f.toml"),
    )
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    result = run_development(config, args.config)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
