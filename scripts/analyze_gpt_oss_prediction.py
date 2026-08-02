#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from ep_predict.analysis.gpt_oss_prediction import analyze_gpt_oss_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=Path, default=Path("artifacts/runs/gpt-oss-20b-milestone-e")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/gpt-oss-20b-milestone-e.toml"),
    )
    args = parser.parse_args()
    result = analyze_gpt_oss_prediction(
        args.run, tomllib.loads(args.config.read_text(encoding="utf-8"))
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
