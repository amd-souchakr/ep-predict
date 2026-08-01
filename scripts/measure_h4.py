from __future__ import annotations

import argparse
import json
from pathlib import Path

from ep_predict.config import load_toml
from ep_predict.hardware.h4 import measure_h4


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure hook-free H4 timing.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    args = parser.parse_args()
    result = measure_h4(
        load_toml(args.model_config),
        load_toml(args.experiment_config),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

