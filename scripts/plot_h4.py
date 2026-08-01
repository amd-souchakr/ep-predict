from __future__ import annotations

import argparse
import json
from pathlib import Path

from ep_predict.config import load_toml
from ep_predict.visualize.h4 import plot_h4


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot H4 oracle feasibility.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = plot_h4(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

