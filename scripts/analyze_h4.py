from __future__ import annotations

import argparse
import json
from pathlib import Path

from ep_predict.analysis.h4 import analyze_h4
from ep_predict.config import load_toml


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the H4 oracle.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = analyze_h4(args.run, load_toml(args.config))
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

