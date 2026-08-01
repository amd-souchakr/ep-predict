from __future__ import annotations

import argparse
import json
from pathlib import Path

from ep_predict.analysis.codesign import analyze_codesign_map
from ep_predict.config import load_toml


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the H4 co-design map.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = analyze_codesign_map(load_toml(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

