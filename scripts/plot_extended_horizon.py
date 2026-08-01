from __future__ import annotations

import argparse
import json
from pathlib import Path

from ep_predict.config import load_toml
from ep_predict.visualize.extended_horizon import plot_extended_horizon


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot post-hoc H2/H3 coverage through the final layer."
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = plot_extended_horizon(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
