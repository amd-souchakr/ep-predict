from __future__ import annotations

import json

from ep_predict.analysis.h5 import analyze_h5
from ep_predict.config import load_toml


if __name__ == "__main__":
    result = analyze_h5(load_toml("configs/experiment/h5-first-order.toml"))
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
