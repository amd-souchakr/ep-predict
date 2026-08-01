from __future__ import annotations

import json

from ep_predict.config import load_toml
from ep_predict.visualize.h5 import plot_h5


if __name__ == "__main__":
    result = plot_h5(load_toml("configs/experiment/h5-first-order.toml"))
    print(json.dumps(result, indent=2, sort_keys=True))
