from __future__ import annotations

import json

from ep_predict.config import load_toml
from ep_predict.visualize.admission import plot_admission


if __name__ == "__main__":
    result = plot_admission(
        load_toml("configs/experiment/h5-admission.toml")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
