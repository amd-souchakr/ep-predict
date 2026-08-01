from __future__ import annotations

import json

from ep_predict.analysis.admission import analyze_admission
from ep_predict.config import load_toml


if __name__ == "__main__":
    result = analyze_admission(
        load_toml("configs/experiment/h5-admission.toml")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
