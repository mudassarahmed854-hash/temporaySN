from __future__ import annotations

from typing import Dict


def build_xgb_params(trial_params: Dict, device: str, seed: int, deterministic: bool = False) -> Dict:
    params = dict(trial_params)
    params.update({
        "objective": "reg:squarederror",
        "verbosity": 0,
        "seed": seed,
    })

    if device != "cpu":
        dev = device
        if device == "gpu":
            dev = "cuda"
        params.update({
            "device": dev,
            "tree_method": "hist",
        })
        if deterministic:
            params["deterministic_histogram"] = 1
    else:
        params.update({
            "tree_method": "hist",
            "nthread": 1 if deterministic else -1,
        })

    return params
