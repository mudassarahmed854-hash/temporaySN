from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List

# Target mapping
TARGETS: Dict[str, str] = {
    "sn": "S(n)",
    "s2n": "S(2n)",
}

# Feature set registry (update here to add new features)
FEATURE_SETS: Dict[str, List[str]] = {
    "basic": ["A", "Z", "N"],
    "extended": ["A", "Z", "N", "Shell_Closure", "N_Z_excess", "N_to_Z_ratio"],
}

MAGIC_NUMBERS = {2, 8, 20, 28, 50, 82, 126}

# Default base (untuned) parameters for evaluation
BASE_XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "objective": "reg:squarederror",
    "random_state": 42,
}

# Output base
OUTPUTS_DIR = os.getenv("SNML_OUTPUTS_DIR", "outputs")

@dataclass
class TuningConfig:
    data_path: str = "finaldata.csv"
    target_key: str = "sn"
    feature_set: str = "basic"
    use_extended: bool = False
    n_trials: int = 15
    cv_folds: int = 10
    seed: int = 42
    patience: int = 1000
    min_delta: float = 1e-4
    device: str = "cpu"  # cpu | gpu
    study_name: str = "xgb_sn_cv"
    run_name: str | None = None
    deterministic: bool = False
    holdout_split_path: str | None = None
    holdout_test_size: float = 0.2
    storage: str | None = None
    n_jobs: int = 1
    gpu_count: int = 1

@dataclass
class EvalConfig:
    data_path: str = "finaldata.csv"
    target_key: str = "sn"
    feature_set: str = "basic"
    use_extended: bool = False
    seed: int = 42
    cv_folds: int = 10
    device: str = "cpu"
    tuned_params_path: str | None = None
    run_name: str | None = None
    make_plots: bool = True
    deterministic: bool = False
    run_loco: bool = False
    loco_group_col: str = "Z"  # isotopic chain by proton number
    loco_max_groups: int | None = None
    loco_min_group_size: int = 5
    loco_include_base: bool = False
    holdout_split_path: str | None = None
    holdout_test_size: float = 0.2
