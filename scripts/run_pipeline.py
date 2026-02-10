from __future__ import annotations

from pathlib import Path

from snml.config import TuningConfig, EvalConfig, OUTPUTS_DIR
from snml.pipeline import run_pipeline

# Edit these defaults as needed
TUNE_CFG = TuningConfig(
    data_path="finaldata.csv",
    target_key="sn",
    use_extended=False,
    n_trials=10,
    device="cpu",
    deterministic=True,
    run_name="pipeline_run",
    study_name="xgb_pipeline_run",
    storage=f"sqlite:///{(Path(OUTPUTS_DIR) / 'optuna.db').as_posix()}",
    n_jobs=10,
)

EVAL_CFG = EvalConfig(
    data_path="finaldata.csv",
    target_key="sn",
    use_extended=False,
    device="cpu",
    deterministic=True,
    run_name="pipeline_run_eval",
    run_loco=True,
    loco_group_col="Z",
    loco_max_groups=5,
    loco_min_group_size=5,
    loco_include_base=False,
)

if __name__ == "__main__":
    out = run_pipeline(TUNE_CFG, EVAL_CFG)
    print(out)
