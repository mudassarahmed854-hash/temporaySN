# Usage (Ubuntu Server)

## Quick Start

### 1) EDA

```python
from snml.eda import run_eda

run_eda("finaldata.csv", "sn")
run_eda("finaldata.csv", "s2n")
```

Outputs are saved under `outputs/eda/` by default, or under
`outputs/{target}/{feature_set}/eda/{run_id}/` when run via the pipeline.

### 2) Tuning (10-20 trials recommended for quick runs)

```python
from pathlib import Path

from snml.config import TuningConfig, OUTPUTS_DIR
from snml.tuning import run_tuning

cfg = TuningConfig(
    data_path="finaldata.csv",
    target_key="sn",
    use_extended=False,
    n_trials=15,
    device="cpu",
    deterministic=True,
    run_name="tune_sn_basic",
    storage=f"sqlite:///{(Path(OUTPUTS_DIR) / 'optuna.db').as_posix()}",
    n_jobs=10,
)
run_dir = run_tuning(cfg)
print(run_dir)
```

Tuning outputs:
- `table_search_space.csv` (Table I)
- `table_feature_space.csv` (Table II)
- `table_best_params.csv` (Table III)
- `best_params.csv` (preferred for downstream evaluation)
- `best_params.json`
- `trials.csv`
- `holdout_metrics.json`
- `fig_cv_rmse_convergence.pdf` (Fig. 1, per configuration)
- `fig_optuna_history.pdf` (Fig. 2)
- `fig_param_importance.pdf` (Fig. 3)

### 3) Evaluation + LOCO

```python
from snml.config import EvalConfig
from snml.evaluation import run_evaluation

cfg = EvalConfig(
    data_path="finaldata.csv",
    target_key="sn",
    use_extended=False,
    device="cpu",
    deterministic=True,
    run_loco=True,
    loco_group_col="Z",
    loco_max_groups=10,
    loco_min_group_size=5,
    loco_include_base=True,
)
run_dir = run_evaluation(cfg)
print(run_dir)
```

Evaluation outputs:
- `cv_fold_metrics.csv` (base vs tuned XGBoost; 10-fold)
- `cv_summary.json`
- `holdout_metrics.json` (includes LR/RF on the strict holdout)
- `holdout_predictions.csv` (includes base/tuned/LR/RF)
- `feature_importance.csv`
- `correlation_matrix.csv`
- `loco_group_metrics.csv` (tuned/base/LR/RF; if enabled)
- `loco_group_predictions/chain_*.csv` (actual/predicted per chain; if enabled)
- `split_summary.json` (train/test counts + target stats)
- `data_report.json` (flag counts + rows dropped due to missing target)

Figures (saved as PDF under the evaluation run directory):
- Scatter plots (Fig. 4 equivalents): `fig_actual_vs_pred_*.pdf`
- Residual analysis (Fig. 5 equivalents): `fig_residual*.pdf`
- Feature importance (Fig. 10): `fig_feature_importance.pdf`
- Correlation heatmap (Figs. 11-14 equivalents): `fig_corr_heatmap.pdf`
- LOCO comparison plots: `fig_loco_{rmse,mae,r2}_by_group.pdf`

### 4) One-shot pipeline (tuning -> evaluation waterfall)

```bash
python scripts/run_pipeline.py
```

### 5) Ubuntu CLI (recommended)

```bash
python scripts/run_pipeline_ubuntu.py --target sn --n-trials 1000 --n-jobs 10
python scripts/run_pipeline_ubuntu.py --target s2n --n-trials 1000 --n-jobs 10
python scripts/run_pipeline_ubuntu.py --target sn --use-extended --n-trials 1000 --n-jobs 10

# Run all 4 paper configurations (sn/s2n x basic/extended)
python scripts/run_pipeline_ubuntu.py --all-targets --all-feature-sets --n-trials 1000 --n-jobs 10 --run-name paper_run

# LOCO controls
python scripts/run_pipeline_ubuntu.py --target sn --n-trials 50 --n-jobs 10 --run-name smoke --loco-max-groups 0
python scripts/run_pipeline_ubuntu.py --target sn --n-trials 50 --n-jobs 10 --run-name smoke --no-loco
```

Notes:
- `--loco-max-groups 0` means "run LOCO for all eligible groups".
- Tuning uses a strict holdout split that is hidden from Optuna by training on a generated `train.csv` only.

## Feature Sets

Edit `snml/config.py`:
- `FEATURE_SETS` defines each feature set.
- Add derived features in `snml/features.py`.

## Reproducibility

For deterministic CPU runs:
- Set `deterministic=True` in configs (CLI does this by default).
- This forces single-threaded XGBoost and seeds RNGs.

## Outputs

```
outputs/{target}/{feature_set}/{stage}/{run_id}/
```

Additional folders:
- `outputs/splits/` (strict holdout splits + generated train.csv)
- `outputs/progress_logs/` (trial progress + ETA)

Stages: `tuning`, `evaluation`, `eda`.

To redirect outputs, set `SNML_OUTPUTS_DIR` before running, e.g.:

```bash
export SNML_OUTPUTS_DIR=outputs_paper
python scripts/run_pipeline_ubuntu.py --all-targets --all-feature-sets --n-trials 1000 --n-jobs 10 --run-name paper_run
```
