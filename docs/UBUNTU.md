# Ubuntu Server Notes

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn xgboost optuna matplotlib seaborn
```

## Run (basic examples)

```bash
python scripts/run_pipeline_ubuntu.py --target sn --n-trials 1000 --n-jobs 10
python scripts/run_pipeline_ubuntu.py --target s2n --n-trials 1000 --n-jobs 10
python scripts/run_pipeline_ubuntu.py --target sn --use-extended --n-trials 1000 --n-jobs 10

# All 4 paper configurations (sn/s2n x basic/extended)
python scripts/run_pipeline_ubuntu.py --all-targets --all-feature-sets --n-trials 1000 --n-jobs 10 --run-name paper_run
```

## Outputs

```
outputs/{target}/{feature_set}/{stage}/{run_id}/
outputs/splits/
outputs/progress_logs/
```

To redirect outputs:

```bash
export SNML_OUTPUTS_DIR=outputs_paper
python scripts/run_pipeline_ubuntu.py --all-targets --all-feature-sets --n-trials 1000 --n-jobs 10 --run-name paper_run
```
