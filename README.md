# SN_code (Ubuntu)

Clean, reproducible workflow for predicting neutron separation energies using XGBoost on Ubuntu servers.

## Data

Use `finaldata.csv` as the input dataset. The loader cleans `*` and `#` flags and converts keV to MeV.

## Feature Sets

Feature definitions live in `snml/config.py` and `snml/features.py`:

- `basic`: `A, Z, N`
- `extended`: `A, Z, N, Shell_Closure, N_Z_excess, N_to_Z_ratio`

To add features, update `FEATURE_SETS` in `snml/config.py` and (if derived) implement them in `snml/features.py`.

## Outputs

```
outputs/{target}/{feature_set}/{stage}/{run_id}/
outputs/splits/
outputs/progress_logs/
```

Stages are `tuning`, `evaluation`, and `eda`.

To place all outputs under a different folder, set `SNML_OUTPUTS_DIR` before running, e.g.:

```bash
export SNML_OUTPUTS_DIR=outputs_paper
```

Each run also writes `environment.json` (Python + package versions) inside the run directory for reproducibility.

## Documentation

- `docs/USAGE.md`
- `docs/UBUNTU.md`
- `scripts/run_pipeline_ubuntu.py`

## Notes

- `finaldata.csv` contains `*` and `#` flags; the loader cleans these to NaN and strips `#`.
- The default base (untuned) XGBoost parameters are in `snml/config.py`.
