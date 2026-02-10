# Code Audit And Sanity Check (SN_code)

This document is a referee-style audit focused on scientific/reproducibility pitfalls and “silent failure” risks. It also documents what the current code produces (tables/plots) so paper claims can be matched to artifacts.

## Workflow (High Level)

```mermaid
flowchart TD
  A[finaldata.csv] --> B[load_raw_csv + clean_loaded_data]
  B --> C[EDA reports + plots]
  B --> D[create/reuse strict holdout split]
  D --> E[train-only CSV (holdout hidden)]
  E --> F[Optuna tuning (CV RMSE objective)]
  F --> G[best_params.csv / table_best_params.csv]
  D --> H[Evaluation]
  G --> H
  H --> I[CV metrics (base vs tuned)]
  H --> J[Strict holdout metrics + holdout_predictions.csv]
  H --> K[LOCO group metrics + per-chain CSVs]
  H --> L[Journal-style PDF figures]
```

## Dataset Sanity (As Implemented)

The loader treats:
- `*` and empty-like tokens as missing (dropped for the chosen target only).
- `#` as a *flag* (the `#` character is stripped and the numeric value is retained).
- Target units are assumed keV and converted to MeV by dividing by `1000`.

Quick numbers from `snml.data.load_clean_data("finaldata.csv", ...)`:
- `sn`: `rows_after_clean=3440`, `rows_dropped_missing_target=118`, `target_raw_hash_count=1046`, `duplicate_rows_by_A_Z_N=0`
- `s2n`: `rows_after_clean=3321`, `rows_dropped_missing_target=237`, `target_raw_hash_count=1020`, `duplicate_rows_by_A_Z_N=0`

Per-run files that fully document this:
- `outputs/.../eda/.../cleaning_report.json`
- `outputs/.../eda/.../flags_report.csv`
- `outputs/.../eda/.../missingness_report.csv`
- `outputs/.../eda/.../summary_stats.csv`
- `outputs/.../tuning/.../data_report.json`
- `outputs/.../evaluation/.../data_report.json`

## Split Logic (Leakage Audit)

Goal: a strict holdout set must never influence hyperparameter tuning.

What the code does now:
- A strict holdout split is created once (or reused) under `outputs/splits/*_holdout.json`.
- Tuning never reads the strict holdout directly: the pipeline writes a `*_train.csv` and points Optuna/tuning at that file only.
- Evaluation uses the strict holdout split for:
  - reporting `split_summary.json` (train/test sizes + target stats)
  - strict holdout metrics + plots
  - restricting LOCO and CV metrics to *training-only* when a strict split is provided

Hardening added:
- Holdout split payload stores `data_sha256` to detect when a split file no longer matches the current CSV.
- Split payload may also include `train_keys/test_keys` (A/Z/N keys) for portability if the cleaned row order changes.

Remaining risk to call out in a paper:
- The strict holdout is random by rows (not by physics groups). That is fine for “random generalization” claims, but is not a substitute for out-of-distribution testing. LOCO (by `Z`) is the code’s OOD-style evaluation.

## Output Completeness (Paper ↔ Code Mapping)

The following artifacts are generated per configuration (`target_key` x `basic/extended`):

Tables:
- Table I (search space): `table_search_space.csv` (tuning)
- Table II (feature space): `table_feature_space.csv` (tuning)
- Table III (best hyperparameters): `table_best_params.csv` (tuning)
- CV results across folds (base vs tuned): `cv_fold_metrics.csv` + `cv_summary.json` (evaluation)

Plots (PDF, journal style):
- Fig. 1 (CV-RMSE convergence): `fig_cv_rmse_convergence.pdf` (tuning)
- Fig. 2 (Optuna history / best-so-far): `fig_optuna_history.pdf` (tuning)
- Fig. 3 (param importance): `fig_param_importance.pdf` + `param_importance.csv` (tuning)
- Fig. 4 (actual vs predicted): `fig_actual_vs_pred_*.pdf` (evaluation)
- Fig. 5 (residual analysis): `fig_residual*.pdf` (evaluation)
- Fig. 10 (feature importance): `fig_feature_importance.pdf` + `feature_importance.csv` (evaluation)
- Figs. 11–14 (corr heatmaps): `fig_corr_heatmap.pdf` + `correlation_matrix.csv` (evaluation)
- LOCO comparison (all models): `fig_loco_{rmse,mae,r2}_by_group.pdf` + `loco_group_metrics.csv` + `loco_group_predictions/chain_*.csv` (evaluation)

## Silent Failures And Their Mitigation

Places that used to fail silently or “half-fail” were hardened to leave evidence:
- EDA/reporting in the pipeline is non-blocking but now logs a warning and writes `eda_error.json` if it fails.
- Tuning plots no longer block the run. If importance can’t be computed (e.g., 1 trial), it writes `param_importance_error.json`.
- Evaluation plot generation is non-blocking but writes `plots_error.json` when it fails.
- Matplotlib is configured for headless servers (`Agg`) and writes its cache under the outputs directory (avoids `$HOME/.matplotlib` permission failures).

Recommendation for paper/workflow:
- Treat the presence of any `*_error.json` file as “run incomplete” and report it.

## Major Scientific/Methodological Pitfalls (Retraction Risk)

These are the kinds of issues referees look for. The code is now *mostly safe*, but your paper should explicitly address each item.

1. Data leakage and split reuse
- Fixed: strict holdout is created once and kept out of Optuna by tuning on a generated train-only CSV.
- Remaining: if you manually run `run_tuning()` against the full CSV (not via the pipeline), you can accidentally tune on what you later call “test”. Use the pipeline for the paper runs.

2. Optuna study contamination (mixing trials across runs)
- Mitigated: Ubuntu CLI uses a unique `study_name` derived from `run_name`.
- Remaining: if you reuse the same `run_name` and the same `--storage`, Optuna will resume the old study (intended for resume, risky for fresh experiments). For paper runs, use a unique `--run-name` per experiment or delete the study from the DB.

3. SQLite + parallel/distributed tuning
- Current behavior: `n_jobs` parallelizes trials in one process (threads). This is usually OK with SQLite, but multi-process or multi-node parallelism can lock the DB.
- For genuinely distributed runs, use Postgres-backed Optuna storage and run multiple workers. SQLite is fine for single-node quick work.

4. Treatment of `#` flagged values
- Current behavior: `#` is stripped and the numeric value is retained (so flagged values influence training/evaluation).
- Paper risk: if `#` indicates extrapolated/less-trustworthy measurements, reviewers may ask for a sensitivity analysis:
  - “trained/evaluated excluding `#` values”
  - or report performance separately on flagged vs unflagged subsets

5. Target unit assumption (keV → MeV)
- The code assumes keV in CSV and converts to MeV.
- Paper risk: if the dataset is already MeV, the model would be off by ×1000 and still “fit” numerically. The code now emits target MeV min/max/mean/std in `cleaning_report.json`; confirm it is physically plausible.

## Reproducibility Checklist (Paper-Ready)

The code now saves `environment.json` for EDA/tuning/evaluation runs. For the paper, also report:
- The exact `outputs/splits/*_holdout.json` used (or its SHA).
- `seed`, `test_size`, `cv_folds`, and whether LOCO was restricted to the training split.
- The Optuna storage type (SQLite vs Postgres), number of trials, and the exact search space (Table I).
- Versions of `xgboost`, `optuna`, `numpy`, `pandas`, and `scikit-learn` (from `environment.json`).

Notes on metrics files:
- `tuning/holdout_metrics.json` is explicitly labeled `split_type=internal_random` when tuning runs on a generated train-only CSV (pipeline default). Do not confuse this with the strict holdout metrics in `evaluation/holdout_metrics.json`.
