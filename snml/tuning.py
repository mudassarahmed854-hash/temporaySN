from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import os
import random
import time
import threading
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold

from .config import OUTPUTS_DIR, TuningConfig, TARGETS, FEATURE_SETS
from .features import get_feature_list
from .data import load_clean_data
from .metrics import regression_metrics
from .io_utils import make_run_dir, save_json, save_csv, collect_environment
from .model import build_xgb_params
from .splits import load_holdout_split, resolve_split_indices
from .style import COLORS, set_plot_style


class PlateauStopper:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if study.best_value is None:
            return
        current = study.best_value
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                study.stop()


def _suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 400, 3000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _search_space_table() -> pd.DataFrame:
    rows = [
        {"param": "n_estimators", "type": "int", "min": 400, "max": 3000, "scale": "linear"},
        {"param": "max_depth", "type": "int", "min": 3, "max": 12, "scale": "linear"},
        {"param": "learning_rate", "type": "float", "min": 1e-3, "max": 0.3, "scale": "log"},
        {"param": "subsample", "type": "float", "min": 0.5, "max": 1.0, "scale": "linear"},
        {"param": "colsample_bytree", "type": "float", "min": 0.4, "max": 1.0, "scale": "linear"},
        {"param": "min_child_weight", "type": "float", "min": 0.0, "max": 10.0, "scale": "linear"},
        {"param": "gamma", "type": "float", "min": 0.0, "max": 10.0, "scale": "linear"},
        {"param": "reg_lambda", "type": "float", "min": 1e-3, "max": 10.0, "scale": "log"},
        {"param": "reg_alpha", "type": "float", "min": 1e-3, "max": 10.0, "scale": "log"},
    ]
    return pd.DataFrame(rows)


def run_tuning(cfg: TuningConfig) -> str:
    if cfg.target_key not in TARGETS:
        raise ValueError(f"Unknown target_key: {cfg.target_key}")

    feature_set = "extended" if cfg.use_extended else cfg.feature_set
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    features = get_feature_list(feature_set)

    df, data_report = load_clean_data(cfg.data_path, cfg.target_key)

    target_col = TARGETS[cfg.target_key]
    y = df[f"{target_col}_MeV"].values.astype("float32")
    X = df[features].values.astype("float32")

    if cfg.holdout_split_path:
        split = load_holdout_split(cfg.holdout_split_path)
        tr_idx, te_idx = resolve_split_indices(df, split)
        X_dev, y_dev = X[tr_idx], y[tr_idx]
        X_test, y_test = X[te_idx], y[te_idx]
    else:
        # Outer split for hold-out evaluation
        X_dev, X_test, y_dev, y_test = train_test_split(
            X,
            y,
            test_size=cfg.holdout_test_size,
            random_state=cfg.seed,
        )

    run_dir = make_run_dir(OUTPUTS_DIR, cfg.target_key, feature_set, "tuning", cfg.run_name)
    save_json(run_dir / "environment.json", collect_environment())
    # Data provenance / cleaning stats (asterisks, hashes, rows dropped, etc.)
    save_json(run_dir / "data_report.json", data_report)
    # Table I: Hyperparameter search space
    save_csv(run_dir / "table_search_space.csv", _search_space_table())
    # Table II: Feature space used
    fs_rows = [{"feature_set": k, "features": ", ".join(v)} for k, v in FEATURE_SETS.items()]
    save_csv(run_dir / "table_feature_space.csv", pd.DataFrame(fs_rows))
    save_json(run_dir / "run_config.json", {
        "target_key": cfg.target_key,
        "feature_set": feature_set,
        "features": features,
        "n_trials": cfg.n_trials,
        "cv_folds": cfg.cv_folds,
        "seed": cfg.seed,
        "device": cfg.device,
        "deterministic": cfg.deterministic,
        "holdout_split_path": cfg.holdout_split_path,
        "holdout_test_size": cfg.holdout_test_size,
        "holdout_is_internal_random_split": bool(cfg.holdout_split_path is None),
        "n_total_clean": int(len(df)),
        "n_dev": int(len(X_dev)),
        "n_holdout": int(len(X_test)),
    })

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        n_estimators = params.pop("n_estimators")
        xgb_params = build_xgb_params(
            params,
            device=cfg.device,
            seed=cfg.seed,
            deterministic=cfg.deterministic,
        )
        if cfg.device != "cpu" and cfg.gpu_count > 1:
            xgb_params["device"] = f"cuda:{trial.number % cfg.gpu_count}"

        kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
        rmses = []
        maes = []

        for tr_idx, va_idx in kf.split(X_dev):
            dtr = xgb.DMatrix(X_dev[tr_idx], label=y_dev[tr_idx])
            dva = xgb.DMatrix(X_dev[va_idx], label=y_dev[va_idx])

            booster = xgb.train(xgb_params, dtr, num_boost_round=n_estimators)
            preds = booster.predict(dva)

            rmse = float(np.sqrt(np.mean((y_dev[va_idx] - preds) ** 2)))
            mae = float(np.mean(np.abs(y_dev[va_idx] - preds)))
            rmses.append(rmse)
            maes.append(mae)

        trial.set_user_attr("cv_rmse_std", float(np.std(rmses)))
        trial.set_user_attr("cv_mae_mean", float(np.mean(maes)))
        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    storage = cfg.storage or os.getenv("OPTUNA_STORAGE") or f"sqlite:///{(run_dir / 'optuna.db').as_posix()}"
    study = optuna.create_study(
        study_name=cfg.study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    existing_trials = len(study.trials)
    if existing_trials:
        print(f"[INFO] Loaded existing Optuna study '{cfg.study_name}' with {existing_trials} trials from {storage}")

    start_ts = time.time()
    progress_dir = Path(OUTPUTS_DIR) / "progress_logs"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"{run_dir.name}_progress.log"
    progress_lock = threading.Lock()

    def _log_progress(study: optuna.Study, trial: optuna.Trial) -> None:
        completed = max(0, len(study.trials) - existing_trials)
        total = cfg.n_trials
        elapsed = time.time() - start_ts
        if completed > 0:
            avg = elapsed / completed
            remaining = max(0.0, avg * (total - completed))
        else:
            remaining = 0.0
        pct = min(100.0, (completed / total) * 100.0)
        line = (
            f"trial={completed}/{total} "
            f"pct={pct:0.2f}% "
            f"elapsed_s={elapsed:0.1f} "
            f"eta_s={remaining:0.1f} "
            f"best={study.best_value}\n"
        )
        with progress_lock:
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(line)
        print(line, end="")

    stopper = PlateauStopper(cfg.patience, cfg.min_delta)
    study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs, callbacks=[stopper, _log_progress])

    # Save trials
    trials_df = study.trials_dataframe(attrs=("number", "value", "user_attrs", "params"))
    save_csv(run_dir / "trials.csv", trials_df)

    # Tuning plots (paper-ready artifacts)
    try:
        import matplotlib.pyplot as plt
        set_plot_style(OUTPUTS_DIR)

        hist_df = trials_df[["number", "value"]].sort_values("number").copy()
        hist_df["best_so_far"] = hist_df["value"].cummin()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(hist_df["number"], hist_df["value"], s=14, alpha=0.6, label="Trial value", color=COLORS["tuned"])
        ax.plot(hist_df["number"], hist_df["best_so_far"], color=COLORS["accent"], linewidth=2.0, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("CV RMSE")
        ax.set_title("Optuna Optimization History")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "fig_optuna_history.pdf")
        plt.close(fig)

        # Fig. 1: CV-RMSE convergence curve (best-so-far over trials)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(hist_df["number"], hist_df["best_so_far"], color=COLORS["accent"], linewidth=2.0)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best CV RMSE")
        ax.set_title("CV-RMSE Convergence")
        fig.tight_layout()
        fig.savefig(run_dir / "fig_cv_rmse_convergence.pdf")
        plt.close(fig)
    except Exception as e:
        # Plotting should not fail the training run.
        save_json(run_dir / "tuning_plots_error.json", {"error": str(e)})
        print(f"[WARN] Tuning history/convergence plots skipped: {e}")

    # Parameter importance (CSV + figure). This can fail for very small studies.
    try:
        importances = optuna.importance.get_param_importances(study)
        imp_df = pd.DataFrame({
            "param": list(importances.keys()),
            "importance": list(importances.values()),
        }).sort_values("importance", ascending=False)
        save_csv(run_dir / "param_importance.csv", imp_df)

        try:
            import matplotlib.pyplot as plt
            set_plot_style(OUTPUTS_DIR)

            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(imp_df["param"][::-1], imp_df["importance"][::-1], color=COLORS["tuned"])
            ax.set_xlabel("Importance")
            ax.set_title("Optuna Parameter Importance")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_param_importance.pdf")
            plt.close(fig)
        except Exception as e:
            save_json(run_dir / "param_importance_plot_error.json", {"error": str(e)})
            print(f"[WARN] Param importance plot skipped: {e}")
    except Exception as e:
        save_json(run_dir / "param_importance_error.json", {"error": str(e)})
        print(f"[WARN] Param importance skipped: {e}")

    # Best params
    best_params = study.best_trial.params.copy()
    save_json(run_dir / "best_params.json", best_params)
    # CSV version for downstream evaluation
    best_df = pd.DataFrame(
        {"param": list(best_params.keys()), "value": list(best_params.values())}
    )
    save_csv(run_dir / "best_params.csv", best_df)
    # Table III: Optimal hyperparameters (per run)
    save_csv(run_dir / "table_best_params.csv", best_df)

    # Final model on full dev, evaluate on holdout
    final_rounds = best_params.pop("n_estimators")
    final_params = build_xgb_params(
        best_params,
        device=cfg.device,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )

    dev_dm = xgb.DMatrix(X_dev, label=y_dev)
    test_dm = xgb.DMatrix(X_test, label=y_test)

    final_model = xgb.train(final_params, dev_dm, num_boost_round=final_rounds)
    preds_test = final_model.predict(test_dm)
    holdout_metrics = regression_metrics(y_test, preds_test)

    # Label this explicitly: in the pipeline, tuning runs on train-only CSV, so this is
    # an internal split (not the strict holdout used in the evaluation stage).
    save_json(run_dir / "holdout_metrics.json", {
        "split_type": "precomputed" if cfg.holdout_split_path else "internal_random",
        "n_dev": int(len(X_dev)),
        "n_holdout": int(len(X_test)),
        **holdout_metrics,
    })
    save_json(run_dir / "study_summary.json", {
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "target_key": cfg.target_key,
        "feature_set": feature_set,
        "device": cfg.device,
        "storage": storage,
    })

    return str(run_dir)
