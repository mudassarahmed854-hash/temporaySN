from __future__ import annotations

from typing import Dict, Any
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

from .config import OUTPUTS_DIR, EvalConfig, TARGETS, BASE_XGB_PARAMS
from .features import get_feature_list
from .data import load_clean_data
from .metrics import regression_metrics
from .io_utils import make_run_dir, save_json, save_csv, find_latest_run, collect_environment
from .plots import plot_from_run
from .splits import load_holdout_split, resolve_split_indices


def _normalize_tuned_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if "alpha" in out and "reg_alpha" not in out:
        out["reg_alpha"] = out.pop("alpha")
    if "reg_gamma" in out and "gamma" not in out:
        out["gamma"] = out.pop("reg_gamma")
    for k in ("n_estimators", "max_depth"):
        if k in out:
            out[k] = int(out[k])
    return out


def _load_params(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if not {"param", "value"}.issubset(df.columns):
            raise ValueError("CSV must contain columns: param, value")
        params = {}
        for k, v in zip(df["param"], df["value"]):
            try:
                v_num = float(v)
                params[k] = v_num
            except (TypeError, ValueError):
                params[k] = v
    else:
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
    return _normalize_tuned_params(params)


def _xgb_regressor_params(
    params: Dict[str, Any],
    device: str,
    deterministic: bool,
    seed: int,
) -> Dict[str, Any]:
    out = dict(params)
    # Keep evaluation consistent across XGBoost versions.
    out.setdefault("objective", "reg:squarederror")
    out.setdefault("verbosity", 0)
    if "random_state" not in out and "seed" not in out:
        out["random_state"] = int(seed)
    if device != "cpu":
        dev = device
        if device == "gpu":
            dev = "cuda"
        out.update({"device": dev, "tree_method": "hist"})
        if deterministic:
            out["deterministic_histogram"] = 1
    else:
        out.update({"tree_method": "hist", "n_jobs": 1 if deterministic else -1})
    return out


def _make_lr_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])


def _make_rf_model(seed: int, deterministic: bool) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            random_state=int(seed),
            n_jobs=1 if deterministic else -1,
        )),
    ])


def _leave_one_group_out(
    X: np.ndarray,
    y: np.ndarray,
    a_vals: np.ndarray,
    z_vals: np.ndarray,
    n_vals: np.ndarray,
    groups: np.ndarray,
    tuned_params: Dict[str, Any],
    device: str,
    deterministic: bool,
    seed: int,
    base_params: Dict[str, Any],
    include_base: bool,
    min_group_size: int,
    max_groups: int | None,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # Order groups by size (desc)
    unique, counts = np.unique(groups, return_counts=True)
    unique = unique[np.argsort(-counts)]

    if max_groups is not None:
        unique = unique[:max_groups]

    rows = []
    per_group: Dict[str, pd.DataFrame] = {}
    for g in unique:
        mask = groups == g
        n_test = int(mask.sum())
        if n_test < min_group_size:
            continue

        X_tr, y_tr = X[~mask], y[~mask]
        X_te, y_te = X[mask], y[mask]

        tuned_model = xgb.XGBRegressor(
            **_xgb_regressor_params(tuned_params, device, deterministic, seed)
        )
        tuned_model.fit(X_tr, y_tr)
        pred_tuned = tuned_model.predict(X_te)
        m_tuned = regression_metrics(y_te, pred_tuned)

        # Baselines (default settings, imputed for missing values)
        lr_model = _make_lr_model()
        rf_model = _make_rf_model(seed, deterministic)
        lr_model.fit(X_tr, y_tr)
        rf_model.fit(X_tr, y_tr)
        pred_lr = lr_model.predict(X_te)
        pred_rf = rf_model.predict(X_te)
        m_lr = regression_metrics(y_te, pred_lr)
        m_rf = regression_metrics(y_te, pred_rf)

        row: Dict[str, Any] = {
            "group": g,
            "n_test": n_test,
            "tuned_rmse": m_tuned["rmse"],
            "tuned_mae": m_tuned["mae"],
            "tuned_r2": m_tuned["r2"],
            "lr_rmse": m_lr["rmse"],
            "lr_mae": m_lr["mae"],
            "lr_r2": m_lr["r2"],
            "rf_rmse": m_rf["rmse"],
            "rf_mae": m_rf["mae"],
            "rf_r2": m_rf["r2"],
        }

        pred_df = pd.DataFrame({
            "group": g,
            "A": a_vals[mask],
            "Z": z_vals[mask],
            "N": n_vals[mask],
            "y_true": y_te,
            "y_pred_tuned": pred_tuned,
            "y_pred_lr": pred_lr,
            "y_pred_rf": pred_rf,
            "resid_tuned": y_te - pred_tuned,
            "resid_lr": y_te - pred_lr,
            "resid_rf": y_te - pred_rf,
        })

        if include_base:
            base_model = xgb.XGBRegressor(
                **_xgb_regressor_params(base_params, device, deterministic, seed)
            )
            base_model.fit(X_tr, y_tr)
            pred_base = base_model.predict(X_te)
            m_base = regression_metrics(y_te, pred_base)
            row.update({
                "base_rmse": m_base["rmse"],
                "base_mae": m_base["mae"],
                "base_r2": m_base["r2"],
            })
            pred_df["y_pred_base"] = pred_base
            pred_df["resid_base"] = y_te - pred_base

        rows.append(row)
        per_group[str(g)] = pred_df

    return pd.DataFrame(rows), per_group


def run_evaluation(cfg: EvalConfig) -> str:
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

    # Find tuned params if not provided
    tuned_params_path = cfg.tuned_params_path
    if tuned_params_path is None:
        latest_run = find_latest_run(OUTPUTS_DIR, cfg.target_key, feature_set, "tuning")
        if latest_run is None:
            raise FileNotFoundError("No tuning run found. Provide tuned_params_path.")
        # Prefer CSV if present
        csv_path = latest_run / "best_params.csv"
        tuned_params_path = str(csv_path if csv_path.exists() else latest_run / "best_params.json")

    tuned_params = _load_params(tuned_params_path)

    run_dir = make_run_dir(OUTPUTS_DIR, cfg.target_key, feature_set, "evaluation", cfg.run_name)
    save_json(run_dir / "environment.json", collect_environment())
    # Data provenance / cleaning stats (asterisks, hashes, rows dropped, etc.)
    save_json(run_dir / "data_report.json", data_report)
    save_json(run_dir / "tuned_params.json", tuned_params)
    # CSV version for easy inspection
    pd.DataFrame(
        {"param": list(tuned_params.keys()), "value": list(tuned_params.values())}
    ).to_csv(run_dir / "tuned_params.csv", index=False)

    # Cross-validation (base vs tuned)
    cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    rows = []
    cv_X = X
    cv_y = y
    if cfg.holdout_split_path:
        split = load_holdout_split(cfg.holdout_split_path)
        tr_idx, _ = resolve_split_indices(df, split)
        cv_X = X[tr_idx]
        cv_y = y[tr_idx]

    for fold, (tr_idx, va_idx) in enumerate(cv.split(cv_X), start=1):
        X_tr, X_va = cv_X[tr_idx], cv_X[va_idx]
        y_tr, y_va = cv_y[tr_idx], cv_y[va_idx]

        base_model = xgb.XGBRegressor(
            **_xgb_regressor_params(BASE_XGB_PARAMS, cfg.device, cfg.deterministic, cfg.seed)
        )
        base_model.fit(X_tr, y_tr)
        pred_base = base_model.predict(X_va)

        tuned_model = xgb.XGBRegressor(
            **_xgb_regressor_params(tuned_params, cfg.device, cfg.deterministic, cfg.seed)
        )
        tuned_model.fit(X_tr, y_tr)
        pred_tuned = tuned_model.predict(X_va)

        m_base = regression_metrics(y_va, pred_base)
        m_tuned = regression_metrics(y_va, pred_tuned)

        rows.append({
            "fold": fold,
            "base_rmse": m_base["rmse"],
            "base_mae": m_base["mae"],
            "base_r2": m_base["r2"],
            "tuned_rmse": m_tuned["rmse"],
            "tuned_mae": m_tuned["mae"],
            "tuned_r2": m_tuned["r2"],
        })

    cv_df = pd.DataFrame(rows)
    save_csv(run_dir / "cv_fold_metrics.csv", cv_df)

    cv_summary = {
        "base_rmse_mean": float(cv_df["base_rmse"].mean()),
        "base_rmse_std": float(cv_df["base_rmse"].std(ddof=1)),
        "base_mae_mean": float(cv_df["base_mae"].mean()),
        "base_mae_std": float(cv_df["base_mae"].std(ddof=1)),
        "base_r2_mean": float(cv_df["base_r2"].mean()),
        "base_r2_std": float(cv_df["base_r2"].std(ddof=1)),
        "tuned_rmse_mean": float(cv_df["tuned_rmse"].mean()),
        "tuned_rmse_std": float(cv_df["tuned_rmse"].std(ddof=1)),
        "tuned_mae_mean": float(cv_df["tuned_mae"].mean()),
        "tuned_mae_std": float(cv_df["tuned_mae"].std(ddof=1)),
        "tuned_r2_mean": float(cv_df["tuned_r2"].mean()),
        "tuned_r2_std": float(cv_df["tuned_r2"].std(ddof=1)),
    }
    save_json(run_dir / "cv_summary.json", cv_summary)

    # Hold-out split
    if cfg.holdout_split_path:
        split = load_holdout_split(cfg.holdout_split_path)
        tr_idx, te_idx = resolve_split_indices(df, split)
        X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=cfg.holdout_test_size,
            random_state=cfg.seed,
        )

    split_summary = {
        "split_type": "precomputed" if cfg.holdout_split_path else "random",
        "holdout_split_path": cfg.holdout_split_path,
        "seed": int(cfg.seed),
        "test_size": float(cfg.holdout_test_size),
        "n_total_clean": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "y_train_mean": float(np.mean(y_tr)),
        "y_train_std": float(np.std(y_tr, ddof=1)) if len(y_tr) > 1 else 0.0,
        "y_test_mean": float(np.mean(y_te)),
        "y_test_std": float(np.std(y_te, ddof=1)) if len(y_te) > 1 else 0.0,
    }
    save_json(run_dir / "split_summary.json", split_summary)

    base_model = xgb.XGBRegressor(
        **_xgb_regressor_params(BASE_XGB_PARAMS, cfg.device, cfg.deterministic, cfg.seed)
    )
    tuned_model = xgb.XGBRegressor(
        **_xgb_regressor_params(tuned_params, cfg.device, cfg.deterministic, cfg.seed)
    )

    base_model.fit(X_tr, y_tr)
    tuned_model.fit(X_tr, y_tr)

    lr_model = _make_lr_model()
    rf_model = _make_rf_model(cfg.seed, cfg.deterministic)
    lr_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)

    pred_tr_base = base_model.predict(X_tr)
    pred_te_base = base_model.predict(X_te)
    pred_tr_tuned = tuned_model.predict(X_tr)
    pred_te_tuned = tuned_model.predict(X_te)
    pred_tr_lr = lr_model.predict(X_tr)
    pred_te_lr = lr_model.predict(X_te)
    pred_tr_rf = rf_model.predict(X_tr)
    pred_te_rf = rf_model.predict(X_te)

    base_train = regression_metrics(y_tr, pred_tr_base)
    base_test = regression_metrics(y_te, pred_te_base)
    tuned_train = regression_metrics(y_tr, pred_tr_tuned)
    tuned_test = regression_metrics(y_te, pred_te_tuned)
    lr_train = regression_metrics(y_tr, pred_tr_lr)
    lr_test = regression_metrics(y_te, pred_te_lr)
    rf_train = regression_metrics(y_tr, pred_tr_rf)
    rf_test = regression_metrics(y_te, pred_te_rf)

    save_json(run_dir / "holdout_metrics.json", {
        "base_train": base_train,
        "base_test": base_test,
        "tuned_train": tuned_train,
        "tuned_test": tuned_test,
        "lr_train": lr_train,
        "lr_test": lr_test,
        "rf_train": rf_train,
        "rf_test": rf_test,
    })

    # Predictions/residuals for plotting
    a_idx = features.index("A") if "A" in features else 0
    z_idx = features.index("Z") if "Z" in features else None
    n_idx = features.index("N") if "N" in features else None
    preds_df = pd.DataFrame({
        "A": X_te[:, a_idx],
        **({"Z": X_te[:, z_idx]} if z_idx is not None else {}),
        **({"N": X_te[:, n_idx]} if n_idx is not None else {}),
        "y_true": y_te,
        "y_pred_base": pred_te_base,
        "y_pred_tuned": pred_te_tuned,
        "y_pred_lr": pred_te_lr,
        "y_pred_rf": pred_te_rf,
        "resid_base": y_te - pred_te_base,
        "resid_tuned": y_te - pred_te_tuned,
        "resid_lr": y_te - pred_te_lr,
        "resid_rf": y_te - pred_te_rf,
    })
    save_csv(run_dir / "holdout_predictions.csv", preds_df)

    # Feature importance (tuned model)
    fi = pd.Series(tuned_model.feature_importances_, index=features)
    fi_df = fi.reset_index()
    fi_df.columns = ["feature", "importance"]
    save_csv(run_dir / "feature_importance.csv", fi_df.sort_values("importance", ascending=False))

    # Correlation matrix (features + target)
    corr_df = df[features + [f"{target_col}_MeV"]].corr()
    corr_df.to_csv(run_dir / "correlation_matrix.csv")

    # Leave-one-chain-out (optional)
    if cfg.run_loco:
        if cfg.loco_group_col not in df.columns:
            raise KeyError(f"Group column {cfg.loco_group_col} not found in data")

        groups = df[cfg.loco_group_col].values
        a_vals = df["A"].values if "A" in df.columns else np.arange(len(df))
        z_vals = df["Z"].values if "Z" in df.columns else np.full(len(df), np.nan)
        if "N" in df.columns:
            n_vals = df["N"].values
        elif "A" in df.columns and "Z" in df.columns:
            n_vals = df["A"].values - df["Z"].values
        else:
            n_vals = np.full(len(df), np.nan)
        mask = ~pd.isna(groups)

        loco_X = X[mask]
        loco_y = y[mask]
        loco_groups = groups[mask]
        loco_a = a_vals[mask]
        loco_z = z_vals[mask]
        loco_n = n_vals[mask]
        if cfg.holdout_split_path:
            split = load_holdout_split(cfg.holdout_split_path)
            tr_idx, _ = resolve_split_indices(df, split)
            # Restrict LOCO to training split to keep holdout untouched
            loco_mask = np.isin(np.where(mask)[0], tr_idx)
            loco_X = X[mask][loco_mask]
            loco_y = y[mask][loco_mask]
            loco_groups = groups[mask][loco_mask]
            loco_a = a_vals[mask][loco_mask]
            loco_z = z_vals[mask][loco_mask]
            loco_n = n_vals[mask][loco_mask]

        loco_df, loco_preds = _leave_one_group_out(
            loco_X,
            loco_y,
            loco_a,
            loco_z,
            loco_n,
            loco_groups,
            tuned_params=tuned_params,
            device=cfg.device,
            deterministic=cfg.deterministic,
            seed=cfg.seed,
            base_params=BASE_XGB_PARAMS,
            include_base=cfg.loco_include_base,
            min_group_size=cfg.loco_min_group_size,
            max_groups=cfg.loco_max_groups,
        )
        save_csv(run_dir / "loco_group_metrics.csv", loco_df)

        preds_dir = run_dir / "loco_group_predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        for g, df_pred in loco_preds.items():
            save_csv(preds_dir / f"chain_{g}.csv", df_pred)

    # Summary stats
    summary = {
        "target_key": cfg.target_key,
        "feature_set": feature_set,
        "tuned_params_path": tuned_params_path,
        "features": features,
        "cv_folds": cfg.cv_folds,
        "device": cfg.device,
        "deterministic": cfg.deterministic,
        "run_loco": cfg.run_loco,
        "loco_group_col": cfg.loco_group_col,
        "loco_max_groups": cfg.loco_max_groups,
        "loco_min_group_size": cfg.loco_min_group_size,
        "loco_include_base": cfg.loco_include_base,
        "holdout_split_path": cfg.holdout_split_path,
        "holdout_test_size": cfg.holdout_test_size,
        "n_total_clean": split_summary["n_total_clean"],
        "n_train": split_summary["n_train"],
        "n_test": split_summary["n_test"],
        "cv_summary": cv_summary,
    }
    save_json(run_dir / "evaluation_summary.json", summary)

    if cfg.make_plots:
        try:
            plot_from_run(run_dir)
        except Exception as e:
            save_json(run_dir / "plots_error.json", {"error": str(e)})
            print(f"[WARN] Plot generation skipped: {e}")

    return str(run_dir)
