from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from .config import TARGETS
from .features import add_derived_features


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        c2 = c.replace("\ufeff", "").strip()
        cols.append(c2)
    df = df.copy()
    df.columns = cols
    return df


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    # Work with strings, keep digits, '.', '-', '+', 'e', 'E'
    s = s.astype(str).str.strip()
    # Remove trailing hashes, e.g., "2001#"
    s = s.str.replace("#", "", regex=False)
    # Remove stray commas
    s = s.str.replace(",", "", regex=False)
    # Replace asterisks and empty/nan strings with NaN (case-insensitive)
    s_lower = s.str.lower()
    s = s.mask(s_lower.isin({"*", "", "nan", "none"}), np.nan)
    # Convert to numeric
    return pd.to_numeric(s, errors="coerce")


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="latin1")
    df = _normalize_columns(df)
    return df


def clean_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw dataframe and return clean df + cleaning report."""
    df = df_raw.copy()

    # Drop BOM/index-like columns
    drop_cols = [c for c in df.columns if c.lower() in {"ï»¿1", "1", "index"}]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Identify numeric columns (everything except elt)
    string_cols = {"elt"}
    numeric_cols = [c for c in df.columns if c not in string_cols]

    report = {
        "dropped_columns": drop_cols,
        "numeric_columns": numeric_cols,
        "string_columns": list(string_cols),
    }

    # Clean numeric columns
    for c in numeric_cols:
        df[c] = _clean_numeric_series(df[c])

    return df, report


def clean_loaded_data(
    df_raw: pd.DataFrame,
    target_key: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean an already-loaded raw dataframe and return clean df + report."""
    if target_key not in TARGETS:
        raise ValueError(f"Unknown target_key: {target_key}")

    raw = _normalize_columns(df_raw)
    clean, report = clean_dataframe(raw)
    clean = add_derived_features(clean, overwrite=True)

    # Convert target to MeV and keep a dedicated column
    target_col = TARGETS[target_key]
    if target_col not in clean.columns:
        raise KeyError(f"Target column {target_col} not found in data")

    # Raw target flag stats (useful for paper/EDA reporting)
    try:
        raw_target = raw[target_col].astype(str).str.strip()
        report["target_raw_asterisk_count"] = int(raw_target.str.contains(r"\*", regex=True, na=False).sum())
        report["target_raw_hash_count"] = int(raw_target.str.contains("#", regex=False, na=False).sum())
        report["target_raw_emptyish_count"] = int(raw_target.isin(["", "nan", "NaN", "None"]).sum())
    except Exception:
        report["target_raw_asterisk_count"] = None
        report["target_raw_hash_count"] = None
        report["target_raw_emptyish_count"] = None

    clean[f"{target_col}_MeV"] = clean[target_col] / 1000.0

    # Track how many rows would be dropped due to missing target after cleaning
    target_mev_col = f"{target_col}_MeV"
    report["rows_after_numeric_clean"] = int(len(clean))
    missing_mask = clean[target_mev_col].isna()
    report["target_missing_after_clean_count"] = int(missing_mask.sum())
    # Of the rows that become missing after cleaning/parsing, attribute obvious flag patterns.
    try:
        raw_target2 = raw[target_col].astype(str).str.strip()
        report["target_missing_with_asterisk_count"] = int(raw_target2[missing_mask].str.contains(r"\*", regex=True, na=False).sum())
        report["target_missing_with_hash_count"] = int(raw_target2[missing_mask].str.contains("#", regex=False, na=False).sum())
    except Exception:
        report["target_missing_with_asterisk_count"] = None
        report["target_missing_with_hash_count"] = None

    # Drop rows missing target
    before_drop = len(clean)
    clean = clean.dropna(subset=[target_mev_col])
    report["rows_dropped_missing_target"] = int(before_drop - len(clean))

    # Quick unit sanity stats (helps spot keV/MeV mixups early)
    try:
        y_mev = clean[target_mev_col]
        report["target_mev_min"] = float(y_mev.min())
        report["target_mev_max"] = float(y_mev.max())
        report["target_mev_mean"] = float(y_mev.mean())
        report["target_mev_std"] = float(y_mev.std(ddof=1)) if len(y_mev) > 1 else 0.0
    except Exception:
        report["target_mev_min"] = None
        report["target_mev_max"] = None
        report["target_mev_mean"] = None
        report["target_mev_std"] = None

    # Duplicate nucleus check (can create leakage if duplicates fall across splits)
    try:
        if {"A", "Z", "N"}.issubset(clean.columns):
            dup = int(clean.duplicated(subset=["A", "Z", "N"]).sum())
            report["duplicate_rows_by_A_Z_N"] = dup
            report["unique_nuclei_by_A_Z_N"] = int(clean.drop_duplicates(subset=["A", "Z", "N"]).shape[0])
    except Exception:
        report["duplicate_rows_by_A_Z_N"] = None
        report["unique_nuclei_by_A_Z_N"] = None

    report["target_col"] = target_col
    report["rows_before"] = len(raw)
    report["rows_after_clean"] = len(clean)

    return clean, report


def load_clean_data(
    data_path: str,
    target_key: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load raw CSV, clean numeric columns, add derived features, and return clean df + report."""
    raw = load_raw_csv(data_path)
    clean, report = clean_loaded_data(raw, target_key)
    return clean, report
