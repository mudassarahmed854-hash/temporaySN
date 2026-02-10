from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from .config import FEATURE_SETS, MAGIC_NUMBERS


def get_feature_list(feature_set: str, use_extended: bool | None = None) -> List[str]:
    if use_extended is not None:
        feature_set = "extended" if use_extended else "basic"
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return list(FEATURE_SETS[feature_set])


def _compute_shell_closure(z: pd.Series, n: pd.Series) -> pd.Series:
    return ((z.isin(MAGIC_NUMBERS)) | (n.isin(MAGIC_NUMBERS))).astype(int)


def add_derived_features(df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
    """Ensure derived features exist and are consistent."""
    out = df.copy()

    if "N" not in out.columns and {"A", "Z"}.issubset(out.columns):
        out["N"] = out["A"] - out["Z"]

    # Recompute to ensure consistency if requested
    if overwrite or "Shell_Closure" not in out.columns:
        if {"Z", "N"}.issubset(out.columns):
            out["Shell_Closure"] = _compute_shell_closure(out["Z"], out["N"])

    if overwrite or "N_Z_excess" not in out.columns:
        if {"N", "Z"}.issubset(out.columns):
            out["N_Z_excess"] = out["N"] - out["Z"]

    if overwrite or "N_to_Z_ratio" not in out.columns:
        if {"N", "Z"}.issubset(out.columns):
            z = out["Z"].replace(0, np.nan)
            out["N_to_Z_ratio"] = out["N"] / z

    return out
