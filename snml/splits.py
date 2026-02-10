from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import hashlib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from .data import load_clean_data


def file_sha256(path: str | Path) -> str | None:
    """Compute a sha256 for the raw CSV to detect split/data mismatches."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_keys(df: pd.DataFrame, key_cols: list[str]) -> pd.Series | None:
    """Build a stable row key like 'A_Z_N' for split portability."""
    if not set(key_cols).issubset(df.columns):
        return None

    key_df = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in key_cols})
    # Nuclear identifiers should be integral; rounding avoids float formatting issues.
    key_df = key_df.round().astype("Int64")
    keys = key_df.astype(str).agg("_".join, axis=1)
    invalid = key_df.isna().any(axis=1)
    return keys.mask(invalid, other=pd.NA)


def resolve_split_indices(df: pd.DataFrame, split: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Resolve train/test indices from a split payload.

    Preference order:
    1) Use index-based split if valid for the current cleaned dataframe.
    2) Otherwise, use key-based split (A/Z/N) if present and unambiguous.
    """
    n = int(len(df))

    tr_raw = split.get("train_idx")
    te_raw = split.get("test_idx")
    if tr_raw is not None and te_raw is not None:
        try:
            tr_idx = np.array(tr_raw, dtype=int)
            te_idx = np.array(te_raw, dtype=int)
            if len(tr_idx) and len(te_idx) and tr_idx.min() >= 0 and te_idx.min() >= 0 and tr_idx.max() < n and te_idx.max() < n:
                return tr_idx, te_idx
        except Exception:
            pass

    key_cols = split.get("key_cols") or ["A", "Z", "N"]
    tr_keys = split.get("train_keys")
    te_keys = split.get("test_keys")
    if tr_keys is None or te_keys is None:
        raise ValueError("Split payload is missing usable train/test indices and has no key-based fallback.")

    keys = _row_keys(df, list(key_cols))
    if keys is None:
        raise ValueError(f"Cannot resolve split via keys: cleaned dataframe missing columns {key_cols}.")
    if keys.isna().any():
        raise ValueError("Cannot resolve split via keys: some rows have missing A/Z/N after cleaning.")
    if keys.duplicated().any():
        raise ValueError("Cannot resolve split via keys: duplicate A/Z/N keys exist; mapping would be ambiguous.")

    key_to_idx = {k: i for i, k in enumerate(keys.astype(str).tolist())}

    def _map(keys_list: list[Any]) -> np.ndarray:
        idxs: list[int] = []
        missing: list[Any] = []
        for k in keys_list:
            if k is None:
                missing.append(k)
                continue
            kk = str(k)
            j = key_to_idx.get(kk)
            if j is None:
                missing.append(kk)
            else:
                idxs.append(int(j))
        if missing:
            raise ValueError(f"Key-based split could not be mapped to the current data. Missing keys: {missing[:10]}")
        return np.array(idxs, dtype=int)

    return _map(tr_keys), _map(te_keys)


def create_holdout_split(
    data_path: str,
    target_key: str,
    test_size: float,
    seed: int,
    out_path: str | Path,
) -> Path:
    """Create a strict holdout split and save indices for reuse."""
    df, _ = load_clean_data(data_path, target_key)
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = _row_keys(df, ["A", "Z", "N"])
    payload: Dict[str, Any] = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_path": data_path,
        "data_sha256": file_sha256(data_path),
        "target_key": target_key,
        "test_size": float(test_size),
        "seed": int(seed),
        "n_total": int(len(df)),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "train_idx": tr_idx.tolist(),
        "test_idx": te_idx.tolist(),
    }
    if keys is not None and (not keys.isna().any()) and (not keys.duplicated().any()):
        payload["key_cols"] = ["A", "Z", "N"]
        payload["train_keys"] = keys.iloc[tr_idx].astype(str).tolist()
        payload["test_keys"] = keys.iloc[te_idx].astype(str).tolist()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path


def load_holdout_split(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def create_train_csv_from_split(
    data_path: str,
    target_key: str,
    split_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Create a cleaned train-only CSV from a holdout split."""
    split = load_holdout_split(split_path)
    df, _ = load_clean_data(data_path, target_key)
    tr_idx, _ = resolve_split_indices(df, split)
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_path, index=False)
    return out_path
