from __future__ import annotations

from pathlib import Path
import pandas as pd

from .data import load_raw_csv, clean_loaded_data
from .io_utils import ensure_dir, save_csv, save_json, collect_environment
from .config import OUTPUTS_DIR, TARGETS
from .style import COLORS, set_plot_style


def _flag_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df_raw.columns:
        s = df_raw[col].astype(str)
        asterisk = s.str.contains(r"\*", regex=True, na=False).sum()
        hash_ = s.str.contains(r"#", regex=True, na=False).sum()
        non_numeric = s.str.contains(r"[^0-9eE+\-\. ]", regex=True, na=False).sum()
        rows.append({
            "column": col,
            "asterisk_count": int(asterisk),
            "hash_count": int(hash_),
            "non_numeric_count": int(non_numeric),
            "total_rows": int(len(s)),
        })
    return pd.DataFrame(rows).sort_values("non_numeric_count", ascending=False)


def _missingness_report(df_clean: pd.DataFrame) -> pd.DataFrame:
    miss = df_clean.isna().sum().reset_index()
    miss.columns = ["column", "missing_count"]
    miss["missing_pct"] = (miss["missing_count"] / len(df_clean)) * 100.0
    return miss.sort_values("missing_count", ascending=False)


def run_eda(data_path: str, target_key: str, output_dir: str | Path | None = None) -> Path:
    if target_key not in TARGETS:
        raise ValueError(f"Unknown target_key: {target_key}")

    raw = load_raw_csv(data_path)
    clean, report = clean_loaded_data(raw, target_key)

    out_dir = Path(output_dir) if output_dir else Path(OUTPUTS_DIR) / "eda"
    ensure_dir(out_dir)

    # Reports
    save_json(out_dir / "environment.json", collect_environment())
    flag_df = _flag_report(raw)
    miss_df = _missingness_report(clean)
    desc_df = clean.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})

    save_csv(out_dir / "flags_report.csv", flag_df)
    save_csv(out_dir / "missingness_report.csv", miss_df)
    save_csv(out_dir / "summary_stats.csv", desc_df)
    save_json(out_dir / "cleaning_report.json", report)

    # Dataset overview / sanity checks (helps prevent paper-vs-code ambiguity)
    overview = {
        "data_path": data_path,
        "target_key": target_key,
        "target_col": report.get("target_col"),
        "rows_before": report.get("rows_before"),
        "rows_after_clean": report.get("rows_after_clean"),
        "rows_dropped_missing_target": report.get("rows_dropped_missing_target"),
        "target_raw_asterisk_count": report.get("target_raw_asterisk_count"),
        "target_raw_hash_count": report.get("target_raw_hash_count"),
        "target_raw_emptyish_count": report.get("target_raw_emptyish_count"),
    }
    dup_cols = [c for c in ("A", "Z", "N") if c in clean.columns]
    if len(dup_cols) >= 2:
        overview["duplicate_rows_by_" + "_".join(dup_cols)] = int(clean.duplicated(subset=dup_cols).sum())
    else:
        overview["duplicate_rows_by_A_Z_N"] = None
    for c in ("A", "Z", "N", "elt"):
        if c in clean.columns:
            overview[f"unique_{c}"] = int(pd.Series(clean[c]).nunique(dropna=True))
    save_json(out_dir / "dataset_overview.json", overview)

    # Basic plots
    set_plot_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot missingness
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    miss_plot = miss_df.head(20)
    ax1.barh(miss_plot["column"], miss_plot["missing_count"], color=COLORS["base"])
    ax1.set_title("Top Missing Columns")
    ax1.set_xlabel("Missing Count")
    fig1.tight_layout()
    fig1.savefig(out_dir / "missingness_top20.pdf")
    plt.close(fig1)

    # Target distribution (MeV)
    target_col = TARGETS[target_key]
    target_mev = f"{target_col}_MeV"
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(clean[target_mev].dropna(), bins=50, color=COLORS["tuned"], alpha=0.85)
    ax2.set_title(f"{target_col} Distribution (MeV)")
    ax2.set_xlabel("MeV")
    ax2.set_ylabel("Count")
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{target_key}_distribution.pdf")
    plt.close(fig2)

    # Flags overview
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    flag_plot = flag_df.head(20)
    ax3.barh(flag_plot["column"], flag_plot["non_numeric_count"], color=COLORS["accent"])
    ax3.set_title("Top Non-Numeric Columns (flags, strings)")
    ax3.set_xlabel("Non-numeric Count")
    fig3.tight_layout()
    fig3.savefig(out_dir / "non_numeric_top20.pdf")
    plt.close(fig3)

    return out_dir
