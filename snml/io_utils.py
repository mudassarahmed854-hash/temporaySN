from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(base_dir: str, target_key: str, feature_set: str, stage: str, run_name: Optional[str] = None) -> Path:
    base = Path(base_dir) / target_key / feature_set / stage
    base.mkdir(parents=True, exist_ok=True)
    name = run_name or timestamp()
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run(base_dir: str, target_key: str, feature_set: str, stage: str) -> Optional[Path]:
    base = Path(base_dir) / target_key / feature_set / stage
    if not base.exists():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def save_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: str | Path, df) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def collect_environment(packages: list[str] | None = None) -> dict[str, Any]:
    """Collect minimal environment info for reproducibility (paper-friendly)."""
    import platform
    import sys

    if packages is None:
        packages = [
            "numpy",
            "pandas",
            "scikit-learn",
            "xgboost",
            "optuna",
            "matplotlib",
            "seaborn",
        ]

    versions: dict[str, str | None] = {}
    try:
        from importlib import metadata as importlib_metadata  # py>=3.8
    except Exception:  # pragma: no cover
        importlib_metadata = None

    for pkg in packages:
        v = None
        if importlib_metadata is not None:
            try:
                v = importlib_metadata.version(pkg)
            except Exception:
                v = None
        versions[pkg] = v

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }
