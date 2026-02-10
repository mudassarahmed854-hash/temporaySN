from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import TuningConfig, EvalConfig, OUTPUTS_DIR
from .eda import run_eda
from .tuning import run_tuning
from .evaluation import run_evaluation
from .splits import create_holdout_split, create_train_csv_from_split
from .splits import load_holdout_split, file_sha256
from .io_utils import make_run_dir, save_json


def run_pipeline(tune_cfg: TuningConfig, eval_cfg: EvalConfig) -> Dict[str, str]:
    """Run tuning then evaluation, wiring best_params.csv to evaluation."""
    original_data_path = tune_cfg.data_path
    feature_set = "extended" if tune_cfg.use_extended else tune_cfg.feature_set

    # Ensure a strict holdout exists for evaluation, but keep it hidden from tuning
    split_path = None
    if eval_cfg.holdout_split_path is None:
        split_path = Path(OUTPUTS_DIR) / "splits" / f"{tune_cfg.target_key}_{'extended' if tune_cfg.use_extended else tune_cfg.feature_set}_holdout.json"
        if split_path.exists():
            # Reuse the existing strict holdout for reproducibility.
            existing = load_holdout_split(split_path)
            if (existing.get("seed") != tune_cfg.seed) or (existing.get("test_size") != tune_cfg.holdout_test_size):
                print(
                    f"[WARN] Reusing existing holdout split with seed={existing.get('seed')} "
                    f"test_size={existing.get('test_size')} (current config seed={tune_cfg.seed} "
                    f"test_size={tune_cfg.holdout_test_size})."
                )
            existing_hash = existing.get("data_sha256")
            current_hash = file_sha256(original_data_path)
            if existing_hash and current_hash and existing_hash != current_hash:
                print(
                    "[WARN] Holdout split data SHA mismatch. The split file was created on a different "
                    "CSV content than the current data_path. For strict reproducibility, delete the "
                    f"split file and re-run: {split_path}"
                )
        else:
            create_holdout_split(
                data_path=tune_cfg.data_path,
                target_key=tune_cfg.target_key,
                test_size=tune_cfg.holdout_test_size,
                seed=tune_cfg.seed,
                out_path=split_path,
            )
        eval_cfg.holdout_split_path = str(split_path)
    else:
        split_path = Path(eval_cfg.holdout_split_path)

    # Dataset report (paper-friendly, includes asterisk/hash counts + summary stats)
    try:
        eda_run_name = f"{tune_cfg.run_name}_eda" if tune_cfg.run_name else None
        eda_dir = make_run_dir(OUTPUTS_DIR, tune_cfg.target_key, feature_set, "eda", eda_run_name)
        run_eda(original_data_path, tune_cfg.target_key, output_dir=eda_dir)

        split = load_holdout_split(split_path)
        save_json(eda_dir / "split_summary.json", {
            "holdout_split_path": str(split_path),
            "n_total": split.get("n_total"),
            "n_train": split.get("n_train"),
            "n_test": split.get("n_test"),
            "seed": split.get("seed"),
            "test_size": split.get("test_size"),
        })
    except Exception as e:
        # Reporting should not block training runs
        try:
            save_json(eda_dir / "eda_error.json", {"error": str(e)})
        except Exception:
            pass
        print(f"[WARN] EDA/report step skipped: {e}")
        eda_dir = None

    # Create a train-only CSV so tuning never sees the holdout split directly
    train_csv = Path(OUTPUTS_DIR) / "splits" / f"{tune_cfg.target_key}_{'extended' if tune_cfg.use_extended else tune_cfg.feature_set}_train.csv"
    if split_path is not None:
        create_train_csv_from_split(
            data_path=tune_cfg.data_path,
            target_key=tune_cfg.target_key,
            split_path=split_path,
            out_path=train_csv,
        )
        tune_cfg.data_path = str(train_csv)
        tune_cfg.holdout_split_path = None

    tune_dir = Path(run_tuning(tune_cfg))
    best_csv = tune_dir / "best_params.csv"
    best_json = tune_dir / "best_params.json"

    if best_csv.exists():
        eval_cfg.tuned_params_path = str(best_csv)
    elif best_json.exists():
        eval_cfg.tuned_params_path = str(best_json)

    eval_dir = run_evaluation(eval_cfg)
    out = {"tuning_dir": str(tune_dir), "evaluation_dir": str(eval_dir)}
    if eda_dir is not None:
        out["eda_dir"] = str(eda_dir)
    return out
