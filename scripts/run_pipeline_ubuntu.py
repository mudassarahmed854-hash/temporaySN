from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snml.config import TuningConfig, EvalConfig
from snml.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNML pipeline on Ubuntu (CPU)")
    parser.add_argument("--target", default="sn", choices=["sn", "s2n"])
    parser.add_argument("--use-extended", action="store_true")
    parser.add_argument("--all-targets", action="store_true", help="Run both sn and s2n")
    parser.add_argument("--all-feature-sets", action="store_true", help="Run both basic and extended features")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="ubuntu_run")
    outputs_dir = Path(os.getenv("SNML_OUTPUTS_DIR", "outputs"))
    parser.add_argument("--storage", default=f"sqlite:///{(outputs_dir / 'optuna.db').as_posix()}")
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--loco-max-groups", type=int, default=5, help="0 = all groups")
    parser.add_argument("--loco-min-group-size", type=int, default=5)
    parser.add_argument("--no-loco", action="store_true", help="Disable LOCO evaluation")
    args = parser.parse_args()

    targets = ["sn", "s2n"] if args.all_targets else [args.target]
    feature_flags = [False, True] if args.all_feature_sets else [args.use_extended]

    results = []
    for target in targets:
        for use_extended in feature_flags:
            feature_set = "extended" if use_extended else "basic"
            run_name = f"{args.run_name}_{target}_{feature_set}" if (args.all_targets or args.all_feature_sets) else args.run_name
            loco_max = None if args.loco_max_groups <= 0 else args.loco_max_groups

            tune_cfg = TuningConfig(
                data_path="finaldata.csv",
                target_key=target,
                use_extended=use_extended,
                n_trials=args.n_trials,
                cv_folds=args.cv_folds,
                seed=args.seed,
                device="cpu",
                deterministic=True,
                run_name=run_name,
                storage=args.storage,
                n_jobs=args.n_jobs,
                # Keep study names unique per run to avoid accidentally reusing prior Optuna studies
                study_name=f"xgb_{run_name}".replace(" ", "_"),
            )

            eval_cfg = EvalConfig(
                data_path="finaldata.csv",
                target_key=target,
                use_extended=use_extended,
                cv_folds=args.cv_folds,
                seed=args.seed,
                device="cpu",
                deterministic=True,
                run_name=f"{run_name}_eval",
                run_loco=not args.no_loco,
                loco_group_col="Z",
                loco_max_groups=loco_max,
                loco_min_group_size=args.loco_min_group_size,
                loco_include_base=True,
            )

            results.append(run_pipeline(tune_cfg, eval_cfg))

    print(results)


if __name__ == "__main__":
    main()
