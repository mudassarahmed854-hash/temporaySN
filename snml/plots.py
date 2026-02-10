from __future__ import annotations

from pathlib import Path
import pandas as pd

from .style import COLORS, set_plot_style


def plot_from_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    set_plot_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Feature importance
    fi_path = run_dir / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(fi["feature"], fi["importance"], color=COLORS["tuned"])
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(run_dir / "fig_feature_importance.pdf")
        plt.close(fig)

    # Correlation heatmap
    corr_path = run_dir / "correlation_matrix.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=True,
            cbar_kws={"shrink": 0.85},
        )
        ax.set_title("Correlation Matrix")
        fig.tight_layout()
        fig.savefig(run_dir / "fig_corr_heatmap.pdf")
        plt.close(fig)

    # Holdout predictions
    pred_path = run_dir / "holdout_predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        # Actual vs predicted (tuned)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(preds["y_true"], preds["y_pred_tuned"], s=12, alpha=0.7, color=COLORS["tuned"])
        lims = [preds["y_true"].min(), preds["y_true"].max()]
        ax.plot(lims, lims, "--", color=COLORS["accent"])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted (tuned)")
        ax.set_title("Actual vs Predicted (Tuned)")
        fig.tight_layout()
        fig.savefig(run_dir / "fig_actual_vs_pred_tuned.pdf")
        plt.close(fig)

        # Actual vs predicted (base)
        if "y_pred_base" in preds.columns:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(preds["y_true"], preds["y_pred_base"], s=12, alpha=0.7, color=COLORS["base"])
            lims = [preds["y_true"].min(), preds["y_true"].max()]
            ax.plot(lims, lims, "--", color=COLORS["accent"])
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted (base)")
            ax.set_title("Actual vs Predicted (Base)")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_actual_vs_pred_base.pdf")
            plt.close(fig)

            # Side-by-side compare (tuned vs base)
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            axes[0].scatter(preds["y_true"], preds["y_pred_tuned"], s=12, alpha=0.7, color=COLORS["tuned"])
            axes[0].plot(lims, lims, "--", color=COLORS["accent"])
            axes[0].set_title("Tuned")
            axes[0].set_xlabel("Actual")
            axes[0].set_ylabel("Predicted")

            axes[1].scatter(preds["y_true"], preds["y_pred_base"], s=12, alpha=0.7, color=COLORS["base"])
            axes[1].plot(lims, lims, "--", color=COLORS["accent"])
            axes[1].set_title("Base")
            axes[1].set_xlabel("Actual")

            fig.suptitle("Actual vs Predicted: Tuned vs Base")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_actual_vs_pred_compare.pdf")
            plt.close(fig)

        # Actual vs predicted (Linear Regression)
        if "y_pred_lr" in preds.columns:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(preds["y_true"], preds["y_pred_lr"], s=12, alpha=0.7, color=COLORS["lr"])
            lims = [preds["y_true"].min(), preds["y_true"].max()]
            ax.plot(lims, lims, "--", color=COLORS["accent"])
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted (LR)")
            ax.set_title("Actual vs Predicted (LR)")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_actual_vs_pred_lr.pdf")
            plt.close(fig)

        # Actual vs predicted (Random Forest)
        if "y_pred_rf" in preds.columns:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(preds["y_true"], preds["y_pred_rf"], s=12, alpha=0.7, color=COLORS["rf"])
            lims = [preds["y_true"].min(), preds["y_true"].max()]
            ax.plot(lims, lims, "--", color=COLORS["accent"])
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted (RF)")
            ax.set_title("Actual vs Predicted (RF)")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_actual_vs_pred_rf.pdf")
            plt.close(fig)

        # Residuals vs A (tuned)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(preds["A"], preds["resid_tuned"], s=12, alpha=0.7, color=COLORS["tuned"])
        ax.axhline(0, linestyle="--", color="k", linewidth=0.8)
        ax.set_xlabel("A")
        ax.set_ylabel("Residual (tuned)")
        ax.set_title("Residuals vs A (Tuned)")
        fig.tight_layout()
        fig.savefig(run_dir / "fig_residuals_vs_A_tuned.pdf")
        plt.close(fig)

        # Residuals vs A (base)
        if "resid_base" in preds.columns:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(preds["A"], preds["resid_base"], s=12, alpha=0.7, color=COLORS["base"])
            ax.axhline(0, linestyle="--", color="k", linewidth=0.8)
            ax.set_xlabel("A")
            ax.set_ylabel("Residual (base)")
            ax.set_title("Residuals vs A (Base)")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_residuals_vs_A_base.pdf")
            plt.close(fig)

        # Residual distributions (tuned vs base overlay)
        if "resid_base" in preds.columns:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(preds["resid_base"], bins=60, alpha=0.55, color=COLORS["base"], label="Base")
            ax.hist(preds["resid_tuned"], bins=60, alpha=0.55, color=COLORS["tuned"], label="Tuned")
            ax.axvline(0, linestyle="--", color="k", linewidth=0.8)
            ax.set_xlabel("Residual")
            ax.set_ylabel("Count")
            ax.set_title("Residual Distribution (Holdout)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(run_dir / "fig_residual_hist_compare.pdf")
            plt.close(fig)

        # Residuals vs predicted (tuned)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(preds["y_pred_tuned"], preds["resid_tuned"], s=12, alpha=0.7, color=COLORS["tuned"])
        ax.axhline(0, linestyle="--", color="k", linewidth=0.8)
        ax.set_xlabel("Predicted (tuned)")
        ax.set_ylabel("Residual (tuned)")
        ax.set_title("Residuals vs Predicted (Tuned)")
        fig.tight_layout()
        fig.savefig(run_dir / "fig_residuals_vs_pred_tuned.pdf")
        plt.close(fig)

        # Residuals vs predicted (base)
        if "y_pred_base" in preds.columns and "resid_base" in preds.columns:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(preds["y_pred_base"], preds["resid_base"], s=12, alpha=0.7, color=COLORS["base"])
            ax.axhline(0, linestyle="--", color="k", linewidth=0.8)
            ax.set_xlabel("Predicted (base)")
            ax.set_ylabel("Residual (base)")
            ax.set_title("Residuals vs Predicted (Base)")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_residuals_vs_pred_base.pdf")
            plt.close(fig)

        # Side-by-side residuals vs A (base vs tuned)
        if "resid_base" in preds.columns:
            y_min = float(min(preds["resid_base"].min(), preds["resid_tuned"].min()))
            y_max = float(max(preds["resid_base"].max(), preds["resid_tuned"].max()))
            fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
            axes[0].scatter(preds["A"], preds["resid_base"], s=12, alpha=0.7, color=COLORS["base"])
            axes[0].axhline(0, linestyle="--", color="k", linewidth=0.8)
            axes[0].set_title("Base")
            axes[0].set_xlabel("A")
            axes[0].set_ylabel("Residual")
            axes[0].set_ylim(y_min, y_max)

            axes[1].scatter(preds["A"], preds["resid_tuned"], s=12, alpha=0.7, color=COLORS["tuned"])
            axes[1].axhline(0, linestyle="--", color="k", linewidth=0.8)
            axes[1].set_title("Tuned")
            axes[1].set_xlabel("A")
            axes[1].set_ylim(y_min, y_max)

            fig.suptitle("Residuals vs A: Base vs Tuned (Holdout)")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_residuals_vs_A_compare.pdf")
            plt.close(fig)

    # CV metrics
    cv_path = run_dir / "cv_fold_metrics.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(cv["fold"], cv["base_rmse"], label="Base RMSE", marker="o", color=COLORS["base"])
        ax.plot(cv["fold"], cv["tuned_rmse"], label="Tuned RMSE", marker="o", color=COLORS["tuned"])
        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE")
        ax.set_title("CV RMSE by Fold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "fig_cv_rmse.pdf")
        plt.close(fig)

        # CV MAE by fold
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(cv["fold"], cv["base_mae"], label="Base MAE", marker="o", color=COLORS["base"])
        ax.plot(cv["fold"], cv["tuned_mae"], label="Tuned MAE", marker="o", color=COLORS["tuned"])
        ax.set_xlabel("Fold")
        ax.set_ylabel("MAE")
        ax.set_title("CV MAE by Fold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "fig_cv_mae.pdf")
        plt.close(fig)

        # CV R2 by fold
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(cv["fold"], cv["base_r2"], label="Base R2", marker="o", color=COLORS["base"])
        ax.plot(cv["fold"], cv["tuned_r2"], label="Tuned R2", marker="o", color=COLORS["tuned"])
        ax.set_xlabel("Fold")
        ax.set_ylabel("R2")
        ax.set_title("CV R2 by Fold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "fig_cv_r2.pdf")
        plt.close(fig)

    # LOCO (leave-one-chain-out) metrics across groups (baseline vs XGBoost)
    loco_path = run_dir / "loco_group_metrics.csv"
    if loco_path.exists():
        loco = pd.read_csv(loco_path)
        if "group" in loco.columns and len(loco) > 0:
            # Prefer numeric sorting for typical use (group=Z), but fall back to string.
            loco["group_num"] = pd.to_numeric(loco["group"], errors="coerce")
            loco = loco.sort_values(["group_num", "group"])
            x = loco["group_num"].where(~loco["group_num"].isna(), loco["group"])

            model_specs = [
                ("tuned", "XGB Tuned", COLORS["tuned"]),
                ("base", "XGB Base", COLORS["base"]),
                ("lr", "Linear Regression", COLORS["lr"]),
                ("rf", "Random Forest", COLORS["rf"]),
            ]

            for metric in ("rmse", "mae", "r2"):
                series = []
                for key, label, color in model_specs:
                    col = f"{key}_{metric}"
                    if col in loco.columns:
                        series.append((col, label, color))
                if not series:
                    continue

                fig, ax = plt.subplots(figsize=(7, 3.5))
                for col, label, color in series:
                    ax.plot(
                        x,
                        loco[col],
                        marker="o",
                        linewidth=1.4,
                        markersize=3.5,
                        label=label,
                        color=color,
                    )
                ax.set_xlabel("Group")
                ax.set_ylabel(metric.upper())
                ax.set_title(f"LOCO {metric.upper()} by Group")
                ax.legend(ncol=2, fontsize=8)
                fig.tight_layout()
                fig.savefig(run_dir / f"fig_loco_{metric}_by_group.pdf")
                plt.close(fig)
