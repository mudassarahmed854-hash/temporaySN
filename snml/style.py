from __future__ import annotations

from pathlib import Path
import os

# Consistent, journal-friendly palette (colorblind-aware, muted)
COLORS = {
    "tuned": "#1B4965",
    "base": "#5FA8D3",
    "lr": "#CA6702",
    "rf": "#6A994E",
    "accent": "#9B2226",
    "neutral": "#495057",
}


def set_plot_style(outputs_dir: str | None = None) -> None:
    """Apply a consistent plotting style across the project.

    - Uses Matplotlib PDF outputs (vector) with sane defaults.
    - Keeps colors consistent across tuning/eval/EDA figures.
    """
    if outputs_dir is None:
        outputs_dir = os.getenv("SNML_OUTPUTS_DIR", "outputs")

    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_config_dir = Path(outputs_dir) / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    # Import inside the function so this module stays lightweight.
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.8,
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        # Embed TrueType fonts in vector exports.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
