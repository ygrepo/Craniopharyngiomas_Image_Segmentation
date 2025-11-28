import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
)

logger = get_logger(__name__)


def save_image(img: np.ndarray, path: Path):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    logger.info(f"Saving to {path}")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_binary_lasso_cv_metrics(
    df: pd.DataFrame,
    fn: Optional[Path] = None,
    title: str = "Binary LASSO: 5-fold CV Results",
) -> None:
    """
    Plot CV AUC and F1 vs C using the metrics saved by
    *_binary_l1_lasso_cv_metrics.csv.

    Expected columns:
        C,
        mean_auc, std_auc,
        ci_auc_lower_95, ci_auc_upper_95,
        mean_f1, std_f1,
        mean_f1_baseline,
        is_best (optional)
    """
    # Ensure sorted by C
    df = df.sort_values("C").reset_index(drop=True)

    x = df["C"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xscale("log")

    # ------------------------------------------------------------------
    # Left axis: F1 (model) and baseline F1
    # ------------------------------------------------------------------
    mean_f1 = df["mean_f1"].to_numpy()
    std_f1 = df["std_f1"].to_numpy()

    ax1.plot(x, mean_f1, marker="o", label="Mean F1 (model)")
    ax1.fill_between(x, mean_f1 - std_f1, mean_f1 + std_f1, alpha=0.2)

    # Baseline F1 if available
    mean_f1_base = df["mean_f1_baseline"].to_numpy()
    ax1.plot(
        x,
        mean_f1_base,
        linestyle="--",
        marker="^",
        label="Mean F1 (baseline)",
    )

    ax1.set_xlabel("C (log scale)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("F1 score", fontsize=14, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # ------------------------------------------------------------------
    # Right axis: AUC with 95% CI (or std if CI missing)
    # ------------------------------------------------------------------
    ax2 = ax1.twinx()

    mean_auc = df["mean_auc"].to_numpy()
    ax2.plot(x, mean_auc, marker="s", linestyle="--", label="Mean AUC")

    lower = df["ci_auc_lower_95"].to_numpy()
    upper = df["ci_auc_upper_95"].to_numpy()
    ax2.fill_between(x, lower, upper, alpha=0.2, label="AUC 95% CI")

    # Random-guess AUC reference
    ax2.axhline(0.5, color="red", linestyle=":", linewidth=1, label="AUC = 0.5")

    ax2.set_ylabel("AUC", fontsize=14, fontweight="bold")

    # ------------------------------------------------------------------
    # Highlight best C, if flagged
    # ------------------------------------------------------------------
    if "is_best" in df.columns and df["is_best"].any():
        best_row = df.loc[df["is_best"]].iloc[0]
        best_C = best_row["C"]
        best_auc = best_row["mean_auc"]
        best_f1 = best_row["mean_f1"]

        # Highlight on both axes
        ax1.scatter(best_C, best_f1, s=80, edgecolor="k", zorder=5)
        ax2.scatter(best_C, best_auc, s=80, edgecolor="k", zorder=5)

        # Optional vertical line at best C
        ax1.axvline(best_C, color="k", linestyle=":", alpha=0.4)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(title, fontsize=20, fontweight="bold")
    plt.tight_layout()

    if fn is not None:
        fn.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {fn}")
        plt.savefig(fn, dpi=200)
    plt.show()
    plt.close(fig)


def plot_multinomial_lasso_cv_metrics(
    df: pd.DataFrame,
    fn: Optional[Path] = None,
    title: str = "Multinomial LASSO: 5-fold CV Results",
):
    """
    Plot CV macro-AUC and macro-F1 vs C using metrics saved in
    *_multinomial_l1_lasso_cv_metrics.csv.

    Expected columns:
        C,
        mean_auc_macro, std_auc_macro,
        ci_auc_macro_lower_95, ci_auc_macro_upper_95,
        mean_f1_macro, std_f1_macro,
        mean_f1_macro_baseline, std_f1_macro_baseline,
        is_best
    """

    df = df.sort_values("C").reset_index(drop=True)
    x = df["C"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.set_xscale("log")

    # ============================================================
    # LEFT AXIS: Macro-F1
    # ============================================================
    mean_f1 = df["mean_f1_macro"].to_numpy()
    std_f1 = df["std_f1_macro"].to_numpy()

    ax1.plot(
        x,
        mean_f1,
        marker="o",
        color="tab:blue",
        label="Macro F1 (model)",
        linewidth=2,
    )
    ax1.fill_between(
        x,
        mean_f1 - std_f1,
        mean_f1 + std_f1,
        color="tab:blue",
        alpha=0.2,
    )

    # Baseline F1
    mean_f1_base = df["mean_f1_macro_baseline"].to_numpy()
    ax1.plot(
        x,
        mean_f1_base,
        marker="^",
        linestyle="--",
        color="tab:gray",
        label="Macro F1 (baseline)",
        linewidth=2,
    )

    ax1.set_xlabel("C (log scale)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Macro F1 score", fontsize=14, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # ============================================================
    # RIGHT AXIS: Macro-AUC
    # ============================================================
    ax2 = ax1.twinx()

    mean_auc = df["mean_auc_macro"].to_numpy()
    ci_lower = df["ci_auc_macro_lower_95"].to_numpy()
    ci_upper = df["ci_auc_macro_upper_95"].to_numpy()

    ax2.plot(
        x,
        mean_auc,
        marker="s",
        linestyle="--",
        color="tab:red",
        label="Macro AUC",
        linewidth=2,
    )

    ax2.fill_between(
        x,
        ci_lower,
        ci_upper,
        color="tab:red",
        alpha=0.2,
        label="Macro AUC 95% CI",
    )

    ax2.axhline(0.5, color="black", linestyle=":", linewidth=1, label="AUC = 0.5")

    ax2.set_ylabel("Macro AUC", fontsize=14, fontweight="bold")

    # ============================================================
    # Highlight BEST C
    # ============================================================
    if "is_best" in df.columns and df["is_best"].any():
        best = df[df["is_best"]].iloc[0]

        ax1.scatter(
            best["C"],
            best["mean_f1_macro"],
            s=120,
            color="gold",
            edgecolor="black",
            zorder=5,
            label="Best C (F1)",
        )
        ax2.scatter(
            best["C"],
            best["mean_auc_macro"],
            s=120,
            color="gold",
            edgecolor="black",
            zorder=5,
        )
        ax1.axvline(best["C"], color="gold", linestyle=":", alpha=0.5)

    # ============================================================
    # Legend
    # ============================================================
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower left",
        fontsize=11,
    )

    # ============================================================
    # Title
    # ============================================================
    plt.title(title, fontsize=20, fontweight="bold")
    plt.tight_layout()

    if fn is not None:
        fn.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fn, dpi=200)
    plt.show()
    plt.close(fig)
