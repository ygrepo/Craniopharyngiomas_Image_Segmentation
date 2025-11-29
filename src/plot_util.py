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
    title_fontsize: int = 20,
    title_fontweight: str = "bold",
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

    plt.title(title, fontsize=title_fontsize, fontweight=title_fontweight)
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


def plot_feature_importances(
    df: pd.DataFrame,
    title: str,
    figsize: tuple = (16, 8),
    feature_col: str = "feature",
    importance_col: str = "importance",
    rank_col: str = "rank_within_group",
    output: Path | None = None,
    top_n: int | None = 20,  # Show only top N features
    show_values: bool = True,  # Show importance values on bars
):
    # Basic validation
    for col in [feature_col, importance_col, rank_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    # Sort by importance descending and take top N
    if top_n is not None:
        df_plot = (
            df.sort_values(importance_col, ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
    else:
        df_plot = df.sort_values(importance_col, ascending=False).reset_index(drop=True)

    # Clean feature names (remove prefixes, truncate long names)
    features = df_plot[feature_col].astype(str).tolist()
    features_clean = []
    for feat in features:
        # Remove common prefixes
        clean_feat = feat.replace("Preop_", "").replace("_", " ")
        # Truncate long names
        if len(clean_feat) > 25:
            clean_feat = clean_feat[:22] + "..."
        features_clean.append(clean_feat)

    importances = df_plot[importance_col].astype(float).tolist()
    ranks = df_plot[rank_col].tolist()

    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=figsize)

    # Use consistent blue color from viridis (around 0.6 gives a nice blue)
    blue_color = plt.cm.viridis(0.3)

    x = range(len(features_clean))
    bars = ax.bar(x, importances, color=blue_color, edgecolor="black", linewidth=0.5)

    # Add rank and importance values on top of bars
    if show_values:
        for xi, bar, importance, rank in zip(x, bars, importances, ranks):
            height = bar.get_height()
            # Show rank (bold) and importance value on top
            ax.text(
                xi,
                height + max(importances) * 0.01,
                f"{rank}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Improve x-axis labels
    ax.set_xticks(list(x))
    if top_n is not None:
        ax.set_xticklabels(
            features_clean,
            rotation=45,
            ha="right",
            fontsize=11,
            fontweight="normal",
        )
    else:
        ax.set_xticklabels(
            features_clean,
            rotation=45,
            ha="right",
            fontsize=6,
            fontweight="normal",
        )

    # Styling improvements
    ax.set_ylabel("Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Features", fontsize=14, fontweight="bold")
    if top_n is not None:
        ax.set_title(
            f"{title}\n(Top {top_n} Features)",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
    else:
        ax.set_title(
            f"{title}",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )

    # Improve grid
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="gray")
    ax.set_axisbelow(True)

    # Set y-axis to start from 0 and add some padding at top for labels
    ax.set_ylim(0, max(importances) * 1.25)

    # Improve spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Add summary statistics as text box
    mean_importance = np.mean(importances)
    if top_n is not None:
        ax.text(
            0.02,
            0.98,
            f"Mean Importance (top {top_n}): {mean_importance:.4f}\nTotal Features Shown: {len(features_clean)}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
    else:
        ax.text(
            0.02,
            0.98,
            f"Mean Importance: {mean_importance:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_feature_importance_accumulation(
    df: pd.DataFrame,
    prefix: str,
    target_threshold: float = 0.90,
    output: Path | None = None,
):
    # Sort by importance descending
    df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Calculate cumulative importance
    df["cumulative_importance"] = df["importance"].cumsum()
    total_importance = df["importance"].sum()
    df["cumulative_importance"] /= total_importance

    # --- Alignment Logic ---
    # Define the "Elbow" / Cutoff point based on the Cumulative Threshold
    # Find the rank where cumulative importance hits the target threshold
    cutoff_idx = df[df["cumulative_importance"] >= target_threshold].index[0]
    cutoff_rank = cutoff_idx + 1
    cutoff_importance_val = df.iloc[cutoff_idx]["importance"]
    cutoff_cumulative_val = df.iloc[cutoff_idx]["cumulative_importance"]

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Individual Importance (Left)
    ax1.plot(df.index + 1, df["importance"], linewidth=2, label="Importance Curve")

    # Mark the point corresponding to the 90% threshold
    ax1.plot(
        cutoff_rank,
        cutoff_importance_val,
        "ro",
        markersize=8,
        label=f"{int(target_threshold*100)}% Cutoff (Rank {cutoff_rank})",
    )
    ax1.axhline(
        y=cutoff_importance_val,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Importance > {cutoff_importance_val:.4f}",
    )
    ax1.axvline(x=cutoff_rank, color="r", linestyle=":", alpha=0.5)

    ax1.set_title(
        f"{prefix} Feature Importance Decay\n({int(target_threshold*100)}% Cumulative Signal)",
        fontsize=18,
        fontweight="bold",
    )
    ax1.set_xlabel("Feature Rank", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Importance Score", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Cumulative Importance (Right)
    ax2.plot(
        df.index + 1,
        df["cumulative_importance"],
        linewidth=2,
        color="orange",
        label="Cumulative Curve",
    )

    # Mark the Threshold
    ax2.axhline(
        y=target_threshold,
        color="g",
        linestyle="--",
        label=f"{int(target_threshold*100)}% Threshold",
    )
    ax2.axvline(
        x=cutoff_rank, color="g", linestyle=":", label=f"Met at Rank {cutoff_rank}"
    )
    ax2.plot(cutoff_rank, cutoff_cumulative_val, "ro", markersize=8)

    ax2.set_title(
        f"{prefix} Cumulative Feature Importance", fontsize=18, fontweight="bold"
    )
    ax2.set_xlabel("Number of Features Kept", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Cumulative Importance", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Alternative: Single elbow plot with enhanced features
def plot_simple_elbow(
    df: pd.DataFrame,
    prefix: str,
    figsize: tuple = (12, 8),
    output: Path | None = None,
    top_n: int = 50,
):

    df_plot = df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    if top_n is not None:
        df_plot = df_plot.head(top_n)

    # Create single plot
    fig, ax = plt.subplots(figsize=figsize)

    x_vals = df_plot.index + 1

    # Main elbow curve
    ax.plot(
        x_vals,
        df_plot["importance"],
        "o-",
        linewidth=3,
        markersize=6,
        color="steelblue",
        alpha=0.8,
        label="Feature Importance",
    )

    # Add trend line for comparison
    if len(df_plot) > 3:
        z = np.polyfit(x_vals, df_plot["importance"], 2)  # Quadratic fit
        p = np.poly1d(z)
        ax.plot(
            x_vals,
            p(x_vals),
            "--",
            color="red",
            alpha=0.6,
            linewidth=2,
            label="Trend (Quadratic)",
        )

    # Styling
    ax.set_title(
        f"{prefix} Feature Importance Elbow Plot", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Feature Rank", fontsize=14, fontweight="bold")
    #    ax.set_xticklabels([f"{x}" for x in x_vals], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Importance Score", fontsize=14, fontweight="bold")
    # ax.set_yticklabels(
    #     [f"{y:.4f}" for y in ax.get_yticks()], fontsize=10, fontweight="normal"
    # )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    # Add statistics box
    stats_text = f"Total Features: {len(df_plot)}\n"
    stats_text += f'Max Importance: {df_plot["importance"].max():.4f}\n'
    stats_text += f'Min Importance: {df_plot["importance"].min():.4f}\n'
    stats_text += f'Mean: {df_plot["importance"].mean():.4f}'

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)
