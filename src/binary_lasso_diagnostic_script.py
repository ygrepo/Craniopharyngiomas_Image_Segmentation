#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(
        description=(
            "Diagnostics for L1-Logistic (LASSO): "
            "CV metrics, sparsity and feature-selection stability vs C."
        )
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
        help="Directory containing *_train_binary_scaled.npz or *_train_multinomial_scaled.npz.",
    )
    ap.add_argument(
        "--model_type",
        type=str,
        choices=["preop", "postop"],
        required=True,
        help="Which model design: 'preop' or 'postop'.",
    )
    ap.add_argument(
        "--task",
        type=str,
        choices=["binary", "multinomial"],
        required=True,
        help="Binary (Outcome_Improved) or multinomial (Neurosurgeon_Postop_Visual_Outcome).",
    )
    ap.add_argument(
        "--C_grid",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0],
        help="List of C values to evaluate.",
    )
    ap.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds (StratifiedKFold).",
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Path to save CSV with diagnostics (default: <data_dir>/<model_type>_<task>_C_diagnostics.csv).",
    )
    ap.add_argument(
        "--output_prefix",
        type=Path,
        default=None,
        help=(
            "Prefix for PNG plots (default: <data_dir>/<model_type>_<task>_C_diagnostics). "
            "Plots will be <prefix>_metrics.png and <prefix>_sparsity_stability.png."
        ),
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="lasso_diagnostics.log",
        help="Log file.",
    )
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return ap.parse_args()


def load_npz(path: Path, task: str):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    if task == "binary":
        y = d["y_bin"]
    else:
        y = d["y_multi"]
    feature_names = d["feature_names"]
    return X, y, feature_names


def jaccard_similarity(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return np.nan  # no features selected in either; treat as NaN
    inter = len(s1 & s2)
    union = len(s1 | s2)
    if union == 0:
        return np.nan
    return inter / union


def compute_stability(feature_sets):
    """
    feature_sets: list[set[int]]
    Returns mean pairwise Jaccard similarity.
    """
    pairs = list(combinations(range(len(feature_sets)), 2))
    if not pairs:
        return np.nan
    sims = []
    for i, j in pairs:
        sims.append(jaccard_similarity(feature_sets[i], feature_sets[j]))
    sims = np.array(sims, dtype=float)
    return float(np.nanmean(sims))


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info(
        "=== BEGIN LASSO DIAGNOSTICS (C-grid vs metrics/sparsity/stability) ==="
    )
    logger.info(f"model_type = {args.model_type}, task = {args.task}")
    logger.info(f"C_grid = {args.C_grid}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    if args.task == "binary":
        train_path = args.data_dir / f"{args.model_type}_train_binary_scaled.npz"
    else:
        train_path = args.data_dir / f"{args.model_type}_train_multinomial_scaled.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Train npz not found: {train_path}")

    X, y, feat_names = load_npz(train_path, task=args.task)
    logger.info(f"Train shape: {X.shape}")
    classes, counts = np.unique(y, return_counts=True)
    logger.info(f"Train label distribution: {dict(zip(classes, counts))}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    records = []

    # -------------------------------------------------------------------------
    # Loop over C values
    # -------------------------------------------------------------------------
    for C in args.C_grid:
        logger.info(f"Evaluating C = {C} with {args.n_splits}-fold Stratified CV")

        fold_aucs = []
        fold_f1s = []
        fold_nnz = []
        fold_stability_sets = []  # We'll compute stability from coeff sets

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                n_jobs=-1,
                C=C,
            )
            model.fit(X_tr, y_tr)

            # Coefficients and non-zero counts
            coef = model.coef_  # shape: (n_classes, n_features) or (1, n_features)
            if coef.ndim == 1:  # just in case
                coef = coef.reshape(1, -1)

            # For sparsity: union of non-zero across classes
            nonzero_mask = np.any(coef != 0.0, axis=0)
            nnz = int(nonzero_mask.sum())
            fold_nnz.append(nnz)

            # For stability: store indices of non-zero features
            feature_set = set(np.where(nonzero_mask)[0])
            fold_stability_sets.append(feature_set)

            # Predictions + metrics
            if args.task == "binary":
                # Binary AUC + F1
                y_prob = model.predict_proba(X_val)[:, 1]
                try:
                    auc = roc_auc_score(y_val, y_prob)
                except ValueError:
                    auc = np.nan
                    logger.warning(
                        f"Fold {fold_idx}, C={C}: could not compute AUC, setting NaN."
                    )
                y_pred = model.predict(X_val)
                try:
                    f1 = f1_score(y_val, y_pred)  # binary F1
                except ValueError:
                    f1 = np.nan
                    logger.warning(
                        f"Fold {fold_idx}, C={C}: could not compute F1, setting NaN."
                    )
            else:
                # Multinomial macro-AUC + macro-F1
                auc = np.nan
                if len(np.unique(y_val)) > 1:
                    try:
                        y_prob = model.predict_proba(X_val)
                        auc = roc_auc_score(
                            y_val, y_prob, multi_class="ovr", average="macro"
                        )
                    except ValueError:
                        logger.warning(
                            f"Fold {fold_idx}, C={C}: could not compute macro AUC, setting NaN."
                        )

                y_pred = model.predict(X_val)
                try:
                    f1 = f1_score(y_val, y_pred, average="macro")
                except ValueError:
                    f1 = np.nan
                    logger.warning(
                        f"Fold {fold_idx}, C={C}: could not compute macro F1, setting NaN."
                    )

            fold_aucs.append(auc)
            fold_f1s.append(f1)

            logger.debug(
                f"  Fold {fold_idx}: AUC = {auc if not np.isnan(auc) else 'NaN'}, "
                f"F1 = {f1 if not np.isnan(f1) else 'NaN'}, "
                f"nnz = {nnz}"
            )

        # Aggregate over folds
        fold_aucs = np.array(fold_aucs, dtype=float)
        fold_f1s = np.array(fold_f1s, dtype=float)
        fold_nnz = np.array(fold_nnz, dtype=float)

        mean_auc = float(np.nanmean(fold_aucs))
        std_auc = float(np.nanstd(fold_aucs))
        mean_f1 = float(np.nanmean(fold_f1s))
        std_f1 = float(np.nanstd(fold_f1s))
        mean_nnz = float(np.nanmean(fold_nnz))
        std_nnz = float(np.nanstd(fold_nnz))

        # Feature-selection stability via mean pairwise Jaccard
        stability = compute_stability(fold_stability_sets)

        logger.info(
            f"C = {C}: "
            f"mean AUC = {mean_auc:.3f} (std {std_auc:.3f}), "
            f"mean F1 = {mean_f1:.3f} (std {std_f1:.3f}), "
            f"mean nnz = {mean_nnz:.1f} (std {std_nnz:.1f}), "
            f"stability (Jaccard) = {stability:.3f}"
        )

        records.append(
            {
                "C": C,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_nnz": mean_nnz,
                "std_nnz": std_nnz,
                "stability_jaccard": stability,
            }
        )

    df = pd.DataFrame(records).sort_values("C").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    if args.output_csv is None:
        args.output_csv = (
            args.data_dir / f"{args.model_type}_{args.task}_lasso_C_diagnostics.csv"
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved diagnostics CSV to {args.output_csv}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    if args.output_prefix is None:
        args.output_prefix = (
            args.data_dir / f"{args.model_type}_{args.task}_lasso_C_diagnostics"
        )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1) Metrics vs C
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xscale("log")
    ax1.plot(df["C"], df["mean_f1"], marker="o", label="mean F1", linestyle="-")
    ax1.fill_between(
        df["C"],
        df["mean_f1"] - df["std_f1"],
        df["mean_f1"] + df["std_f1"],
        alpha=0.2,
    )
    ax1.set_xlabel("C (log scale)")
    ax1.set_ylabel("F1")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(df["C"], df["mean_auc"], marker="s", label="mean AUC", linestyle="--")
    ax2.fill_between(
        df["C"],
        df["mean_auc"] - df["std_auc"],
        df["mean_auc"] + df["std_auc"],
        alpha=0.2,
    )
    ax2.set_ylabel("AUC")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"{args.model_type} {args.task} LASSO: CV metrics vs C")
    metrics_png = args.output_prefix.with_name(args.output_prefix.name + "_metrics.png")
    plt.tight_layout()
    plt.savefig(metrics_png, dpi=200)
    plt.close(fig)
    logger.info(f"Saved metrics plot to {metrics_png}")

    # 2) Sparsity & stability vs C
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xscale("log")
    ax1.plot(
        df["C"], df["mean_nnz"], marker="o", label="# non-zero coeffs", linestyle="-"
    )
    ax1.fill_between(
        df["C"],
        df["mean_nnz"] - df["std_nnz"],
        df["mean_nnz"] + df["std_nnz"],
        alpha=0.2,
    )
    ax1.set_xlabel("C (log scale)")
    ax1.set_ylabel("# non-zero coefficients")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(
        df["C"],
        df["stability_jaccard"],
        marker="s",
        linestyle="--",
        label="stability (Jaccard)",
    )
    ax2.set_ylabel("Mean pairwise Jaccard (feature sets)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"{args.model_type} {args.task} LASSO: sparsity & stability vs C")
    sparsity_png = args.output_prefix.with_name(
        args.output_prefix.name + "_sparsity_stability.png"
    )
    plt.tight_layout()
    plt.savefig(sparsity_png, dpi=200)
    plt.close(fig)
    logger.info(f"Saved sparsity/stability plot to {sparsity_png}")

    logger.info("=== DONE LASSO DIAGNOSTICS ===")


if __name__ == "__main__":
    main()
