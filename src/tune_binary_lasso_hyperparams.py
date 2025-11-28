#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
            "Grid search LASSO hyperparameters (C) using 5-fold stratified CV "
            "on the training set for a given model_type (preop/postop)."
        )
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
    )
    ap.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["preop", "postop"],
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="If None, saved as <data_dir>/<model_type>_binary_lasso_cv_metrics.csv",
    )
    ap.add_argument("--log_file", type=Path, default="tune_lasso_hyperparams.log")
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return ap.parse_args()


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y_bin = d["y_bin"]
    feature_names = d["feature_names"]
    return X, y_bin, feature_names


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN LASSO HYPERPARAM TUNING (5-fold CV) ===")
    logger.info(f"model_type = {args.model_type}")

    train_path = args.data_dir / f"{args.model_type}_train_binary_scaled.npz"
    X_train, y_train, feat_names = load_npz(train_path)

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(
        f"Train class counts: {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    # Current refined grid
    C_grid = [30, 50, 100, 150, 200, 300]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_C = None
    best_mean_auc = -np.inf

    results = []

    logger.info("Starting 5-fold CV...")

    for C in C_grid:
        logger.info(f"Evaluating C = {C}")
        fold_aucs = []
        fold_f1s = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                class_weight="balanced",
                C=C,
                max_iter=5000,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)

            # AUC
            if len(np.unique(y_val)) > 1:
                try:
                    y_prob = model.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, y_prob)
                except ValueError:
                    auc = np.nan
            else:
                auc = np.nan

            # F1
            try:
                f1 = f1_score(y_val, model.predict(X_val), average="binary")
            except ValueError:
                f1 = np.nan

            fold_aucs.append(auc)
            fold_f1s.append(f1)

            logger.debug(f"    Fold {fold_idx}: AUC={auc}, F1={f1}")

        mean_auc = float(np.nanmean(fold_aucs))
        std_auc = float(np.nanstd(fold_aucs))
        mean_f1 = float(np.nanmean(fold_f1s))
        std_f1 = float(np.nanstd(fold_f1s))

        logger.info(
            f"C={C}: mean AUC={mean_auc:.3f} (std={std_auc:.3f}), "
            f"mean F1={mean_f1:.3f} (std={std_f1:.3f})"
        )

        results.append(
            {
                "C": C,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
            }
        )

        if mean_auc > best_mean_auc:
            logger.info(f"  [!] New best C = {C} (mean AUC={mean_auc:.3f})")
            best_mean_auc = mean_auc
            best_C = C

    logger.info(f"Best C = {best_C} (mean AUC={best_mean_auc:.3f})")

    # Mark best row
    for row in results:
        row["is_best"] = row["C"] == best_C

    df = pd.DataFrame(results).sort_values("C").reset_index(drop=True)

    # Save CSV
    if args.output_csv is None:
        args.output_csv = (
            args.data_dir / f"{args.model_type}_binary_lasso_cv_metrics.csv"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    logger.info(f"Saved CV metrics to {args.output_csv}")
    logger.info("=== DONE LASSO HYPERPARAM TUNING (5-fold CV) ===")


if __name__ == "__main__":
    main()
