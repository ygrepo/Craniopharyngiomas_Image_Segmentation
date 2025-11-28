#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
        help=(
            "Directory containing <model_type>_train_binary_scaled.npz "
            "(output of create_binary_lasso_features.py)."
        ),
    )
    ap.add_argument(
        "--model_type",
        type=str,
        choices=["preop", "postop"],
        required=True,
        help="Which model design to tune: 'preop' or 'postop'.",
    )
    ap.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help=(
            "Path to save best hyperparameters. "
            "If not provided, defaults to <data_dir>/<model_type>_binary_lasso_hyperparams.json"
        ),
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

    # Expecting file like preop_train_binary_scaled.npz
    train_path = args.data_dir / f"{args.model_type}_train_binary_scaled.npz"

    X_train, y_train_bin, feat_names = load_npz(train_path)
    logger.info(f"Train shape: {X_train.shape}")
    logger.info(
        f"Train class counts: {dict(zip(*np.unique(y_train_bin, return_counts=True)))}"
    )

    # C_grid = [0.01, 0.1, 1.0, 10.0]
    # C_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    C_grid = [30, 50, 100, 150, 200, 300]

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Tuning binary LASSO (Outcome_Improved) with 5-fold stratified CV...")
    best_C_bin = None
    best_mean_auc = -np.inf
    best_mean_f1 = -np.inf

    cv_results = {}

    for C in C_grid:
        logger.info(f"Evaluating C = {C} (binary) with 5-fold CV")
        fold_aucs = []
        fold_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train_bin), start=1
        ):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train_bin[train_idx], y_train_bin[val_idx]

            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                n_jobs=-1,
                C=C,
            )
            model.fit(X_tr, y_tr)

            # AUC
            if len(np.unique(y_val)) > 1:
                y_prob = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_prob)
            else:
                auc = np.nan
                logger.warning(
                    f"Fold {fold_idx}: val set has a single class for binary outcome; AUC set to NaN."
                )

            # F1 (binary)
            y_pred = model.predict(X_val)
            try:
                f1 = f1_score(y_val, y_pred, average="binary")
            except ValueError as e:
                logger.warning(
                    f"Fold {fold_idx}: could not compute F1 (binary), setting F1 to NaN. Error: {e}"
                )
                f1 = np.nan

            fold_aucs.append(auc)
            fold_f1s.append(f1)

            logger.info(
                f"  Fold {fold_idx}: AUC = {auc if not np.isnan(auc) else 'NaN'}, "
                f"F1 = {f1 if not np.isnan(f1) else 'NaN'}"
            )

        mean_auc = float(np.nanmean(fold_aucs))
        mean_f1 = float(np.nanmean(fold_f1s))
        std_auc = float(np.nanstd(fold_aucs))
        std_f1 = float(np.nanstd(fold_f1s))

        logger.info(
            f"C = {C}: mean AUC = {mean_auc:.3f} (std {std_auc:.3f}), "
            f"mean F1 = {mean_f1:.3f} (std {std_f1:.3f})"
        )

        cv_results[C] = {
            "fold_aucs": [None if np.isnan(a) else float(a) for a in fold_aucs],
            "fold_f1s": [None if np.isnan(f) else float(f) for f in fold_f1s],
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        }

        # Select C by mean AUC; F1 is secondary diagnostic
        if mean_auc > best_mean_auc:
            logger.info(f"  [!] New best C (binary) = {C}, mean AUC = {mean_auc:.3f}")
            best_mean_auc = mean_auc
            best_mean_f1 = mean_f1
            best_C_bin = C

    logger.info(
        f"Best C (binary) = {best_C_bin}, "
        f"CV mean AUC = {best_mean_auc:.3f}, "
        f"CV mean F1 = {best_mean_f1:.3f}"
    )

    # Save best hyperparameters + CV summary
    hparams = {
        "model_type": args.model_type,
        "binary": {
            "C": best_C_bin,
            "cv_mean_auc": best_mean_auc,
            "cv_mean_f1": best_mean_f1,
            "cv_results_per_C": cv_results,
        },
    }

    if args.output_json is None:
        args.output_json = (
            args.data_dir / f"{args.model_type}_binary_lasso_hyperparams.json"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(hparams, f, indent=2)

    logger.info(f"Saved best hyperparameters to {args.output_json}")
    logger.info("=== DONE LASSO HYPERPARAM TUNING (5-fold CV) ===")


if __name__ == "__main__":
    main()
