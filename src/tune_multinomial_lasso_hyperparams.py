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
            "Grid search multinomial LASSO hyperparameters (C) using 5-fold "
            "stratified CV on the training set for a given model_type (preop/postop)."
        )
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
        help=(
            "Directory containing <model_type>_train_multinomial_scaled.npz "
            "(output of create_multinomial_lasso_features.py)."
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
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Path to save aggregated CV metrics for plotting. "
            "If not provided, defaults to "
            "<data_dir>/<model_type>_multinomial_lasso_cv_metrics.csv"
        ),
    )
    ap.add_argument(
        "--log_file", type=Path, default="tune_multinomial_lasso_hyperparams.log"
    )
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
    y_multi = d["y_multi"]
    feature_names = d["feature_names"]
    return X, y_multi, feature_names


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN MULTINOMIAL LASSO HYPERPARAM TUNING (5-fold CV) ===")
    logger.info(f"model_type = {args.model_type}")

    # Expecting file like preop_train_multinomial_scaled.npz
    train_path = args.data_dir / f"{args.model_type}_train_multinomial_scaled.npz"

    X_train, y_train, feat_names = load_npz(train_path)
    logger.info(f"Train shape: {X_train.shape}")
    classes, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Train class counts: {dict(zip(classes, counts))}")

    # Same broad grid you used before
    C_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info(
        "Tuning multinomial LASSO "
        "(Neurosurgeon_Postop_Visual_Outcome) with 5-fold stratified CV..."
    )

    best_C = None
    best_mean_f1 = -np.inf
    best_mean_auc = -np.inf

    results = []

    for C in C_grid:
        logger.info(f"Evaluating C = {C} (multinomial) with 5-fold CV")
        fold_aucs = []
        fold_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train), start=1
        ):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                n_jobs=-1,
                C=C,
                # multi_class="multinomial"  # "auto" will usually pick multinomial for multi-class
            )
            model.fit(X_tr, y_tr)

            # Multiclass macro-ROC AUC (one-vs-rest)
            if len(np.unique(y_val)) > 1:
                try:
                    y_prob = model.predict_proba(X_val)
                    auc = roc_auc_score(
                        y_val, y_prob, multi_class="ovr", average="macro"
                    )
                except ValueError as e:
                    logger.warning(
                        f"Fold {fold_idx}: could not compute multiclass AUC; "
                        f"setting AUC to NaN. Error: {e}"
                    )
                    auc = np.nan
            else:
                logger.warning(
                    f"Fold {fold_idx}: val set has a single class; AUC set to NaN."
                )
                auc = np.nan

            # Macro-F1
            y_pred = model.predict(X_val)
            try:
                f1 = f1_score(y_val, y_pred, average="macro")
            except ValueError as e:
                logger.warning(
                    f"Fold {fold_idx}: could not compute macro-F1; "
                    f"setting F1 to NaN. Error: {e}"
                )
                f1 = np.nan

            fold_aucs.append(auc)
            fold_f1s.append(f1)

            logger.info(
                f"  Fold {fold_idx}: "
                f"AUC_macro = {auc if not np.isnan(auc) else 'NaN'}, "
                f"F1_macro = {f1 if not np.isnan(f1) else 'NaN'}"
            )

        mean_auc = float(np.nanmean(fold_aucs))
        std_auc = float(np.nanstd(fold_aucs))
        mean_f1 = float(np.nanmean(fold_f1s))
        std_f1 = float(np.nanstd(fold_f1s))

        logger.info(
            f"C = {C}: "
            f"mean AUC_macro = {mean_auc:.3f} (std {std_auc:.3f}), "
            f"mean F1_macro = {mean_f1:.3f} (std {std_f1:.3f})"
        )

        results.append(
            {
                "C": C,
                "mean_auc_macro": mean_auc,
                "std_auc_macro": std_auc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro": std_f1,
            }
        )

        # Select C by mean macro-F1; use AUC as tie-breaker if F1 is equal
        is_better = False
        if mean_f1 > best_mean_f1:
            is_better = True
        elif np.isclose(mean_f1, best_mean_f1) and mean_auc > best_mean_auc:
            is_better = True

        if is_better:
            best_mean_f1 = mean_f1
            best_mean_auc = mean_auc
            best_C = C
            logger.info(
                f"  [!] New best C (multinomial) = {best_C}, "
                f"mean AUC_macro = {best_mean_auc:.3f}, "
                f"mean F1_macro = {best_mean_f1:.3f}"
            )

    logger.info(
        f"Best C (multinomial) = {best_C}, "
        f"CV mean AUC_macro = {best_mean_auc:.3f}, "
        f"CV mean F1_macro = {best_mean_f1:.3f}"
    )

    # Mark best row
    for row in results:
        row["is_best"] = row["C"] == best_C

    # Save aggregated metrics to CSV for plotting
    df = pd.DataFrame(results).sort_values("C").reset_index(drop=True)

    if args.output_csv is None:
        args.output_csv = (
            args.data_dir / f"{args.model_type}_multinomial_lasso_cv_metrics.csv"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved CV metrics CSV to {args.output_csv}")

    logger.info("=== DONE MULTINOMIAL LASSO HYPERPARAM TUNING (5-fold CV) ===")


if __name__ == "__main__":
    main()
