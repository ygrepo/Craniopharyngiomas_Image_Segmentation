#!/usr/bin/env python3
import argparse
import sys
import math
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
            "Grid search multinomial regularized logistic regression hyperparameters "
            "(C) using 5-fold stratified CV on the training set for a given "
            "model_type (preop/postop). Supports L1 (LASSO), L2 (ridge), and "
            "ElasticNet penalties."
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
        "--penalty",
        "--loss",
        dest="penalty",
        type=str,
        default="l1",
        choices=["l1", "l2", "elasticnet"],
        help=(
            "Regularization penalty: 'l1' (LASSO), 'l2' (ridge), or 'elasticnet'. "
            "Default: 'l1'."
        ),
    )
    ap.add_argument(
        "--l1_ratio",
        type=float,
        default=0.5,
        help=(
            "ElasticNet mixing parameter (0.0 = pure L2, 1.0 = pure L1). "
            "Used only if --penalty elasticnet. Default: 0.5."
        ),
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Path to save aggregated CV metrics for plotting. "
            "If not provided, defaults to "
            "<data_dir>/<model_type>_multinomial_<penalty>_cv_metrics.csv "
            "(for L1, keeps the legacy name "
            "'*_multinomial_l1_lasso_cv_metrics.csv')."
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


def normal_p_value(mean: float, target: float, std: float, n: int) -> float:
    """
    Two-sided p-value under a normal approximation for H0: mean = target,
    using sample mean/std over n observations (here: folds).
    """
    if n <= 1 or np.isnan(std) or std == 0.0:
        return np.nan
    se = std / math.sqrt(n)
    z = (mean - target) / se
    # standard normal CDF via erf
    phi = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    return p


def normal_ci(mean: float, std: float, n: int, alpha: float = 0.05):
    """
    (1 - alpha) CI under normal approximation based on sample mean/std and n observations.
    Default: alpha=0.05 -> 95% CI.
    """
    if n <= 1 or np.isnan(std) or std == 0.0:
        return (np.nan, np.nan)
    se = std / math.sqrt(n)
    # z for 95% CI
    z = 1.96
    lower = mean - z * se
    upper = mean + z * se
    return lower, upper


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info(
        "=== BEGIN MULTINOMIAL REGULARIZED LOGISTIC HYPERPARAM TUNING (5-fold CV) ==="
    )
    logger.info(f"model_type = {args.model_type}")
    logger.info(f"penalty    = {args.penalty}")
    if args.penalty == "elasticnet":
        logger.info(f"l1_ratio   = {args.l1_ratio}")

    # Expecting file like preop_train_multinomial_scaled.npz
    train_path = args.data_dir / f"{args.model_type}_train_multinomial_top40_scaled.npz"
    # train_path = args.data_dir / f"{args.model_type}_train_multinomial_scaled.npz"

    X_train, y_train, feat_names = load_npz(train_path)
    logger.info(f"Train shape: {X_train.shape}")
    classes, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Train class counts: {dict(zip(classes, counts))}")

    C_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info(
        "Tuning multinomial regularized logistic regression "
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
        fold_f1_baseline = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train), start=1
        ):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Build model depending on penalty
            if args.penalty == "elasticnet":
                model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=5000,
                    class_weight="balanced",
                    n_jobs=-1,
                    C=C,
                    l1_ratio=args.l1_ratio,
                )
            else:
                model = LogisticRegression(
                    penalty=args.penalty,  # "l1" or "l2"
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=5000,
                    class_weight="balanced",
                    n_jobs=-1,
                    C=C,
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

            # Macro-F1 for the model
            y_pred = model.predict(X_val)
            try:
                f1 = f1_score(y_val, y_pred, average="macro")
            except ValueError as e:
                logger.warning(
                    f"Fold {fold_idx}: could not compute macro-F1; "
                    f"setting F1 to NaN. Error: {e}"
                )
                f1 = np.nan

            # Macro-F1 for majority-class baseline in this fold
            vals, val_counts = np.unique(y_val, return_counts=True)
            maj_class = vals[np.argmax(val_counts)]
            y_pred_maj = np.full_like(y_val, maj_class)
            try:
                f1_base = f1_score(y_val, y_pred_maj, average="macro")
            except ValueError:
                logger.warning(
                    f"Fold {fold_idx}: could not compute F1_baseline; "
                    f"setting F1_baseline to NaN."
                )
                f1_base = np.nan

            fold_aucs.append(auc)
            fold_f1s.append(f1)
            fold_f1_baseline.append(f1_base)

            logger.info(
                f"  Fold {fold_idx}: "
                f"AUC_macro = {auc if not np.isnan(auc) else 'NaN'}, "
                f"F1_macro = {f1 if not np.isnan(f1) else 'NaN'}, "
                f"F1_macro_baseline = {f1_base if not np.isnan(f1_base) else 'NaN'} "
                f"(maj_class={maj_class})"
            )

        # Aggregate across folds
        mean_auc = float(np.nanmean(fold_aucs))
        std_auc = float(np.nanstd(fold_aucs))
        mean_f1 = float(np.nanmean(fold_f1s))
        std_f1 = float(np.nanstd(fold_f1s))
        mean_f1_base = float(np.nanmean(fold_f1_baseline))
        std_f1_base = float(np.nanstd(fold_f1_baseline))

        # 95% CI for macro-AUC across folds
        ci_auc_lower, ci_auc_upper = normal_ci(mean_auc, std_auc, n=len(fold_aucs))

        # Approximate p-values:
        #   - macro-AUC vs 0.5 (random-like)
        p_auc = normal_p_value(mean_auc, 0.5, std_auc, n=len(fold_aucs))

        #   - macro-F1 improvement vs majority-class baseline
        f1_diffs = np.array(fold_f1s) - np.array(fold_f1_baseline)
        mean_f1_diff = float(np.nanmean(f1_diffs))
        std_f1_diff = float(np.nanstd(f1_diffs))
        p_f1_vs_base = normal_p_value(mean_f1_diff, 0.0, std_f1_diff, n=len(f1_diffs))

        logger.info(
            f"C = {C}: "
            f"mean AUC_macro = {mean_auc:.3f} "
            f"(std {std_auc:.3f}, 95% CI=[{ci_auc_lower:.3f}, {ci_auc_upper:.3f}]), "
            f"mean F1_macro = {mean_f1:.3f} (std {std_f1:.3f}), "
            f"baseline F1_macro = {mean_f1_base:.3f}, "
            f"p_auc_macro_vs_0.5 = {p_auc:.3g}, "
            f"p_f1_macro_vs_baseline = {p_f1_vs_base:.3g}"
        )

        results.append(
            {
                "C": C,
                "penalty": args.penalty,
                "l1_ratio": args.l1_ratio if args.penalty == "elasticnet" else np.nan,
                "mean_auc_macro": mean_auc,
                "std_auc_macro": std_auc,
                "ci_auc_macro_lower_95": ci_auc_lower,
                "ci_auc_macro_upper_95": ci_auc_upper,
                "p_auc_macro_vs_0_5": p_auc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro": std_f1,
                "mean_f1_macro_baseline": mean_f1_base,
                "std_f1_macro_baseline": std_f1_base,
                "mean_f1_macro_diff_vs_baseline": mean_f1_diff,
                "std_f1_macro_diff_vs_baseline": std_f1_diff,
                "p_f1_macro_vs_baseline": p_f1_vs_base,
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
        if args.penalty == "l1":
            suffix = "multinomial_l1_lasso_top40_cv_metrics.csv"
        elif args.penalty == "l2":
            suffix = "multinomial_l2_ridge_top40_cv_metrics.csv"
        else:  # elasticnet
            suffix = (
                f"multinomial_elasticnet_l1ratio_{args.l1_ratio:g}_top40_cv_metrics.csv"
            )

        args.output_csv = args.data_dir / f"{args.model_type}_{suffix}"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved CV metrics CSV to {args.output_csv}")

    logger.info(
        "=== DONE MULTINOMIAL REGULARIZED LOGISTIC HYPERPARAM TUNING (5-fold CV) ==="
    )


if __name__ == "__main__":
    main()
