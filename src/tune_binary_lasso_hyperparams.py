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
            "Grid search regularized logistic regression hyperparameters (C) using "
            "5-fold stratified CV on the training set for a given model_type "
            "(preop/postop). Supports L1 (LASSO), L2 (ridge), and ElasticNet penalties."
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
        "--C",
        type=float,
        default=None,
        help=(
            "Single C value to try. If provided, overrides internal default C grid. "
            "If --C_grid is also provided, --C_grid takes precedence."
        ),
    )
    ap.add_argument(
        "--C_grid",
        type=str,
        default=None,
        help=(
            "Comma-separated list of C values to try, e.g. '0.01,0.1,1.0'. "
            "If provided, overrides --C and the internal default grid."
        ),
    )
    ap.add_argument(
        "--l1_ratio",
        type=float,
        default=0.5,
        help=(
            "ElasticNet mixing parameter (0.0 = pure L2, 1.0 = pure L1). "
            "Used only if --penalty elasticnet and --l1_ratio_grid is not set. "
            "Default: 0.5."
        ),
    )
    ap.add_argument(
        "--l1_ratio_grid",
        type=str,
        default=None,
        help=(
            "Comma-separated list of ElasticNet l1_ratio values to try, e.g. "
            "'0.1,0.5,0.9'. Used only if --penalty elasticnet. "
            "If provided, overrides --l1_ratio."
        ),
    )
    ap.add_argument(
        "--K",
        type=int,
        default=None,
        help="Number of top features to load from the NPZ file.",
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Path to save aggregated CV metrics. "
            "If None, defaults to "
            "<data_dir>/<model_type>_binary_<penalty>_cv_metrics.csv "
            "(for L1, keeps the legacy name *_binary_l1_lasso_cv_metrics.csv)."
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


def normal_p_value(mean: float, target: float, std: float, n: int) -> float:
    """
    Two-sided p-value under a normal approximation for H0: mean = target,
    using sample mean/std over n observations (here: folds).
    """
    if n <= 1 or np.isnan(std) or std == 0.0:
        return np.nan
    se = std / math.sqrt(n)
    z = (mean - target) / se
    # Phi(z) via erf
    phi = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    return p


def normal_ci(
    mean: float, std: float, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """
    (1 - alpha) CI under normal approximation based on sample mean/std and n observations.
    Currently implements 95% CI (alpha=0.05) using z=1.96.
    """
    if n <= 1 or np.isnan(std) or std == 0.0:
        return (np.nan, np.nan)
    se = std / math.sqrt(n)
    # 1.96 is z_{0.975} for 95% CI
    z = 1.96
    lower = mean - z * se
    upper = mean + z * se
    return lower, upper


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN REGULARIZED LOGISTIC HYPERPARAM TUNING (5-fold CV) ===")
    logger.info(f"model_type    = {args.model_type}")
    logger.info(f"penalty       = {args.penalty}")
    logger.info(f"C             = {args.C}")
    logger.info(f"C_grid        = {args.C_grid}")
    logger.info(f"l1_ratio      = {args.l1_ratio}")
    logger.info(f"l1_ratio_grid = {args.l1_ratio_grid}")
    logger.info(f"K             = {args.K}")
    logger.info(f"output_csv    = {args.output_csv}")

    if args.penalty == "elasticnet":
        logger.info("ElasticNet selected; l1_ratio / l1_ratio_grid will be used.")
    if args.K is not None:
        logger.info(f"Loading top-{args.K} features from NPZ file.")
    else:
        logger.info("Loading all features from NPZ file.")

    # Expecting file like preop_train_binary_top40_scaled.npz
    if args.K is not None:
        args.train_suffix = f"_top{args.K}_scaled.npz"
    else:
        args.train_suffix = "_scaled.npz"
    train_path = args.data_dir / f"{args.model_type}_train_binary{args.train_suffix}"
    logger.info(f"Loading training data from {train_path}")

    X_train, y_train, feat_names = load_npz(train_path)

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(
        f"Train class counts: {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    # -------------------------------------------------------------------------
    # C grid logic: like previous script
    # -------------------------------------------------------------------------
    default_C_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]

    if args.C_grid is not None:
        C_grid = [float(x) for x in args.C_grid.split(",")]
        logger.info(f"Using C_grid from CLI: {C_grid}")
    elif args.C is not None:
        C_grid = [args.C]
        logger.info(f"Using single C from CLI: {C_grid}")
    else:
        C_grid = default_C_grid
        logger.info(f"Using default C_grid: {C_grid}")

    # -------------------------------------------------------------------------
    # l1_ratio grid logic (only relevant for ElasticNet)
    # -------------------------------------------------------------------------
    if args.penalty == "elasticnet":
        if args.l1_ratio_grid is not None:
            l1_ratio_grid = [float(x) for x in args.l1_ratio_grid.split(",")]
        else:
            l1_ratio_grid = [args.l1_ratio]
        logger.info(f"ElasticNet l1_ratio grid: {l1_ratio_grid}")
    else:
        l1_ratio_grid = [np.nan]  # placeholder for non-elasticnet
        logger.info("Penalty is not ElasticNet; l1_ratio grid is not used.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_C = None
    best_l1_ratio = None
    best_mean_auc = -np.inf

    results = []

    logger.info("Starting 5-fold CV...")

    for C in C_grid:
        for l1_ratio in l1_ratio_grid:
            if args.penalty == "elasticnet":
                logger.info(f"Evaluating C = {C}, l1_ratio = {l1_ratio}")
            else:
                logger.info(f"Evaluating C = {C}")

            fold_aucs = []
            fold_f1s = []
            fold_f1_baseline = []
            # per-fold sensitivity and specificity (positive class = 1)
            fold_sensitivities = []
            fold_specificities = []

            for fold_idx, (tr_idx, val_idx) in enumerate(
                skf.split(X_train, y_train), 1
            ):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                # Construct model according to penalty
                if args.penalty == "elasticnet":
                    model = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        class_weight="balanced",
                        C=C,
                        l1_ratio=l1_ratio,
                        max_iter=5000,
                        n_jobs=-1,
                    )
                else:
                    # 'l1' or 'l2'
                    model = LogisticRegression(
                        penalty=args.penalty,
                        solver="saga",  # saga supports l1, l2
                        class_weight="balanced",
                        C=C,
                        max_iter=5000,
                        n_jobs=-1,
                    )

                model.fit(X_tr, y_tr)

                # AUC for this fold
                if len(np.unique(y_val)) > 1:
                    try:
                        y_prob = model.predict_proba(X_val)[:, 1]
                        auc = roc_auc_score(y_val, y_prob)
                    except ValueError:
                        logger.warning(
                            f"Fold {fold_idx}: could not compute AUC, setting NaN."
                        )
                        auc = np.nan
                else:
                    auc = np.nan

                # F1 for this fold
                try:
                    y_pred = model.predict(X_val)
                    f1 = f1_score(y_val, y_pred, average="binary")
                except ValueError:
                    logger.warning(
                        f"Fold {fold_idx}: could not compute F1, setting NaN."
                    )
                    f1 = np.nan

                # Sensitivity and specificity (positive class = 1)
                if len(np.unique(y_val)) > 1:
                    tp = np.sum((y_val == 1) & (y_pred == 1))
                    fn = np.sum((y_val == 1) & (y_pred == 0))
                    tn = np.sum((y_val == 0) & (y_pred == 0))
                    fp = np.sum((y_val == 0) & (y_pred == 1))

                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                else:
                    sensitivity = np.nan
                    specificity = np.nan

                fold_sensitivities.append(sensitivity)
                fold_specificities.append(specificity)

                # Baseline F1 for majority-class classifier in this fold
                majority_class = 1 if np.sum(y_val) >= len(y_val) / 2 else 0
                y_pred_maj = np.full_like(y_val, majority_class)
                try:
                    f1_base = f1_score(y_val, y_pred_maj, average="binary")
                except ValueError:
                    logger.warning(
                        f"Fold {fold_idx}: could not compute F1_baseline, setting NaN."
                    )
                    f1_base = np.nan

                fold_aucs.append(auc)
                fold_f1s.append(f1)
                fold_f1_baseline.append(f1_base)

                logger.debug(
                    f"    Fold {fold_idx}: "
                    f"AUC={auc}, F1={f1}, F1_baseline={f1_base} "
                    f"(maj_class={majority_class}), "
                    f"Sensitivity={sensitivity}, Specificity={specificity}"
                )

            # Aggregate across folds
            mean_auc = float(np.nanmean(fold_aucs))
            std_auc = float(np.nanstd(fold_aucs))
            mean_f1 = float(np.nanmean(fold_f1s))
            std_f1 = float(np.nanstd(fold_f1s))
            mean_f1_base = float(np.nanmean(fold_f1_baseline))
            std_f1_base = float(np.nanstd(fold_f1_baseline))

            # Sensitivity/specificity aggregates
            mean_sens = float(np.nanmean(fold_sensitivities))
            std_sens = float(np.nanstd(fold_sensitivities))
            mean_spec = float(np.nanmean(fold_specificities))
            std_spec = float(np.nanstd(fold_specificities))

            # 95% CI for AUC across folds
            ci_auc_lower, ci_auc_upper = normal_ci(mean_auc, std_auc, n=len(fold_aucs))

            # Approximate p-values:
            #   - AUC vs 0.5 (random)
            p_auc = normal_p_value(mean_auc, 0.5, std_auc, n=len(fold_aucs))

            #   - F1 improvement vs majority-class baseline (per-fold improvement)
            f1_diffs = np.array(fold_f1s) - np.array(fold_f1_baseline)
            mean_f1_diff = float(np.nanmean(f1_diffs))
            std_f1_diff = float(np.nanstd(f1_diffs))
            p_f1_vs_base = normal_p_value(
                mean_f1_diff, 0.0, std_f1_diff, n=len(f1_diffs)
            )

            logger.info(
                f"C={C}, "
                f"l1_ratio={l1_ratio if args.penalty == 'elasticnet' else 'N/A'}: "
                f"mean AUC={mean_auc:.3f} (std={std_auc:.3f}, "
                f"95% CI=[{ci_auc_lower:.3f}, {ci_auc_upper:.3f}]), "
                f"mean F1={mean_f1:.3f} (std={std_f1:.3f}), "
                f"baseline F1={mean_f1_base:.3f}, "
                f"p_auc_vs_0.5={p_auc:.3g}, p_f1_vs_baseline={p_f1_vs_base:.3g}, "
                f"mean Sensitivity={mean_sens:.3f} (std={std_sens:.3f}), "
                f"mean Specificity={mean_spec:.3f} (std={std_spec:.3f})"
            )

            results.append(
                {
                    "C": C,
                    "penalty": args.penalty,
                    "l1_ratio": (
                        float(l1_ratio) if args.penalty == "elasticnet" else np.nan
                    ),
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                    "ci_auc_lower_95": ci_auc_lower,
                    "ci_auc_upper_95": ci_auc_upper,
                    "p_auc_vs_0_5": p_auc,
                    "mean_f1": mean_f1,
                    "std_f1": std_f1,
                    "mean_f1_baseline": mean_f1_base,
                    "std_f1_baseline": std_f1_base,
                    "mean_f1_diff_vs_baseline": mean_f1_diff,
                    "std_f1_diff_vs_baseline": std_f1_diff,
                    "p_f1_vs_baseline": p_f1_vs_base,
                    "mean_sensitivity": mean_sens,
                    "std_sensitivity": std_sens,
                    "mean_specificity": mean_spec,
                    "std_specificity": std_spec,
                }
            )

            # Track best by mean AUC (you can change criterion if needed)
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_C = C
                best_l1_ratio = (
                    float(l1_ratio) if args.penalty == "elasticnet" else None
                )
                logger.info(
                    f"  [!] New best: C = {best_C}, "
                    f"l1_ratio = {best_l1_ratio if best_l1_ratio is not None else 'N/A'} "
                    f"(mean AUC={best_mean_auc:.3f})"
                )

    logger.info(
        f"Best C = {best_C}, "
        f"best l1_ratio = {best_l1_ratio if best_l1_ratio is not None else 'N/A'} "
        f"(mean AUC={best_mean_auc:.3f})"
    )

    # Build DataFrame
    df = pd.DataFrame(results).sort_values(["C", "l1_ratio"]).reset_index(drop=True)

    # Mark best row(s) robustly
    if args.penalty == "elasticnet":
        if best_l1_ratio is not None:
            df["is_best"] = (df["C"] == best_C) & np.isclose(
                df["l1_ratio"].astype(float), float(best_l1_ratio)
            )
        else:
            df["is_best"] = False
    else:
        df["is_best"] = df["C"] == best_C

    # Save CSV (penalty-specific default; keep legacy name for L1)
    if args.output_csv is None:
        if args.penalty == "l1":
            if args.K is None:
                suffix = "binary_l1_lasso_cv_metrics.csv"
            else:
                suffix = f"binary_l1_lasso_top{args.K}_cv_metrics.csv"
        elif args.penalty == "l2":
            if args.K is None:
                suffix = "binary_l2_ridge_cv_metrics.csv"
            else:
                suffix = f"binary_l2_ridge_top{args.K}_cv_metrics.csv"
        else:  # elasticnet
            if len(l1_ratio_grid) == 1:
                if args.K is None:
                    suffix = (
                        f"binary_elasticnet_l1ratio_{l1_ratio_grid[0]:g}_cv_metrics.csv"
                    )
                else:
                    suffix = f"binary_elasticnet_l1ratio_{l1_ratio_grid[0]:g}_top{args.K}_cv_metrics.csv"
            else:
                if args.K is None:
                    suffix = "binary_elasticnet_l1ratio_grid_cv_metrics.csv"
                else:
                    suffix = (
                        f"binary_elasticnet_l1ratio_grid_top{args.K}_cv_metrics.csv"
                    )

        args.output_csv = args.data_dir / f"{args.model_type}_{suffix}"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    logger.info(
        f"Saved CV metrics (with p-values, AUC CIs, sens/spec, grid over C/l1_ratio) "
        f"to {args.output_csv}"
    )
    logger.info("=== DONE REGULARIZED LOGISTIC HYPERPARAM TUNING (5-fold CV) ===")


if __name__ == "__main__":
    main()
