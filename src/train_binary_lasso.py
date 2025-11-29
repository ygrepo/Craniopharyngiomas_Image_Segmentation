#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(
        description=(
            "Train and evaluate regularized logistic models with given C values "
            "for a given model_type (preop/postop)."
        )
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
        help="Directory containing *_train_binary_scaled.npz and *_test_binary_scaled.npz.",
    )
    ap.add_argument(
        "--model_type",
        type=str,
        choices=["preop", "postop"],
        required=True,
        help="Which model design to evaluate: 'preop' or 'postop'.",
    )
    ap.add_argument(
        "--C",
        type=float,
        required=True,
        help="C value for binary classification (Outcome_Improved).",
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
        "--K",
        type=int,
        default=None,
        help="If set, use *_binary_top{K}_scaled.npz instead of *_binary_scaled.npz.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save CSV files. If not provided, defaults to <data_dir>.",
    )
    ap.add_argument("--log_file", type=Path, default="evaluate_lasso.log")
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return ap.parse_args()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load binary NPZ with keys: X, y_bin, feature_names.
    """
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y_bin = d["y_bin"]
    feature_names = d["feature_names"]
    return X, y_bin, feature_names


def compute_binary_metrics_df(
    y_true, y_pred, y_prob=None, model_type="", C_value=None
) -> pd.DataFrame:
    """Compute binary classification metrics and return as a single-row DataFrame."""

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = recall  # Same as recall
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix for specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # All samples in one class – define everything safely
        tn = fp = fn = tp = 0
        if np.unique(y_true)[0] == 1:
            # All positives
            tp = cm[0, 0]
        else:
            # All negatives
            tn = cm[0, 0]

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC if probabilities are provided
    auc_roc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_prob)

    metrics_data = {
        "model_type": [model_type],
        "task": ["binary"],
        "target": ["Outcome_Improved"],
        "C_value": [C_value],
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "sensitivity": [sensitivity],
        "specificity": [specificity],
        "f1_score": [f1],
        "auc_roc": [auc_roc],
        "true_positives": [int(tp)],
        "true_negatives": [int(tn)],
        "false_positives": [int(fp)],
        "false_negatives": [int(fn)],
    }

    return pd.DataFrame(metrics_data)


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN LASSO MODEL EVALUATION (BINARY) ===")
    logger.info(f"model_type = {args.model_type}")
    logger.info(f"C = {args.C}")
    logger.info(f"K = {args.K}")
    logger.info(f"penalty = {args.penalty}")
    logger.info(f"l1_ratio = {args.l1_ratio}")
    logger.info(f"output_dir = {args.output_dir}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.data_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data (binary NPZs)
    # Naming assumed consistent with your CV script:
    #   <model_type>_train_binary_scaled.npz
    #   <model_type>_test_binary_scaled.npz
    #   or with top-K:
    #   <model_type>_train_binary_top{K}_scaled.npz, etc.
    # ------------------------------------------------------------------
    if args.K is not None:
        train_path = (
            args.data_dir / f"{args.model_type}_train_binary_top{args.K}_scaled.npz"
        )
        test_path = (
            args.data_dir / f"{args.model_type}_test_binary_top{args.K}_scaled.npz"
        )
    else:
        train_path = args.data_dir / f"{args.model_type}_train_binary_scaled.npz"
        test_path = args.data_dir / f"{args.model_type}_test_binary_scaled.npz"

    logger.info(f"Loading train from {train_path}")
    logger.info(f"Loading test  from {test_path}")

    X_train, y_train_bin, feat_names = load_npz(train_path)
    X_test, y_test_bin, _ = load_npz(test_path)

    logger.info(f"Train outcome counts: {np.bincount(y_train_bin)}")
    logger.info(f"Test outcome counts: {np.bincount(y_test_bin)}")
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if args.penalty == "elasticnet":
        binary_model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
            C=args.C,
            l1_ratio=args.l1_ratio,
            random_state=42,
        )
    else:
        # 'l1' or 'l2' – l1_ratio must NOT be passed here
        binary_model = LogisticRegression(
            penalty=args.penalty,
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
            C=args.C,
            random_state=42,
        )

    logger.info("Fitting binary logistic model...")
    binary_model.fit(X_train, y_train_bin)

    # Predictions
    y_pred_bin = binary_model.predict(X_test)
    y_prob_bin = (
        binary_model.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test_bin)) > 1
        else None
    )

    # Compute metrics as DataFrame
    binary_df = compute_binary_metrics_df(
        y_test_bin, y_pred_bin, y_prob_bin, args.model_type, args.C
    )

    logger.info("Binary Classification Results:")
    logger.info(f"  Accuracy: {binary_df['accuracy'].iloc[0]:.3f}")
    logger.info(f"  Sensitivity (Recall): {binary_df['sensitivity'].iloc[0]:.3f}")
    logger.info(f"  Specificity: {binary_df['specificity'].iloc[0]:.3f}")
    logger.info(f"  Precision: {binary_df['precision'].iloc[0]:.3f}")
    logger.info(f"  F1-Score: {binary_df['f1_score'].iloc[0]:.3f}")
    if binary_df["auc_roc"].iloc[0] is not None:
        logger.info(f"  AUC-ROC: {binary_df['auc_roc'].iloc[0]:.3f}")

    # ------------------------------------------------------------------
    # Save metrics as CSVs
    # ------------------------------------------------------------------
    if args.K is not None:
        suffix_top = f"_top{args.K}"
    else:
        suffix_top = ""

    # 1. Overall metrics (here just the binary metrics)
    overall_csv_path = (
        args.output_dir / f"{args.model_type}_lasso_overall_metrics{suffix_top}.csv"
    )
    binary_df.to_csv(overall_csv_path, index=False)
    logger.info(f"Saved overall metrics to {overall_csv_path}")

    # 2. Comprehensive metrics (same as overall for now, but separate file)
    comprehensive_csv_path = (
        args.output_dir / f"{args.model_type}_lasso_all_metrics{suffix_top}.csv"
    )
    binary_df.to_csv(comprehensive_csv_path, index=False)
    logger.info(f"Saved comprehensive metrics to {comprehensive_csv_path}")

    # 3. Summary CSV (flat, one row)
    summary_row = {
        "model_type": args.model_type,
        "C": args.C,
        "penalty": args.penalty,
        "l1_ratio": args.l1_ratio if args.penalty == "elasticnet" else None,
        "accuracy": binary_df["accuracy"].iloc[0],
        "sensitivity": binary_df["sensitivity"].iloc[0],
        "specificity": binary_df["specificity"].iloc[0],
        "precision": binary_df["precision"].iloc[0],
        "f1_score": binary_df["f1_score"].iloc[0],
        "auc_roc": binary_df["auc_roc"].iloc[0],
    }
    summary_df = pd.DataFrame([summary_row])
    summary_csv_path = (
        args.output_dir / f"{args.model_type}_lasso_evaluation_summary_{suffix_top}.csv"
    )
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved evaluation summary to {summary_csv_path}")

    logger.info("=== DONE LASSO MODEL EVALUATION (BINARY) ===")

    # Return DataFrames for programmatic use (if imported as a module)
    return {
        "overall_metrics": binary_df,
        "comprehensive_metrics": binary_df,
        "summary": summary_df,
    }


if __name__ == "__main__":
    main()
