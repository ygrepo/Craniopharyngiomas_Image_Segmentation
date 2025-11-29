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
    confusion_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(
        description=(
            "Train and evaluate multinomial regularized logistic models "
            "with a given C for a given model_type (preop/postop)."
        )
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
        help=(
            "Directory containing "
            "<model_type>_train_multinomial_scaled.npz and "
            "<model_type>_test_multinomial_scaled.npz or their top-K variants."
        ),
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
        help="C value for multinomial classification "
        "(Neurosurgeon_Postop_Visual_Outcome).",
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
        help=(
            "If set, use *_multinomial_top{K}_scaled.npz instead of "
            "*_multinomial_scaled.npz."
        ),
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save CSV files. If not provided, defaults to <data_dir>.",
    )
    ap.add_argument("--log_file", type=Path, default="evaluate_multinomial_lasso.log")
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return ap.parse_args()


def load_multinomial_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load multinomial NPZ with keys: X, y_multi, feature_names.
    """
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y_multi = d["y_multi"]
    feature_names = d["feature_names"]
    return X, y_multi, feature_names


def compute_multiclass_metrics_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str = "",
    C_value: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute multiclass classification metrics (overall, per-class, confusion matrix)
    and return them as DataFrames.
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    overall_data = {
        "model_type": [model_type],
        "task": ["multinomial"],
        "target": ["Neurosurgeon_Postop_Visual_Outcome"],
        "C_value": [C_value],
        "accuracy": [accuracy],
        "precision_macro": [precision_macro],
        "precision_weighted": [precision_weighted],
        "recall_macro": [recall_macro],
        "recall_weighted": [recall_weighted],
        "f1_macro": [f1_macro],
        "f1_weighted": [f1_weighted],
    }
    overall_df = pd.DataFrame(overall_data)

    # Per-class metrics
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=unique_classes, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=unique_classes, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=unique_classes, zero_division=0
    )

    per_class_data = []
    for i, class_label in enumerate(unique_classes):
        class_str = str(class_label)
        per_class_data.append(
            {
                "model_type": model_type,
                "task": "multinomial",
                "target": "Neurosurgeon_Postop_Visual_Outcome",
                "C_value": C_value,
                "class": class_str,
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "sensitivity": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i]),
            }
        )

    per_class_df = pd.DataFrame(per_class_data)

    # Confusion matrix DataFrame
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{str(cls)}" for cls in unique_classes],
        columns=[f"Pred_{str(cls)}" for cls in unique_classes],
    )
    cm_df.insert(0, "model_type", model_type)
    cm_df.insert(1, "task", "multinomial")
    cm_df.insert(2, "C_value", C_value)

    return overall_df, per_class_df, cm_df


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN MULTINOMIAL LASSO MODEL EVALUATION ===")
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
    # Load data (multinomial NPZs)
    # ------------------------------------------------------------------
    if args.K is not None:
        train_path = (
            args.data_dir
            / f"{args.model_type}_train_multinomial_top{args.K}_scaled.npz"
        )
        test_path = (
            args.data_dir / f"{args.model_type}_test_multinomial_top{args.K}_scaled.npz"
        )
        suffix_top = f"_top{args.K}"
    else:
        train_path = args.data_dir / f"{args.model_type}_train_multinomial_scaled.npz"
        test_path = args.data_dir / f"{args.model_type}_test_multinomial_scaled.npz"
        suffix_top = ""

    logger.info(f"Loading train from {train_path}")
    logger.info(f"Loading test  from {test_path}")

    X_train, y_train_multi, _ = load_multinomial_npz(train_path)
    X_test, y_test_multi, _ = load_multinomial_npz(test_path)

    logger.info(
        f"Train class counts: {dict(zip(*np.unique(y_train_multi, return_counts=True)))}"
    )
    logger.info(
        f"Test class counts: {dict(zip(*np.unique(y_test_multi, return_counts=True)))}"
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # Build multinomial logistic model
    # ------------------------------------------------------------------
    if args.penalty == "elasticnet":
        multinomial_model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            multi_class="multinomial",
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
            C=args.C,
            l1_ratio=args.l1_ratio,
            random_state=42,
        )
    else:
        multinomial_model = LogisticRegression(
            penalty=args.penalty,  # 'l1' or 'l2'
            solver="saga",
            multi_class="multinomial",
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
            C=args.C,
            random_state=42,
        )

    logger.info("Fitting multinomial logistic model...")
    multinomial_model.fit(X_train, y_train_multi)

    # Predictions
    y_pred_multi = multinomial_model.predict(X_test)

    # Compute metrics as DataFrames
    (
        multiclass_overall_df,
        multiclass_per_class_df,
        confusion_matrix_df,
    ) = compute_multiclass_metrics_df(
        y_test_multi, y_pred_multi, args.model_type, args.C
    )

    logger.info("Multinomial Classification Results:")
    logger.info(f"  Accuracy: {multiclass_overall_df['accuracy'].iloc[0]:.3f}")
    logger.info(
        f"  Precision (macro): "
        f"{multiclass_overall_df['precision_macro'].iloc[0]:.3f}"
    )
    logger.info(
        f"  Recall (macro): {multiclass_overall_df['recall_macro'].iloc[0]:.3f}"
    )
    logger.info(f"  F1-Score (macro): {multiclass_overall_df['f1_macro'].iloc[0]:.3f}")
    logger.info(
        f"  F1-Score (weighted): " f"{multiclass_overall_df['f1_weighted'].iloc[0]:.3f}"
    )

    logger.info("Per-class metrics:")
    for _, row in multiclass_per_class_df.iterrows():
        logger.info(f"  Class {row['class']}:")
        logger.info(f"    Precision: {row['precision']:.3f}")
        logger.info(f"    Sensitivity: {row['sensitivity']:.3f}")
        logger.info(f"    F1-Score: {row['f1_score']:.3f}")

    # ------------------------------------------------------------------
    # Save DataFrames as CSV files
    # ------------------------------------------------------------------
    overall_csv_path = (
        args.output_dir
        / f"{args.model_type}_lasso_multinomial_overall_metrics{suffix_top}.csv"
    )
    multiclass_overall_df.to_csv(overall_csv_path, index=False)
    logger.info(f"Saved overall metrics to {overall_csv_path}")

    per_class_csv_path = (
        args.output_dir
        / f"{args.model_type}_lasso_multinomial_per_class_metrics{suffix_top}.csv"
    )
    multiclass_per_class_df.to_csv(per_class_csv_path, index=False)
    logger.info(f"Saved per-class metrics to {per_class_csv_path}")

    confusion_csv_path = (
        args.output_dir
        / f"{args.model_type}_lasso_multinomial_confusion_matrix{suffix_top}.csv"
    )
    confusion_matrix_df.to_csv(confusion_csv_path, index=False)
    logger.info(f"Saved confusion matrix to {confusion_csv_path}")

    # Summary CSV (flat one-row)
    summary_row = {
        "model_type": args.model_type,
        "C": args.C,
        "penalty": args.penalty,
        "l1_ratio": args.l1_ratio if args.penalty == "elasticnet" else None,
        "accuracy": multiclass_overall_df["accuracy"].iloc[0],
        "precision_macro": multiclass_overall_df["precision_macro"].iloc[0],
        "precision_weighted": multiclass_overall_df["precision_weighted"].iloc[0],
        "recall_macro": multiclass_overall_df["recall_macro"].iloc[0],
        "recall_weighted": multiclass_overall_df["recall_weighted"].iloc[0],
        "f1_macro": multiclass_overall_df["f1_macro"].iloc[0],
        "f1_weighted": multiclass_overall_df["f1_weighted"].iloc[0],
    }
    summary_df = pd.DataFrame([summary_row])
    summary_csv_path = (
        args.output_dir
        / f"{args.model_type}_lasso_multinomial_evaluation_summary{suffix_top}.csv"
    )
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved evaluation summary to {summary_csv_path}")

    logger.info("=== DONE MULTINOMIAL LASSO MODEL EVALUATION ===")

    # Return DataFrames for programmatic use
    return {
        "overall_metrics": multiclass_overall_df,
        "per_class_metrics": multiclass_per_class_df,
        "confusion_matrix": confusion_matrix_df,
        "summary": summary_df,
    }


if __name__ == "__main__":
    main()
