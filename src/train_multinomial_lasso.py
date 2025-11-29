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
            "Train and evaluate LASSO models with given C values "
            "for a given model_type (preop/postop)."
        )
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
        help="Directory containing *_train_scaled.npz and *_test_scaled.npz.",
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
            "Used only if --penalty elasticnet and --l1_ratio_grid is not set. "
            "Default: 0.5."
        ),
    )
    ap.add_argument(
        "--K",
        type=int,
        default=None,
        help="Number of top features to load from the binary NPZ file.",
    )
    ap.add_argument(
        "--Multinomial_K",
        type=int,
        default=None,
        help="Number of top features to load from the multinomial NPZ file.",
    )
    ap.add_argument(
        "--hyperparams_json",
        type=Path,
        default=None,
        help="Optional: Load C values from hyperparameters JSON file. "
        "If provided, overrides --C_binary and --C_multinomial.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save CSV files. If not provided, defaults to <data_dir>",
    )
    ap.add_argument("--log_file", type=Path, default="evaluate_lasso.log")
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return ap.parse_args()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y_multi = d["y_multi"]
    feature_names = d["feature_names"]
    return X, y_multi, feature_names


def compute_binary_metrics_df(
    y_true, y_pred, y_prob=None, model_type="", C_value=None
) -> pd.DataFrame:
    """Compute comprehensive binary classification metrics and return as DataFrame."""

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

    # Create DataFrame
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


def compute_multiclass_metrics_df(
    y_true: np.ndarray, y_pred: np.ndarray, model_type="", C_value=None
):
    """Compute comprehensive multiclass classification metrics and return as DataFrames."""

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

    # Overall metrics DataFrame
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
    # Add metadata columns
    cm_df.insert(0, "model_type", model_type)
    cm_df.insert(1, "task", "multinomial")
    cm_df.insert(2, "C_value", C_value)

    return overall_df, per_class_df, cm_df


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN LASSO MODEL EVALUATION ===")
    logger.info(f"model_type = {args.model_type}")
    logger.info(f"C_binary = {args.C_binary}, C_multinomial = {args.C_multinomial}")
    logger.info(f"Binary_K = {args.Binary_K}, Multinomial_K = {args.Multinomial_K}")

    # Load C values from hyperparameters JSON if provided
    if args.hyperparams_json is not None:
        logger.info(f"Loading hyperparameters from {args.hyperparams_json}")
        import json  # local import since only used here

        with open(args.hyperparams_json, "r") as f:
            hparams = json.load(f)
        C_binary = hparams["binary"]["C"]
        C_multinomial = hparams["multinomial"]["C"]
    else:
        C_binary = args.C_binary
        C_multinomial = args.C_multinomial

    logger.info(f"C_binary = {C_binary}, C_multinomial = {C_multinomial}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.data_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.Binary_K is not None and args.Multinomial_K is not None:
        train_path = (
            args.data_dir
            / f"{args.model_type}_train_top_{args.Binary_K}_top_{args.Multinomial_K}_scaled.npz"
        )
        test_path = (
            args.data_dir
            / f"{args.model_type}_test_top_{args.Binary_K}_top_{args.Multinomial_K}_scaled.npz"
        )
    else:
        train_path = args.data_dir / f"{args.model_type}_train_scaled.npz"
        test_path = args.data_dir / f"{args.model_type}_test_scaled.npz"

    X_train, y_train_bin, y_train_multi, feat_names = load_npz(train_path)
    X_test, y_test_bin, y_test_multi, _ = load_npz(test_path)
    logger.info(f"Train outcome counts:{np.bincount(y_train_bin)}")
    logger.info(f"Test outcome counts:{np.bincount(y_test_bin)}")

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Initialize list to store all DataFrames for overall metrics
    all_dfs = []

    # ---------------- Binary Classification ----------------
    logger.info("Training and evaluating binary LASSO (Outcome_Improved)...")

    binary_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        C=C_binary,
        random_state=42,
    )

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
        y_test_bin, y_pred_bin, y_prob_bin, args.model_type, C_binary
    )
    all_dfs.append(binary_df)

    logger.info("Binary Classification Results:")
    logger.info(f"  Accuracy: {binary_df['accuracy'].iloc[0]:.3f}")
    logger.info(f"  Sensitivity (Recall): {binary_df['sensitivity'].iloc[0]:.3f}")
    logger.info(f"  Specificity: {binary_df['specificity'].iloc[0]:.3f}")
    logger.info(f"  Precision: {binary_df['precision'].iloc[0]:.3f}")
    logger.info(f"  F1-Score: {binary_df['f1_score'].iloc[0]:.3f}")
    if binary_df["auc_roc"].iloc[0] is not None:
        logger.info(f"  AUC-ROC: {binary_df['auc_roc'].iloc[0]:.3f}")

    # ---------------- Multinomial Classification ----------------
    logger.info(
        "Training and evaluating multinomial LASSO (Neurosurgeon_Postop_Visual_Outcome)..."
    )

    multinomial_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        C=C_multinomial,
        random_state=42,
    )

    multinomial_model.fit(X_train, y_train_multi)

    # Predictions
    y_pred_multi = multinomial_model.predict(X_test)

    # Compute metrics as DataFrames
    multiclass_overall_df, multiclass_per_class_df, confusion_matrix_df = (
        compute_multiclass_metrics_df(
            y_test_multi, y_pred_multi, args.model_type, C_multinomial
        )
    )
    all_dfs.append(multiclass_overall_df)

    logger.info("Multinomial Classification Results:")
    logger.info(f"  Accuracy: {multiclass_overall_df['accuracy'].iloc[0]:.3f}")
    logger.info(
        f"  Precision (macro): {multiclass_overall_df['precision_macro'].iloc[0]:.3f}"
    )
    logger.info(
        f"  Recall (macro): {multiclass_overall_df['recall_macro'].iloc[0]:.3f}"
    )
    logger.info(f"  F1-Score (macro): {multiclass_overall_df['f1_macro'].iloc[0]:.3f}")
    logger.info(
        f"  F1-Score (weighted): {multiclass_overall_df['f1_weighted'].iloc[0]:.3f}"
    )

    # Log per-class metrics
    logger.info("Per-class metrics:")
    for _, row in multiclass_per_class_df.iterrows():
        logger.info(f"  Class {row['class']}:")
        logger.info(f"    Precision: {row['precision']:.3f}")
        logger.info(f"    Sensitivity: {row['sensitivity']:.3f}")
        logger.info(f"    F1-Score: {row['f1_score']:.3f}")

    # ------------------------------------------------------------------
    # Save DataFrames as CSV files
    # ------------------------------------------------------------------

    # 1. Combined overall metrics (binary + multinomial overall)
    combined_overall_df = pd.concat(all_dfs, ignore_index=True)
    if args.K is not None:
        overall_csv_path = (
            args.output_dir / f"{args.model_type}_lasso_overall_metrics_top{args.K}.csv"
        )
    else:
        overall_csv_path = (
            args.output_dir / f"{args.model_type}_lasso_overall_metrics.csv"
        )
    combined_overall_df.to_csv(overall_csv_path, index=False)
    logger.info(f"Saved overall metrics to {overall_csv_path}")

    # 2. Per-class metrics for multinomial
    per_class_csv_path = (
        args.output_dir / f"{args.model_type}_lasso_per_class_metrics.csv"
    )
    multiclass_per_class_df.to_csv(per_class_csv_path, index=False)
    logger.info(f"Saved per-class metrics to {per_class_csv_path}")

    # 3. Confusion matrix
    confusion_csv_path = (
        args.output_dir / f"{args.model_type}_lasso_confusion_matrix.csv"
    )
    confusion_matrix_df.to_csv(confusion_csv_path, index=False)
    logger.info(f"Saved confusion matrix to {confusion_csv_path}")

    # 4. Comprehensive CSV with all metrics (binary + multiclass overall + per-class)
    comprehensive_csv_path = (
        args.output_dir / f"{args.model_type}_lasso_all_metrics.csv"
    )

    comprehensive_data = []

    # Add binary metrics
    for _, row in binary_df.iterrows():
        comprehensive_data.append(row.to_dict())

    # Add multinomial overall metrics
    for _, row in multiclass_overall_df.iterrows():
        comprehensive_data.append(row.to_dict())

    # Add per-class metrics; align columns to binary_df's columns for consistency
    for _, row in multiclass_per_class_df.iterrows():
        row_dict = row.to_dict()
        for col in binary_df.columns:
            if col not in row_dict:
                row_dict[col] = None
        comprehensive_data.append(row_dict)

    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_df.to_csv(comprehensive_csv_path, index=False)
    logger.info(f"Saved comprehensive metrics to {comprehensive_csv_path}")

    # 5. Summary CSV replacing previous JSON summary
    summary_row = {
        "model_type": args.model_type,
        "C_binary": C_binary,
        "C_multinomial": C_multinomial,
        "binary_accuracy": binary_df["accuracy"].iloc[0],
        "binary_sensitivity": binary_df["sensitivity"].iloc[0],
        "binary_specificity": binary_df["specificity"].iloc[0],
        "binary_precision": binary_df["precision"].iloc[0],
        "binary_f1": binary_df["f1_score"].iloc[0],
        "binary_auc_roc": binary_df["auc_roc"].iloc[0],
        "multinomial_accuracy": multiclass_overall_df["accuracy"].iloc[0],
        "multinomial_precision_macro": multiclass_overall_df["precision_macro"].iloc[0],
        "multinomial_recall_macro": multiclass_overall_df["recall_macro"].iloc[0],
        "multinomial_f1_macro": multiclass_overall_df["f1_macro"].iloc[0],
        "multinomial_f1_weighted": multiclass_overall_df["f1_weighted"].iloc[0],
    }
    summary_df = pd.DataFrame([summary_row])
    summary_csv_path = (
        args.output_dir / f"{args.model_type}_lasso_evaluation_summary.csv"
    )
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved evaluation summary to {summary_csv_path}")

    logger.info("=== DONE LASSO MODEL EVALUATION ===")

    # Return DataFrames for programmatic use
    return {
        "overall_metrics": combined_overall_df,
        "per_class_metrics": multiclass_per_class_df,
        "confusion_matrix": confusion_matrix_df,
        "comprehensive_metrics": comprehensive_df,
        "summary": summary_df,
    }


if __name__ == "__main__":
    main()
