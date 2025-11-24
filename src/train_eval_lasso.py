#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, f1_macro

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(
        description=(
            "Train LASSO models with best hyperparameters; "
            "evaluate on validation and test sets for a given model_type."
        )
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
        help="Directory containing <model_type>_*_scaled.npz.",
    )
    ap.add_argument(
        "--model_type",
        type=str,
        choices=["preop", "postop"],
        required=True,
        help="Which model design to train/evaluate: 'preop' or 'postop'.",
    )
    ap.add_argument(
        "--hparams_json",
        type=Path,
        default=None,
        help="JSON file containing best hyperparameters. "
             "If not provided, defaults to <data_dir>/<model_type>_lasso_hyperparams.json",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="train_eval_lasso.log",
    )
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return ap.args = get_args()
    return ap.parse_args()


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y_bin = d["y_bin"]
    y_multi = d["y_multi"]
    feature_names = d["feature_names"]
    return X, y_bin, y_multi, feature_names


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    if args.hparams_json is None:
        args.hparams_json = args.data_dir / f"{args.model_type}_lasso_hyperparams.json"

    logger.info("=== BEGIN TRAIN/EVAL LASSO ===")
    logger.info(f"model_type = {args.model_type}")

    # Load data
    train_path = args.data_dir / f"{args.model_type}_train_scaled.npz"
    val_path = args.data_dir / f"{args.model_type}_val_scaled.npz"
    test_path = args.data_dir / f"{args.model_type}_test_scaled.npz"

    X_train, y_train_bin, y_train_multi, feat_names = load_npz(train_path)
    X_val, y_val_bin, y_val_multi, _ = load_npz(val_path)
    X_test, y_test_bin, y_test_multi, _ = load_npz(test_path)

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Load hyperparameters
    with open(args.hparams_json, "r") as f:
        hparams = json.load(f)

    C_bin = hparams["binary"]["C"]
    C_multi = hparams["multinomial"]["C"]

    logger.info(f"Using C_bin = {C_bin}, C_multi = {C_multi}")

    # ---------------------------------------------------------
    # Binary LASSO model
    # ---------------------------------------------------------
    logger.info("Fitting binary LASSO (Outcome_Worsened)...")
    model_bin = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        C=C_bin,
    )
    model_bin.fit(X_train, y_train_bin)

    # Eval on validation
    logger.info("Evaluating binary LASSO on validation set...")
    if len(np.unique(y_val_bin)) > 1:
        y_val_prob = model_bin.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val_bin, y_val_prob)
        logger.info(f"Val AUC (binary): {auc_val:.3f}")
    y_val_pred = model_bin.predict(X_val)
    logger.info("Val classification report (binary):")
    logger.info("\n" + classification_report(y_val_bin, y_val_pred))

    # Eval on test
    logger.info("Evaluating binary LASSO on test set...")
    if len(np.unique(y_test_bin)) > 1:
        y_test_prob = model_bin.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test_bin, y_test_prob)
        logger.info(f"Test AUC (binary): {auc_test:.3f}")
    y_test_pred = model_bin.predict(X_test)
    logger.info("Test classification report (binary):")
    logger.info("\n" + classification_report(y_test_bin, y_test_pred))

    # ---------------------------------------------------------
    # Multinomial LASSO model
    # ---------------------------------------------------------
    logger.info(
        "Fitting multinomial LASSO (Neurosurgeon_Postop_Visual_Outcome)..."
    )
    model_multi = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        multi_class="multinomial",
        n_jobs=-1,
        C=C_multi,
    )
    model_multi.fit(X_train, y_train_multi)

    # Eval on validation
    logger.info("Evaluating multinomial LASSO on validation set...")
    y_val_pred_multi = model_multi.predict(X_val)
    f1_val = f1_macro(y_val_multi, y_val_pred_multi)
    logger.info(f"Val macro-F1 (multinomial): {f1_val:.3f}")
    logger.info("Val classification report (multinomial):")
    logger.info("\n" + classification_report(y_val_multi, y_val_pred_multi))

    # Eval on test
    logger.info("Evaluating multinomial LASSO on test set...")
    y_test_pred_multi = model_multi.predict(X_test)
    f1_test = f1_macro(y_test_multi, y_test_pred_multi)
    logger.info(f"Test macro-F1 (multinomial): {f1_test:.3f}")
    logger.info("Test classification report (multinomial):")
    logger.info("\n" + classification_report(y_test_multi, y_test_pred_multi))

    logger.info("=== DONE TRAIN/EVAL LASSO ===")


if __name__ == "__main__":
    main()
