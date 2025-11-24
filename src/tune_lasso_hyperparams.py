#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_macro

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(
        description=(
            "Grid search LASSO hyperparameters (C) using train/val sets "
            "for a given model_type (preop/postop)."
        )
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
        help="Directory containing *_train_scaled.npz and *_val_scaled.npz.",
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
        help="Path to save best hyperparameters. "
        "If not provided, defaults to <data_dir>/<model_type>_lasso_hyperparams.json",
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
    y_multi = d["y_multi"]
    feature_names = d["feature_names"]
    return X, y_bin, y_multi, feature_names


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    if args.output_json is None:
        args.output_json = args.data_dir / f"{args.model_type}_lasso_hyperparams.json"

    logger.info("=== BEGIN LASSO HYPERPARAM TUNING ===")
    logger.info(f"model_type = {args.model_type}")

    train_path = args.data_dir / f"{args.model_type}_train_scaled.npz"
    val_path = args.data_dir / f"{args.model_type}_val_scaled.npz"

    X_train, y_train_bin, y_train_multi, feat_names = load_npz(train_path)
    X_val, y_val_bin, y_val_multi, _ = load_npz(val_path)

    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    C_grid = [0.01, 0.1, 1.0, 10.0]

    # ---------------- Binary LASSO ----------------
    logger.info("Tuning binary LASSO (Outcome_Worsened)...")
    best_C_bin = None
    best_auc = -np.inf

    for C in C_grid:
        logger.info(f"Evaluating C = {C} (binary)")
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
            C=C,
        )
        model.fit(X_train, y_train_bin)

        if len(np.unique(y_val_bin)) > 1:
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val_bin, y_prob)
            logger.info(f"  Val AUC = {auc:.3f}")
        else:
            auc = 0.0
            logger.warning("Val set has a single class for binary outcome.")

        if auc > best_auc:
            best_auc = auc
            best_C_bin = C

    logger.info(f"Best C (binary) = {best_C_bin}, Val AUC = {best_auc:.3f}")

    # ---------------- Multinomial LASSO ----------------
    logger.info("Tuning multinomial LASSO (Neurosurgeon_Postop_Visual_Outcome)...")
    best_C_multi = None
    best_f1 = -np.inf

    for C in C_grid:
        logger.info(f"Evaluating C = {C} (multinomial)")
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            multi_class="multinomial",
            n_jobs=-1,
            C=C,
        )
        model.fit(X_train, y_train_multi)

        y_pred = model.predict(X_val)
        f1 = f1_macro(y_val_multi, y_pred)
        logger.info(f"  Val macro-F1 = {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_C_multi = C

    logger.info(f"Best C (multinomial) = {best_C_multi}, Val macro-F1 = {best_f1:.3f}")

    # Save best hyperparameters
    hparams = {
        "model_type": args.model_type,
        "binary": {"C": best_C_bin, "val_auc": best_auc},
        "multinomial": {"C": best_C_multi, "val_f1_macro": best_f1},
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(hparams, f, indent=2)

    logger.info(f"Saved best hyperparameters to {args.output_json}")
    logger.info("=== DONE LASSO HYPERPARAM TUNING ===")


if __name__ == "__main__":
    main()
