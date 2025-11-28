#!/usr/bin/env python3
import argparse
import sys
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import get_logger, setup_logging  # type: ignore

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Clinical / radiomics feature lists
# ---------------------------------------------------------------------

RADIOMICS_FEATURES = [
    "Min_Distance_mm",
    "Hausdorff95_mm",
    "Overlap_Volume_mm3",
    "Contact",
]

CLINICAL_PREOP = [
    "Patient_MRN",
    "Age_at_Surgery_Years",
    "Sex_Male",
    "Preop_VIS_Score",
    "Preop_Visual_Field_Deficit",
    "CCI",
    "MFI5",
    "MFI11",
    "Race_Asian Indian",
    "Race_Black or African American",
    "Race_Other",
    "Race_White",
    "Neurosurgeon_Postop_Visual_Outcome",
    "Outcome_Improved",
]

CLINICAL_POSTOP = [
    "Patient_MRN",
    "Age_at_Surgery_Years",
    "Sex_Male",
    "Preop_VIS_Score",
    "Preop_Visual_Field_Deficit",
    "CCI",
    "MFI5",
    "MFI11",
    "EEA",
    "EOR",
    "Race_Asian Indian",
    "Race_Black or African American",
    "Race_Other",
    "Race_White",
    "Neurosurgeon_Postop_Visual_Outcome",
    "Outcome_Improved",
]


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------


def get_args():
    ap = argparse.ArgumentParser(
        description=(
            "Fit a (binary or multinomial) logistic model on scaled features, "
            "compute per-feature importance, and save a ranked CSV."
        )
    )
    ap.add_argument(
        "--npz_path",
        type=Path,
        required=True,
        help=(
            "Path to the scaled NPZ file (e.g. preop_train_binary_scaled.npz or "
            "preop_train_multinomial_scaled.npz). Must contain X, y_bin or y_multi, "
            "and feature_names."
        ),
    )
    ap.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["preop", "postop"],
        help="Determines which clinical feature list to use for grouping.",
    )
    ap.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["binary", "multinomial"],
        help="Binary (Outcome_Improved) or multinomial (Neurosurgeon_Postop_Visual_Outcome).",
    )
    ap.add_argument(
        "--penalty",
        type=str,
        default="l1",
        choices=["l1", "l2", "elasticnet"],
        help="Regularization type for LogisticRegression.",
    )
    ap.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength for LogisticRegression.",
    )
    ap.add_argument(
        "--l1_ratio",
        type=float,
        default=0.3,
        help=(
            "ElasticNet mixing parameter (0 = pure L2, 1 = pure L1). "
            "Used only if penalty='elasticnet'."
        ),
    )
    ap.add_argument(
        "--drop_threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold on importance to flag features to drop. "
            "If None, no drop_flag is computed. If >0, features with "
            "importance < threshold will have drop_flag = True."
        ),
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Where to save feature importance CSV. If None, a default name is "
            "derived from npz_path and penalty."
        ),
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="compute_feature_importance.log",
    )
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return ap.parse_args()


# ---------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------


def load_npz(path: Path, task: Literal["binary", "multinomial"]):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    feature_names = d["feature_names"].astype(str)

    if task == "binary":
        if "y_bin" not in d:
            raise KeyError("NPZ does not contain 'y_bin' for binary task.")
        y = d["y_bin"]
    else:
        if "y_multi" not in d:
            raise KeyError("NPZ does not contain 'y_multi' for multinomial task.")
        y = d["y_multi"]

    return X, y, feature_names


def map_feature_group(
    feature_name: str,
    clinical_list: list[str],
) -> str:
    """Return 'radiomics', 'clinical', or 'latent'."""
    if feature_name in RADIOMICS_FEATURES:
        return "radiomics"
    if feature_name in clinical_list:
        return "clinical"
    # everything else is considered latent (e.g., UNet bottom-layer features)
    return "latent"


# ---------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------


def compute_importance(
    model: LogisticRegression,
    task: Literal["binary", "multinomial"],
) -> np.ndarray:
    """
    Compute per-feature importance from fitted LogisticRegression.

    Binary: abs(coef_).
    Multinomial: L2 norm across classes (sqrt(sum_c coef[c]^2)).
    """
    coef = model.coef_  # shape (1, n_features) or (n_classes, n_features)

    if task == "binary":
        # coef shape: (1, n_features)
        importance = np.abs(coef[0, :])
    else:
        # coef shape: (n_classes, n_features)
        importance = np.linalg.norm(coef, axis=0)  # L2 norm across classes

    return importance


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN FEATURE IMPORTANCE COMPUTATION ===")
    logger.info(f"NPZ: {args.npz_path}")
    logger.info(f"model_type = {args.model_type}, task = {args.task}")
    logger.info(
        f"penalty = {args.penalty}, C = {args.C}, l1_ratio = {args.l1_ratio} "
        f"(only used if penalty='elasticnet')"
    )

    # Pick clinical list
    if args.model_type == "preop":
        clinical_list = CLINICAL_PREOP
    else:
        clinical_list = CLINICAL_POSTOP

    # Load data
    X, y, feature_names = load_npz(args.npz_path, args.task)
    n_samples, n_features = X.shape
    logger.info(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # Configure LogisticRegression
    if args.penalty == "elasticnet":
        solver = "saga"
        l1_ratio = args.l1_ratio
    elif args.penalty == "l1":
        solver = "saga"
        l1_ratio = None
    else:  # "l2"
        # liblinear handles binary; saga can do multi-class as well.
        solver = "lbfgs" if args.task == "multinomial" else "liblinear"
        l1_ratio = None

    multi_class = "auto"  # let sklearn decide

    logger.info(
        f"Fitting LogisticRegression with solver={solver}, penalty={args.penalty}, "
        f"multi_class={multi_class}"
    )

    model = LogisticRegression(
        penalty=args.penalty,
        solver=solver,
        C=args.C,
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        l1_ratio=l1_ratio,
        multi_class=multi_class,
    )
    model.fit(X, y)

    # Compute importance
    importance = compute_importance(model, args.task)
    assert importance.shape[0] == n_features

    # Build DataFrame
    groups = [
        map_feature_group(fn, clinical_list=clinical_list) for fn in feature_names
    ]

    df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "group": groups,
            "importance": importance,
        }
    )

    # Rank globally and within groups
    df["rank_global"] = (
        df["importance"].rank(method="dense", ascending=False).astype(int)
    )
    df["rank_within_group"] = (
        df.groupby("group")["importance"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    # Optional drop flag
    if args.drop_threshold is not None:
        thr = float(args.drop_threshold)
        df["drop_flag"] = df["importance"] < thr
        logger.info(
            f"Applied drop_threshold = {thr}. "
            f"{df['drop_flag'].sum()} / {len(df)} features flagged to drop."
        )
    else:
        df["drop_flag"] = False

    # Summary logging
    logger.info("Top 10 features by importance:")
    for _, row in df.sort_values("importance", ascending=False).head(10).iterrows():
        logger.info(
            f"  {row['feature_name']:<40} "
            f"group={row['group']:<10} importance={row['importance']:.4f} "
            f"rank_global={row['rank_global']}"
        )

    # Default output path
    if args.output_csv is None:
        penalty_tag = args.penalty
        if args.penalty == "elasticnet":
            penalty_tag += f"_l1ratio_{args.l1_ratio}"
        suffix = f"{args.model_type}_{args.task}_{penalty_tag}_feature_importance.csv"
        args.output_csv = args.npz_path.with_name(suffix)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved feature importance CSV to {args.output_csv}")
    logger.info("=== DONE FEATURE IMPORTANCE COMPUTATION ===")


if __name__ == "__main__":
    main()
