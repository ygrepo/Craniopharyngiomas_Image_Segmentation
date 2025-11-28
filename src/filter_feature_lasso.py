#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def get_args():
    ap = argparse.ArgumentParser(
        description=(
            "Select top-K features based on a ranked importance CSV and "
            "optionally save a reduced NPZ for binary or multinomial models."
        )
    )
    ap.add_argument(
        "--npz_path",
        type=Path,
        required=True,
        help=(
            "Path to the scaled NPZ file "
            "(e.g. preop_train_binary_scaled.npz or "
            "preop_train_multinomial_scaled.npz). "
            "Must contain X, y_bin or y_multi, and feature_names."
        ),
    )
    ap.add_argument(
        "--importance_csv",
        type=Path,
        required=True,
        help=(
            "Path to the feature importance CSV. "
            "Must contain at least columns ['feature', 'global_rank']."
        ),
    )
    ap.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of top-ranked features to keep.",
    )
    ap.add_argument(
        "--output_npz",
        type=Path,
        default=None,
        help=(
            "Path to save the reduced NPZ file. "
            "If not provided, a default name is derived from npz_path."
        ),
    )
    ap.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["preop", "postop"],
        help="Preop/postop (used for logging and naming conventions only).",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="select_top_k_features.log",
    )
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return ap.parse_args()


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def select_top_k_features(
    npz_path: Path, importance_csv: Path, k: int, save_path: Path | None = None
):
    """
    Load X, y, feature_names from NPZ and reduce to top-K ranked features.

    Parameters
    ----------
    npz_path : Path
        Path to <model_type>_train_binary_scaled.npz or multinomial version.
    importance_csv : Path
        CSV containing at least ['feature', 'global_rank'].
    k : int
        Number of top-ranked features to keep.
    save_path : Path or None
        If provided, saves a reduced NPZ file for downstream training.

    Returns
    -------
    X_reduced : np.ndarray (n_samples, k)
    y : np.ndarray
    selected_features : list[str]
    y_key : str
        Either 'y_bin' or 'y_multi', depending on what was found in the NPZ.
    """

    # ---------- load NPZ ----------
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]
    feat_names = d["feature_names"].astype(str)

    if "y_bin" in d.files:
        y = d["y_bin"]
        y_key = "y_bin"
        task_type = "binary"
    elif "y_multi" in d.files:
        y = d["y_multi"]
        y_key = "y_multi"
        task_type = "multinomial"
    else:
        raise ValueError(
            f"NPZ file {npz_path} must contain either 'y_bin' or 'y_multi'. "
            f"Found keys: {list(d.files)}"
        )

    logger.info(
        f"Loaded NPZ: {npz_path} | X shape = {X.shape}, "
        f"target key = {y_key}, task_type = {task_type}"
    )

    # ---------- load importance ----------
    imp = pd.read_csv(importance_csv)
    if "global_rank" not in imp.columns or "feature" not in imp.columns:
        raise ValueError(
            "Importance CSV must contain columns 'feature' and 'global_rank'. "
            f"Columns found: {imp.columns.tolist()}"
        )

    # sort by global rank
    imp_sorted = imp.sort_values("global_rank", ascending=True)

    # select top-K features
    selected_features = imp_sorted["feature"].head(k).tolist()
    logger.info(
        f"Selecting top-K={k} features based on global_rank "
        f"(first 5: {selected_features[:5]})"
    )

    # ---------- map selected names to column indices ----------
    name_to_idx = {name: idx for idx, name in enumerate(feat_names)}
    missing = [f for f in selected_features if f not in name_to_idx]
    if missing:
        raise ValueError(
            "The following features from importance CSV are missing in NPZ "
            f"feature_names: {missing}"
        )

    selected_indices = [name_to_idx[f] for f in selected_features]

    # ---------- reduce X ----------
    X_reduced = X[:, selected_indices]
    logger.info(
        f"Reduced X from shape {X.shape} to {X_reduced.shape} using top-K features."
    )

    # ---------- save (optional) ----------
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            X=X_reduced,
            **{y_key: y},
            feature_names=np.array(selected_features, dtype=object),
        )
        logger.info(f"Saved reduced NPZ: {save_path}")

    return X_reduced, y, selected_features, y_key


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN SELECT TOP-K FEATURES ===")
    logger.info(f"model_type = {args.model_type}")
    logger.info(f"npz_path = {args.npz_path}")
    logger.info(f"importance_csv = {args.importance_csv}")
    logger.info(f"k = {args.k}")

    # If no output path given, derive a default from npz_path
    if args.output_npz is None:
        stem = args.npz_path.stem
        # e.g. preop_train_binary_scaled -> preop_train_binary_topK_scaled
        if stem.endswith("_scaled"):
            base = stem[:-7]
            new_stem = f"{base}_top{args.k}_scaled"
        else:
            new_stem = f"{stem}_top{args.k}"
        args.output_npz = args.npz_path.with_name(new_stem + ".npz")

    X_reduced, y, selected_features, y_key = select_top_k_features(
        npz_path=args.npz_path,
        importance_csv=args.importance_csv,
        k=args.k,
        save_path=args.output_npz,
    )

    logger.info(
        f"Finished top-K selection for {args.model_type} "
        f"({y_key}, n_samples={X_reduced.shape[0]}, n_features={X_reduced.shape[1]})."
    )
    logger.info("=== DONE SELECT TOP-K FEATURES ===")


if __name__ == "__main__":
    main()
