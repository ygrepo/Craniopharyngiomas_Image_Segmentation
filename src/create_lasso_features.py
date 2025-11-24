#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(
        description=(
            "Prepare design matrices for LASSO: merge latent, radiomics, clinical; "
            "split train/val/test; scale features for chosen model_type (preop/postop)."
        )
    )
    ap.add_argument(
        "--latent_dir",
        type=Path,
        default=Path("nnUNet_results/Dataset503_CP/.../latent_features"),
        help="Directory containing latent features (.npy files).",
    )
    ap.add_argument(
        "--radiomics_csv",
        type=Path,
        default=Path("nnUNet_raw/Dataset503_CP/radiomics_results.csv"),
        help="Path to radiomics CSV file.",
    )
    ap.add_argument(
        "--clinical_csv",
        type=Path,
        default=Path("clinical_data.csv"),
        help="Path to clinical metadata.",
    )
    ap.add_argument(
        "--model_type",
        type=str,
        choices=["preop", "postop"],
        required=True,
        help="Which model design to prepare: 'preop' or 'postop'.",
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.20,
        help="Fraction of training set to use as validation.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("merged_lasso_inputs"),
        help="Directory to save outputs.",
    )
    ap.add_argument("--log_file", type=Path, default="prepare_lasso_data.log")
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return ap.parse_args()


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    logger.info("=== BEGIN PREPARE LASSO DATA ===")
    logger.info(f"model_type = {args.model_type}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # PART 1: Load Latent Features
    # ---------------------------------------------------------
    logger.info("Loading latent vectors...")
    latent_rows = []

    for npy_file in sorted(args.latent_dir.glob("*.npy")):
        case_id = npy_file.stem
        vec = np.load(npy_file)

        row = {"Case_ID": str(case_id)}
        for i, v in enumerate(vec):
            row[f"Latent_{i}"] = float(v)
        latent_rows.append(row)

    df_latent = pd.DataFrame(latent_rows)
    logger.info(f"latent shape = {df_latent.shape}")

    # ---------------------------------------------------------
    # PART 2: Load Radiomics (has Split column)
    # ---------------------------------------------------------
    logger.info("Loading radiomics...")
    df_rad = pd.read_csv(args.radiomics_csv)
    df_rad["Case_ID"] = df_rad["Case_ID"].astype(str)

    if "Split" not in df_rad.columns:
        raise ValueError("Radiomics CSV missing required column 'Split'")

    rad_features = [
        "Min_Distance_mm",
        "Hausdorff95_mm",
        "Overlap_Volume_mm3",
        "Contact",
    ]
    for col in rad_features:
        if col in df_rad.columns:
            df_rad[col] = pd.to_numeric(df_rad[col], errors="coerce")

    logger.info(f"radiomics shape = {df_rad.shape}")

    # ---------------------------------------------------------
    # PART 3: Load Clinical
    # ---------------------------------------------------------
    logger.info("Loading clinical...")
    df_clin = pd.read_csv(args.clinical_csv)

    # Clean MRN and create Case_ID from it
    if "Patient_MRN" not in df_clin.columns:
        raise ValueError("Clinical CSV must contain column: Patient_MRN")

    df_clin["Patient_MRN"] = df_clin["Patient_MRN"].astype(str)
    df_clin["Case_ID"] = df_clin["Patient_MRN"].str.replace(r"^JH", "", regex=True)

    if "Neurosurgeon_Postop_Visual_Outcome" not in df_clin.columns:
        raise ValueError(
            "Clinical CSV must contain 'Neurosurgeon_Postop_Visual_Outcome'"
        )

    # Binary outcome if not already present
    if "Outcome_Worsened" not in df_clin.columns:
        df_clin["Outcome_Worsened"] = (
            df_clin["Neurosurgeon_Postop_Visual_Outcome"].astype(str) == "Worsened"
        ).astype(int)

    clinical_vars = [
        "Patient_MRN",
        "Patient_Num",
        "Age_at_Surgery_Years",
        "Sex_Male",
        "Preop_VIS_Score",
        "Preop_Visual_Field_Deficit",
        "CCI",
        "MFI5",
        "MFI11",
        "EEA",
        "EOR",
        "Neurosurgeon_Postop_Visual_Outcome",
        "Outcome_Worsened",
    ]
    missing = [c for c in clinical_vars if c not in df_clin.columns]
    if missing:
        logger.error(f"Missing columns in clinical CSV: {missing}")
        raise ValueError(f"Missing clinical variables: {missing}")

    df_clin = df_clin[["Case_ID"] + clinical_vars]
    logger.info(f"clinical shape (clean) = {df_clin.shape}")

    # ---------------------------------------------------------
    # PART 4: Merge latent + radiomics + clinical
    # ---------------------------------------------------------
    logger.info("Merging latent, radiomics, clinical...")
    df = df_rad.merge(df_latent, on="Case_ID", how="inner")
    df = df.merge(df_clin, on="Case_ID", how="inner")
    logger.info(f"merged master shape = {df.shape}")

    merged_path = args.output_dir / "merged_master_full.csv"
    df.to_csv(merged_path, index=False)
    logger.info(f"Saved merged master to {merged_path}")

    # ---------------------------------------------------------
    # PART 5: Build pre-op and post-op design matrices
    # ---------------------------------------------------------
    latent_cols = [c for c in df.columns if c.startswith("Latent_")]

    # Pre-op imaging-only design (radiomics + latent)
    df_preop = df[
        ["Case_ID"]
        + rad_features
        + latent_cols
        + ["Outcome_Worsened", "Neurosurgeon_Postop_Visual_Outcome"]
    ].copy()

    # Post-op full design (imaging + clinical + EEA + EOR)
    df_postop = df[
        ["Case_ID", "Patient_Num"]
        + rad_features
        + latent_cols
        + [
            "Age_at_Surgery_Years",
            "Sex_Male",
            "Preop_VIS_Score",
            "Preop_Visual_Field_Deficit",
            "CCI",
            "MFI5",
            "MFI11",
            "EEA",
            "EOR",
            "Outcome_Worsened",
            "Neurosurgeon_Postop_Visual_Outcome",
        ]
    ].copy()

    df_preop.to_csv(args.output_dir / "design_preop_full.csv", index=False)
    df_postop.to_csv(args.output_dir / "design_postop_full.csv", index=False)
    logger.info(
        f"Saved design matrices: pre-op {df_preop.shape}, post-op {df_postop.shape}"
    )

    # ---------------------------------------------------------
    # PART 6: Select design + features based on model_type
    # ---------------------------------------------------------
    if args.model_type == "preop":
        design_df = df_preop
        feature_cols = rad_features + latent_cols
    else:  # postop
        design_df = df_postop
        postop_clinical_features = [
            "Age_at_Surgery_Years",
            "Sex_Male",
            "Preop_VIS_Score",
            "Preop_Visual_Field_Deficit",
            "CCI",
            "MFI5",
            "MFI11",
            "EEA",
            "EOR",
        ]
        feature_cols = rad_features + latent_cols + postop_clinical_features

    # ---------------------------------------------------------
    # PART 7: Split into Train/Val/Test (by radiomics Split)
    # ---------------------------------------------------------
    logger.info("Splitting into Train / Validation / Test based on radiomics Split...")
    outcome_col_binary = "Outcome_Worsened"

    df_train_all = df[df["Split"] == "train"].copy()
    df_test_all = df[df["Split"] == "test"].copy()

    logger.info(f"train count (all) = {len(df_train_all)}")
    logger.info(f"test count (all) = {len(df_test_all)}")

    stratify_labels = None
    if df_train_all[outcome_col_binary].nunique() > 1:
        stratify_labels = df_train_all[outcome_col_binary]

    df_train, df_val = train_test_split(
        df_train_all,
        test_size=args.val_frac,
        random_state=42,
        stratify=stratify_labels,
    )

    logger.info(f"train_final = {df_train.shape}")
    logger.info(f"val = {df_val.shape}")
    logger.info(f"test = {df_test_all.shape}")

    def subset_design(df_design, df_split):
        return df_design[df_design["Case_ID"].isin(df_split["Case_ID"])].copy()

    df_design_train = subset_design(design_df, df_train)
    df_design_val = subset_design(design_df, df_val)
    df_design_test = subset_design(design_df, df_test_all)

    # Save full design splits for this model_type
    df_design_train.to_csv(
        args.output_dir / f"{args.model_type}_train_full.csv", index=False
    )
    df_design_val.to_csv(
        args.output_dir / f"{args.model_type}_val_full.csv", index=False
    )
    df_design_test.to_csv(
        args.output_dir / f"{args.model_type}_test_full.csv", index=False
    )
    logger.info(
        f"Saved {args.model_type} full design splits: "
        f"train {df_design_train.shape}, val {df_design_val.shape}, test {df_design_test.shape}"
    )

    # ---------------------------------------------------------
    # PART 8: Build numeric & scaled matrices for chosen model_type
    # ---------------------------------------------------------
    def build_arrays(df_design):
        X = df_design[feature_cols].astype(float).to_numpy()
        y_bin = df_design["Outcome_Worsened"].astype(int).to_numpy()
        y_multi = df_design["Neurosurgeon_Postop_Visual_Outcome"].astype(str).to_numpy()
        return X, y_bin, y_multi

    X_train, y_train_bin, y_train_multi = build_arrays(df_design_train)
    X_val, y_val_bin, y_val_multi = build_arrays(df_design_val)
    X_test, y_test_bin, y_test_multi = build_arrays(df_design_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    def save_npz(path, X, y_bin, y_multi, feature_names):
        np.savez_compressed(
            path,
            X=X,
            y_bin=y_bin,
            y_multi=y_multi,
            feature_names=np.array(feature_names),
        )

    save_npz(
        args.output_dir / f"{args.model_type}_train_scaled.npz",
        X_train_scaled,
        y_train_bin,
        y_train_multi,
        feature_cols,
    )
    save_npz(
        args.output_dir / f"{args.model_type}_val_scaled.npz",
        X_val_scaled,
        y_val_bin,
        y_val_multi,
        feature_cols,
    )
    save_npz(
        args.output_dir / f"{args.model_type}_test_scaled.npz",
        X_test_scaled,
        y_test_bin,
        y_test_multi,
        feature_cols,
    )

    logger.info(
        f"Saved scaled matrices for model_type={args.model_type} "
        "(train/val/test npz)."
    )
    logger.info("=== DONE PREPARE LASSO DATA ===")


if __name__ == "__main__":
    main()
