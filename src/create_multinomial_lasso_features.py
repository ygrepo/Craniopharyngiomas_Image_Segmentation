#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
        default=Path(
            "nnUNet_results/Dataset503_CP/nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres/latent_features/"
        ),
        help="Directory containing latent features (.npy files).",
    )
    ap.add_argument(
        "--radiomics_csv",
        type=Path,
        default=Path("nnUNet_raw/Dataset503_CP/radiomics_results.csv"),
        help="Path to radiomics CSV file.",
    )
    ap.add_argument(
        "--clinical_csv_path",
        type=Path,
        default=Path("data/CP"),
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
        "--test_frac",
        type=float,
        default=0.20,
        help="Fraction of training set to use as test.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/CP"),
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
    # Load Latent Features
    # ---------------------------------------------------------
    logger.info("Loading latent vectors...")
    latent_rows = []

    # Load from imagesTr (these will be training cases)
    train_latent_dir = args.latent_dir / "imagesTr"
    if train_latent_dir.exists():
        for npy_file in sorted(train_latent_dir.glob("*.npy")):
            case_id = npy_file.stem
            vec = np.load(npy_file)

            row = {"Case_ID": str(case_id)}
            for i, v in enumerate(vec):
                row[f"Latent_{i}"] = float(v)
            latent_rows.append(row)
        logger.info(
            f"Loaded {len([r for r in latent_rows])} latent vectors from imagesTr"
        )
    # Load from imagesTs (these will be test cases)
    test_latent_dir = args.latent_dir / "imagesTs"
    if test_latent_dir.exists():
        test_count = 0
        for npy_file in sorted(test_latent_dir.glob("*.npy")):
            case_id = npy_file.stem
            vec = np.load(npy_file)

            row = {"Case_ID": str(case_id)}
            for i, v in enumerate(vec):
                row[f"Latent_{i}"] = float(v)
            latent_rows.append(row)
            test_count += 1
        logger.info(f"Loaded {test_count} latent vectors from imagesTs")

    if not latent_rows:
        raise ValueError(f"No latent vectors found in {args.latent_dir}")

    df = pd.DataFrame(latent_rows)
    logger.info(f"# caseIDs: {df['Case_ID'].nunique()}")
    logger.info(f"latent shape = {df.shape}")

    # ---------------------------------------------------------
    # Load Radiomics
    # ---------------------------------------------------------
    logger.info("Loading radiomics...")
    df_rad = pd.read_csv(args.radiomics_csv)
    df_rad["Case_ID"] = df_rad["Case_ID"].astype(str)
    logger.info(f"# caseIDs: {df_rad['Case_ID'].nunique()}")

    rad_features = [
        "Min_Distance_mm",
        "Hausdorff95_mm",
        "Overlap_Volume_mm3",
        "Contact",
    ]
    for col in rad_features:
        if col in df_rad.columns:
            df_rad[col] = pd.to_numeric(df_rad[col], errors="coerce")

    df_rad = df_rad.drop(columns=["Split"])
    logger.info(f"radiomics shape = {df_rad.shape}")

    logger.info(f"# caseIDs: {df_rad['Case_ID'].nunique()}")
    df = df_rad.merge(df, on="Case_ID", how="inner")
    logger.info(f"After merging radiomics, # caseIDs: {df['Case_ID'].nunique()}")

    # ---------------------------------------------------------
    # Load Clinical
    # ---------------------------------------------------------
    logger.info("Loading clinical...")
    if args.model_type == "preop":
        clinical_csv = args.clinical_csv_path / "clinical_data_preop.csv"
    if args.model_type == "postop":
        clinical_csv = args.clinical_csv_path / "clinical_data_postop.csv"
    df_clin = pd.read_csv(clinical_csv)

    df_clin["Patient_MRN"] = df_clin["Patient_MRN"].astype(str)
    df_clin["Case_ID"] = df_clin["Patient_MRN"].str.replace(r"^JH", "", regex=True)
    logger.info(f"# caseIDs: {df_clin['Case_ID'].nunique()}")

    if "Neurosurgeon_Postop_Visual_Outcome" not in df_clin.columns:
        raise ValueError(
            "Clinical CSV must contain 'Neurosurgeon_Postop_Visual_Outcome'"
        )

    # Binary outcome if not already present
    df_clin["Outcome_Improved"] = (
        df_clin["Neurosurgeon_Postop_Visual_Outcome"].astype(str) == "Improved"
    ).astype(int)

    if args.model_type == "preop":
        clinical_vars = [
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
    if args.model_type == "postop":
        clinical_vars = [
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

    missing = [c for c in clinical_vars if c not in df_clin.columns]
    if missing:
        logger.error(f"Missing columns in clinical CSV: {missing}")
        raise ValueError(f"Missing clinical variables: {missing}")

    df_clin = df_clin[["Case_ID"] + clinical_vars]
    logger.info(f"clinical shape (clean) = {df_clin.shape}")

    # ---------------------------------------------------------
    # Merge latent + radiomics + clinical
    # ---------------------------------------------------------
    logger.info("Merging clinical, radiomics...")
    df = df.merge(df_clin, on="Case_ID", how="inner")
    logger.info(f"After merging clinical, # caseIDs: {df['Case_ID'].nunique()}")

    if args.model_type == "preop":
        merged_path = args.output_dir / "preop_multinomial_classifier_data.csv"
    if args.model_type == "postop":
        merged_path = args.output_dir / "postop_multinomial_classifier_data.csv"

    df.to_csv(merged_path, index=False)
    logger.info(f"Saved merged master to {merged_path}")

    # Identify feature columns (exclude IDs, splits, and outcome labels)
    non_feature_cols = [
        "Case_ID",
        "Patient_MRN",
        "Neurosurgeon_Postop_Visual_Outcome",
        "Outcome_Improved",
    ]

    candidate_cols = [c for c in df.columns if c not in non_feature_cols]

    feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    logger.info(f"Number of feature columns: {len(feature_cols)}")
    logger.debug(f"Feature columns: {feature_cols}")

    # ---------------------------------------------------------
    # Split into Train/Test (by radiomics Split)
    # ---------------------------------------------------------
    logger.info(
        "Splitting into Train / Test (random, stratified on Neurosurgeon_Postop_Visual_Outcome)..."
    )
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_frac,
        random_state=42,
        stratify=df["Neurosurgeon_Postop_Visual_Outcome"],
    )

    logger.info(f"train = {df_train['Case_ID'].nunique()}")
    logger.info(f"test = {df_test['Case_ID'].nunique()}")

    # Save full design splits for this model_type
    df_train.to_csv(
        args.output_dir / f"{args.model_type}_train_multinomial.csv", index=False
    )
    df_test.to_csv(
        args.output_dir / f"{args.model_type}_test_multinomial.csv", index=False
    )
    logger.info(
        f"Saved {args.model_type}: " f"train {df_train.shape}, test {df_test.shape}"
    )

    # ---------------------------------------------------------
    # Build numeric & scaled matrices for chosen model_type
    # ---------------------------------------------------------
    def build_arrays(
        df: pd.DataFrame, feature_cols: list
    ) -> tuple[np.ndarray, np.ndarray]:
        X = df[feature_cols].astype(float).to_numpy()
        y_multi = df["Neurosurgeon_Postop_Visual_Outcome"].astype(str).to_numpy()
        return X, y_multi

    X_train, y_train_multi = build_arrays(df_train, feature_cols)
    unique, counts = np.unique(y_train_multi, return_counts=True)
    logger.info(f"Train multi: {dict(zip(unique, counts))}")
    X_test, y_test_multi = build_arrays(df_test, feature_cols)
    unique_test, counts_test = np.unique(y_test_multi, return_counts=True)
    logger.info(f"Test multi:  {dict(zip(unique_test, counts_test))}")

    # ---- Impute missing values (median recommended for mixed-scale numeric features) ----
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Optionally: quick sanity checks
    logger.info(
        f"NaNs after imputation (train/test): "
        f"{np.isnan(X_train).sum()}, "
        f"{np.isnan(X_test).sum()}"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def save_npz(
        path: Path,
        X: np.ndarray,
        y_multi: np.ndarray,
        feature_names: list,
    ):
        np.savez_compressed(
            path,
            X=X,
            y_multi=y_multi,
            feature_names=np.array(feature_names),
        )

    save_npz(
        args.output_dir / f"{args.model_type}_train_multinomial_scaled.npz",
        X_train_scaled,
        y_train_multi,
        feature_cols,
    )
    save_npz(
        args.output_dir / f"{args.model_type}_test_multinomial_scaled.npz",
        X_test_scaled,
        y_test_multi,
        feature_cols,
    )

    logger.info(
        f"Saved scaled matrices for model_type={args.model_type} " "(train/test npz)."
    )
    logger.info("=== DONE PREPARE LASSO DATA ===")


if __name__ == "__main__":
    main()
