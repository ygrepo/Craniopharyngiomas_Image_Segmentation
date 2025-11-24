#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(
        description="Prepare design matrices for pre-op and post-op LASSO/classifier models."
    )
    ap.add_argument(
        "--input_excel",
        type=Path,
        required=False,
        default=Path(
            "data/CP_Full_PT_List_June2025_Updated_July2025_Vision_Collection_Yves_Highlight.xlsx"
        ),
        help="Path to CP clinical Excel file.",
    )
    ap.add_argument(
        "--sheet_name",
        type=str,
        default="Craniopharyngioma",
        help="Sheet name to load.",
    )
    ap.add_argument(
        "--output_fn",
        type=Path,
        required=False,
        default=Path("output/data/clinical_design.csv"),
        help="Output file path prefix (without _preop/_postop).",
    )
    # logging
    ap.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Log file path (in addition to console).",
    )
    return ap.parse_args()


def load_data(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    # Normalize Race typo
    if "Race" in df.columns:
        df["Race"] = df["Race"].replace({"Whtie": "White"})

    # Clean Patient_MRN: remove 'JH' prefix if present
    if "Patient_MRN" in df.columns:
        df["Patient_MRN"] = (
            df["Patient_MRN"]
            .astype(str)
            .str.replace(r"^JH", "", regex=True)  # strip leading JH
        )

    # Drop rows where the final outcome is missing
    df = df[~df["Neurosurgeon_Postop_Visual_Outcome"].isna()].copy()

    # Create binary outcome
    df["Outcome_Worsened"] = (
        df["Neurosurgeon_Postop_Visual_Outcome"].astype(str) == "Worsened"
    ).astype(int)

    return df


def build_preop_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    PRE-OP model inputs:
    - Patient_MRN (cleaned, no 'JH')
    - Patient_Num
    - Age_at_Surgery_Years
    - Sex_Male
    - Preop_VIS_Score
    - Preop_Visual_Field_Deficit
    - CCI, MFI5, MFI11
    - Race (one-hot)
    Outcome: Outcome_Worsened + Neurosurgeon_Postop_Visual_Outcome
    """

    preop_cols = [
        "Patient_MRN",
        "Patient_Num",
        "Age_at_Surgery_Years",
        "Sex_Male",
        "Preop_VIS_Score",
        "Preop_Visual_Field_Deficit",
        "CCI",
        "MFI5",
        "MFI11",
        "Race",
        "Outcome_Worsened",
        "Neurosurgeon_Postop_Visual_Outcome",
    ]

    # Ensure all expected columns exist
    missing = [c for c in preop_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s) for pre-op matrix: {missing}")

    df_pre = df[preop_cols].copy()

    # One-hot encode race
    df_pre = pd.get_dummies(df_pre, columns=["Race"], drop_first=True)

    # Move outcomes to the end, keep Patient_MRN as first column
    outcomes = ["Outcome_Worsened", "Neurosurgeon_Postop_Visual_Outcome"]
    cols = [c for c in df_pre.columns if c not in outcomes]
    df_pre = df_pre[cols + outcomes]

    # Ensure Patient_MRN is first
    other_cols = [c for c in df_pre.columns if c != "Patient_MRN"]
    df_pre = df_pre[["Patient_MRN"] + other_cols]

    return df_pre


def build_postop_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    POST-OP model inputs:
    - Patient_MRN (cleaned, no 'JH')
    - Patient_Num
    - Age_at_Surgery_Years
    - Sex_Male
    - Preop_VIS_Score
    - Preop_Visual_Field_Deficit
    - CCI, MFI5, MFI11
    - Race (one-hot)
    - EEA
    - EOR
    Outcome: Outcome_Worsened + Neurosurgeon_Postop_Visual_Outcome
    """

    postop_cols = [
        "Patient_MRN",
        "Patient_Num",
        "Age_at_Surgery_Years",
        "Sex_Male",
        "Preop_VIS_Score",
        "Preop_Visual_Field_Deficit",
        "CCI",
        "MFI5",
        "MFI11",
        "Race",
        "EEA",
        "EOR",
        "Outcome_Worsened",
        "Neurosurgeon_Postop_Visual_Outcome",
    ]

    missing = [c for c in postop_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s) for post-op matrix: {missing}")

    df_post = df[postop_cols].copy()
    df_post = pd.get_dummies(df_post, columns=["Race"], drop_first=True)

    outcomes = ["Outcome_Worsened", "Neurosurgeon_Postop_Visual_Outcome"]
    cols = [c for c in df_post.columns if c not in outcomes]
    df_post = df_post[cols + outcomes]

    # Ensure Patient_MRN is first
    other_cols = [c for c in df_post.columns if c != "Patient_MRN"]
    df_post = df_post[["Patient_MRN"] + other_cols]

    return df_post


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)

    df = load_data(args.input_excel, args.sheet_name)
    logger.info(f"Loaded {len(df)} rows from {args.input_excel}")

    df_preop = build_preop_matrix(df)
    df_postop = build_postop_matrix(df)

    # Save outputs
    pre_fn = args.output_fn.with_name(args.output_fn.stem + "_preop.csv")
    post_fn = args.output_fn.with_name(args.output_fn.stem + "_postop.csv")

    df_preop.to_csv(pre_fn, index=False)
    logger.info(f"PRE-OP matrix shape: {df_preop.shape}")

    df_postop.to_csv(post_fn, index=False)
    logger.info(f"POST-OP matrix shape: {df_postop.shape}")

    logger.info(f"Saved PRE-OP matrix:  {pre_fn}")
    logger.info(f"Saved POST-OP matrix: {post_fn}")


if __name__ == "__main__":
    main()
