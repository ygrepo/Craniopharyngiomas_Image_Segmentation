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
    ap = argparse.ArgumentParser(description="Compute radiomics stats.")
    ap.add_argument(
        "--input_csv",
        type=Path,
        default=Path("nnUNet_raw/Dataset503_CP/radiomics_results.csv"),
        help="Path to input CSV with radiomics results.",
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/CP/radiomics_stats.csv"),
        help="Path to output CSV with stats.",
    )
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


def main():
    args = get_args()
    setup_logging(args.log_file, args.log_level)
    logger.info(f"Args: {args}")
    input_csv = args.input_csv.resolve()
    # Load CSV
    df = pd.read_csv(input_csv)

    # Identify numeric radiomics variables
    radiomics_cols = [
        "Min_Distance_mm",
        "Hausdorff95_mm",
        "Overlap_Volume_mm3",
        "Contact",
    ]

    # Convert to numeric (handles missing values gracefully)
    df[radiomics_cols] = df[radiomics_cols].apply(pd.to_numeric, errors="coerce")

    # Compute mean, std, median
    stats = df[radiomics_cols].agg(["mean", "std", "median"]).T

    logger.info(f"Stats: {stats}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv.resolve()
    logger.info(f"Saving stats to {output_csv}")
    stats.to_csv(output_csv, index=True)


if __name__ == "__main__":
    main()
