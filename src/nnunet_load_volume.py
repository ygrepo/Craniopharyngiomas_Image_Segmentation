#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pickle
import pprint
import numpy as np
import torch
import blosc2


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging, load_volume

logger = get_logger(__name__)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=Path,
        required=False,
        default="nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres",
        help="Path to data dir (e.g., nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres)",
    )
    ap.add_argument(
        "--case",
        type=str,
        required=False,
        default="Brats17_CBICA_AAG_1",
        help="Case stem (e.g., 'Brats17_CBICA_AAG_1')",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="logs/nnunet_load_volume.log",
        help="Log file path (in addition to console).",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")
    data_dir = args.data_dir.resolve()
    case_stem = args.case  # e.g., 'Brats17_CBICA_AAG_1'

    load_volume(data_dir, case_stem)
    logger.info("[OK] Done.")


if __name__ == "__main__":
    main()
