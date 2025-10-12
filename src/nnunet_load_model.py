#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, load_model_from_results, setup_logging

logger = get_logger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Load a nnU-Net v2 model from results folder and print some info. "
            "Useful to sanity-check paths and configs."
        )
    )
    ap.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to model dir (e.g., nnUNet_results/nnUNetTrainer__nnUNetPlans__3d_fullres/501_BraTS2017_4ch)",
    )
    ap.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold to load (0 for 5-fold xval; -1 for ensemble)",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Log file path (in addition to console).",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    args = ap.parse_args()
    return args


# -------------------------- main -------------------------- #
def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")
    net, meta = load_model_from_results(
        model_dir=args.model_dir.resolve(),
        fold=args.fold,
        checkpoint_name="checkpoint_best.pth",
        trainer=None,
    )
    logger.info(f"Model: {net}")
    logger.info(f"Meta: {meta}")


if __name__ == "__main__":
    main()
