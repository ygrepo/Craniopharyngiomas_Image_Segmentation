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
from src.util import (
    get_logger,
    setup_logging,
    load_model_from_results,
    pick_target_layer,
)

logger = get_logger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(
        description="""Load a nnU-Net v2 layer from results folder and print some info. 
        Useful to sanity-check paths and configs.
        """
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
        "--layer_regex",
        type=str,
        default=r"encoder|down|context|stem",
        help="Regex to pick encoder conv",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="logs/nnunet_pickup_layer.log",
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


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")
    model = load_model_from_results(
        model_dir=args.model_dir.resolve(),
        fold=args.fold,
        checkpoint_name="checkpoint_best.pth",
        trainer=None,
    )
    target_layer = pick_target_layer(model, args.layer_regex)
    logger.info(f"Target layer: {target_layer}")
    logger.info("[OK] Done.")


if __name__ == "__main__":
    main()
