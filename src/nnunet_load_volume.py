#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import os
import pickle
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def load_volume(
    data_dir: Path,
    case_stem: str,
):
    """
    Use nnU-Net v2's helper to load a preprocessed case (.b2nd + .pkl)
    Returns (data, props) where data is float32 array (C, D, H, W).
    """

    b2nd = data_dir / f"{case_stem}.b2nd"
    pkl = data_dir / f"{case_stem}.pkl"
    if not b2nd.exists() or not pkl.exists():
        raise FileNotFoundError(f"Missing .b2nd or .pkl for {case_stem} in {data_dir}")

    # read properties
    with open(pkl, "rb") as f:
        props = pickle.load(f)

    # infer shape and channels
    size = (
        props.get("size_after_cropping")
        or props.get("size_after_resampling")
        or props.get("size_after_padding")
    )
    if size is None:
        raise KeyError("Could not determine volume size from props.")
    D, H, W = map(int, size)
    C = len(props.get("modalities", [])) or int(props.get("num_modalities", 1))
    logger.info(f"Volume shape: {C, D, H, W}")

    # read binary tensor
    data = np.fromfile(b2nd, dtype=np.float32).reshape(C, D, H, W)
    if data.ndim != 4:
        raise ValueError(f"Unexpected data shape {data.shape} for {case_stem}")
    vol_t = torch.from_numpy(data)[None, ...]  # (1, C, D, H, W)
    return vol_t, props


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to data dir (e.g., nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres)",
    )
    ap.add_argument(
        "--case",
        type=str,
        required=True,
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
