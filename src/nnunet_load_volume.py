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
from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def load_volume(data_dir: Path, case_stem: str):
    """
    Load nnU-Net v2 preprocessed case (.b2nd + .pkl) via Blosc2.
    Returns (torch tensor (1, C, D, H, W), props)
    """
    b2nd = data_dir / f"{case_stem}.b2nd"
    pkl = data_dir / f"{case_stem}.pkl"
    if not b2nd.exists() or not pkl.exists():
        raise FileNotFoundError(f"Missing .b2nd or .pkl for {case_stem} in {data_dir}")

    # props (spacing, crop bbox, etc.), useful for later but not needed to decode b2nd
    with open(pkl, "rb") as f:
        props = pickle.load(f)
    print("Keys in props:\n")
    pprint.pprint(list(props.keys()))

    # --- load compressed array (shape + dtype are stored inside) ---
    nd = blosc2.open(str(b2nd))  # Blosc2 NDArray handle
    arr = np.asarray(nd)  # materialize to NumPy

    # Normalize to (C, D, H, W)
    if arr.ndim == 3:
        # single-channel volume (D, H, W)
        arr = arr[None, ...]  # -> (1, D, H, W)
    elif arr.ndim == 4:
        # either (C, D, H, W) or (D, H, W, C)
        if arr.shape[0] in (1, 2, 3, 4, 5):
            pass  # already (C, D, H, W)
        elif arr.shape[-1] in (1, 2, 3, 4, 5):
            arr = np.moveaxis(arr, -1, 0)  # (D,H,W,C) -> (C,D,H,W)
        else:
            raise ValueError(f"Ambiguous channel axis for shape {arr.shape}")
    else:
        raise ValueError(f"Unexpected ndim {arr.ndim} for {b2nd.name}")

    arr = arr.astype(np.float32, copy=False)
    vol_t = torch.from_numpy(arr)[None, ...]  # (1, C, D, H, W)
    return vol_t, props


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
