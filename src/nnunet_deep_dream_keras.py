#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import torch
from typing import Tuple, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, load_model_from_results, setup_logging

logger = get_logger(__name__)


def load_nnunet_preprocessed_case(
    preprocessed_dir: Path, case_id: str
) -> Tuple[torch.Tensor, Dict]:
    """
    Load preprocessed nnU-Net data (.pkl files).

    Args:
        preprocessed_dir: Path to nnUNet_preprocessed/DatasetXXX_BraTS2017/nnUNetTrainer__nnUNetPlans__3d_fullres
        case_id: Case identifier (e.g., 'Brats17_TCIA_001_1')

    Returns:
        Tensor of shape [1, 4, D, H, W] and metadata
    """

    # Load the preprocessed image data
    pkl_path = preprocessed_dir / f"{case_id}.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {pkl_path}")

    # Load the pickle file
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)

    # The pickle file typically contains:
    # - 'data': the preprocessed image array [C, D, H, W]
    # - 'properties': metadata about preprocessing

    if isinstance(data_dict, dict):
        if "data" in data_dict:
            image_data = data_dict["data"]
            properties = data_dict.get("properties", {})
        else:
            # Sometimes the data is stored differently
            image_data = data_dict
            properties = {}
    else:
        # Sometimes it's just the array
        image_data = data_dict
        properties = {}

    # Convert to tensor and add batch dimension
    if isinstance(image_data, np.ndarray):
        tensor = torch.from_numpy(image_data).float()
    else:
        tensor = torch.tensor(image_data).float()

    # Ensure correct shape: [1, C, D, H, W]
    if tensor.dim() == 4:  # [C, D, H, W]
        tensor = tensor.unsqueeze(0)  # [1, C, D, H, W]

    metadata = {
        "case_id": case_id,
        "shape": tensor.shape,
        "properties": properties,
        "preprocessed": True,
    }

    logger.info(f"Loaded preprocessed case {case_id}: shape {tensor.shape}")

    return tensor, metadata


# -------------------------- main -------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()

    # Model arguments
    ap.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to nnU-Net results directory",
    )
    ap.add_argument("--fold", type=int, default=0, help="Model fold to use")
    ap.add_argument(
        "--checkpoint", type=str, default="checkpoint_final.pth", help="Checkpoint name"
    )
    ap.add_argument(
        "--preprocessed_dir",
        type=Path,
        required=True,
        help="Path to nnU-Net preprocessed data (e.g., nnUNet_preprocessed/Dataset001_BraTS2017)",
    )
    ap.add_argument(
        "--case_id",
        type=str,
        required=True,
        help="BraTS case ID (e.g., Brats17_TCIA_001_1)",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="logs/nnunet_deepdream_keras.log",
        help="Log file path (in addition to console).",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")
    preprocessed_data_dir = args.preprocessed_dir.resolve()
    logger.info(f"Preprocessed data dir: {preprocessed_data_dir}")

    input_tensor, data_meta = load_nnunet_preprocessed_case(
        preprocessed_data_dir, args.case_id
    )
