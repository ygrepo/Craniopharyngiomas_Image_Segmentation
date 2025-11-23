#!/usr/bin/env python3
"""
Extract latent vectors from nnU-Net v2 bottleneck layer for all cases.
"""
import argparse
import sys
import os
from pathlib import Path

import torch
import numpy as np
import os
from pathlib import Path

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def extract_case_ids_from_directory(images_dir: Path) -> list:
    """Extract unique case IDs from image files in directory."""
    case_ids = set()
    for file_path in images_dir.glob("*.nii.gz"):
        # Extract case ID by removing channel suffix (_0000, _0001, etc.)
        case_id = file_path.stem.rsplit("_", 1)[0]
        case_ids.add(case_id)
    return sorted(list(case_ids))


def process_case(
    predictor,
    case_id: str,
    images_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> bool:
    """Process a single case and extract latent features."""
    # Channel order must match dataset.json: 0: FLAIR, 1: T1CE, 2: T2
    channel_paths = [
        images_dir / f"{case_id}_0000.nii.gz",  # FLAIR
        images_dir / f"{case_id}_0001.nii.gz",  # T1CE
        images_dir / f"{case_id}_0002.nii.gz",  # T2
    ]

    # Check if all channels exist
    missing_channels = [p for p in channel_paths if not p.exists()]
    if missing_channels:
        logger.warning(f"Skipping {case_id}: missing channels {missing_channels}")
        return False

    logger.info(f"Processing case: {case_id}")

    # Register hook on bottleneck
    latent_features = {}

    def get_activation(name):
        def hook(module, inp, out):
            latent_features[name] = out.detach().cpu()

        return hook

    network = predictor.network
    bottleneck_module = network.encoder.stages[-1]
    handle = bottleneck_module.register_forward_hook(get_activation("bottleneck"))

    try:
        # Use predictor's data iterator
        list_of_lists = [[str(p) for p in channel_paths]]

        # Get data iterator
        data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
            list_of_lists,
            seg_from_prev_stage_files=[None],
            output_filenames_truncated=[None],
            num_processes=1,
        )

        # Get preprocessed data
        preprocessed = next(data_iterator)
        data = preprocessed["data"]

        # Convert to tensor if it's a file path
        if isinstance(data, str):
            data_np = np.load(data)
            # Clean up temp file if needed
            if os.path.exists(data):
                os.remove(data)
            data = torch.from_numpy(data_np)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Ensure proper batch dimension
        if data.dim() == 4:  # (C, D, H, W)
            input_tensor = data.unsqueeze(0).to(device)  # (1, C, D, H, W)
        else:  # Already has batch dim
            input_tensor = data.to(device)

        # Try to make dimensions compatible by padding if needed
        _, c, d, h, w = input_tensor.shape

        def make_divisible(size, divisor=32):
            return int(np.ceil(size / divisor) * divisor)

        target_d = make_divisible(d)
        target_h = make_divisible(h)
        target_w = make_divisible(w)

        if target_d != d or target_h != h or target_w != w:
            pad_d = target_d - d
            pad_h = target_h - h
            pad_w = target_w - w
            padding = (0, pad_w, 0, pad_h, 0, pad_d)
            input_tensor = torch.nn.functional.pad(
                input_tensor, padding, mode="constant", value=0
            )

        # Forward pass to trigger hook
        with torch.no_grad():
            _ = network(input_tensor)

        # Check if hook was triggered
        if "bottleneck" not in latent_features:
            logger.error(f"Hook was not triggered for case {case_id}")
            return False

        raw_features = latent_features["bottleneck"]  # (1, C, D, H, W)

        # Global average pooling over spatial dims
        pooled_features = torch.mean(raw_features, dim=(2, 3, 4)).squeeze()  # (C,)

        # Convert to numpy
        lasso_input_vector = pooled_features.numpy()

        # Save the feature vector
        output_path = output_dir / f"{case_id}.npy"
        np.save(output_path, lasso_input_vector)
        logger.info(f"Saved feature vector for {case_id}: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to process case {case_id}: {e}")
        return False
    finally:
        handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract latent vectors from nnU-Net v2"
    )
    parser.add_argument("--model_folder", type=Path, help="nnU-Net model folder")
    parser.add_argument("--output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint_best.pth", help="Checkpoint name"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument("--log_file", type=Path, help="Log file path")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Setup logging
    setup_logging(args.log_file, args.log_level)

    # --- CONFIGURATION ---
    model_folder = args.model_folder.resolve()
    base_output_dir = args.output_dir.resolve()

    # Input directories
    train_images_dir = Path("nnUNet_raw/Dataset503_CP/imagesTr")
    test_images_dir = Path("nnUNet_raw/Dataset503_CP/imagesTs")

    # Output directories
    train_output_dir = base_output_dir / "imagesTr"
    test_output_dir = base_output_dir / "imagesTs"

    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = args.checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    logger.info(f"Model folder: {model_folder}")
    logger.info(f"Train images dir: {train_images_dir}")
    logger.info(f"Test images dir: {test_images_dir}")
    logger.info(f"Train output dir: {train_output_dir}")
    logger.info(f"Test output dir: {test_output_dir}")

    # --- STEP 1: Load predictor / model ---
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name=checkpoint_name,
    )

    network = predictor.network
    network.eval()
    network.to(device)

    # --- STEP 2: Process training cases ---
    if train_images_dir.exists():
        train_case_ids = extract_case_ids_from_directory(train_images_dir)
        logger.info(f"Found {len(train_case_ids)} training cases")

        train_success = 0
        for case_id in train_case_ids:
            if process_case(
                predictor, case_id, train_images_dir, train_output_dir, device
            ):
                train_success += 1

        logger.info(
            f"Successfully processed {train_success}/{len(train_case_ids)} training cases"
        )
    else:
        logger.warning(f"Training images directory not found: {train_images_dir}")

    # --- STEP 3: Process test cases ---
    if test_images_dir.exists():
        test_case_ids = extract_case_ids_from_directory(test_images_dir)
        logger.info(f"Found {len(test_case_ids)} test cases")

        test_success = 0
        for case_id in test_case_ids:
            if process_case(
                predictor, case_id, test_images_dir, test_output_dir, device
            ):
                test_success += 1

        logger.info(
            f"Successfully processed {test_success}/{len(test_case_ids)} test cases"
        )
    else:
        logger.warning(f"Test images directory not found: {test_images_dir}")

    logger.info("Feature extraction completed!")


if __name__ == "__main__":
    main()
