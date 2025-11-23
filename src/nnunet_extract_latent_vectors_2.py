#!/usr/bin/env python3
"""
Extract latent vectors from nnU-Net v2 bottleneck layer.
"""
import argparse
import sys
import torch
import numpy as np
import os
from pathlib import Path

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Extract latent vectors from nnU-Net v2"
    )
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    parser.add_argument("--log_file", help="Log file path")
    parser.add_argument("--model_folder", help="nnU-Net model folder")
    parser.add_argument("--input_image", help="Input image path")
    parser.add_argument("--case_id", help="Case ID to process")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument(
        "--checkpoint", default="checkpoint_final.pth", help="Checkpoint name"
    )
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    # Setup logging
    if args.log_file:
        setup_logging(args.log_level, args.log_file)
    logger = get_logger(__name__)

    # --- CONFIGURATION ---
    model_folder = args.model_folder or (
        "nnUNet_results/Dataset503_CP/"
        "nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres"
    )

    case_id = args.case_id or "06780898"
    images_dir = Path("nnUNet_raw/Dataset503_CP/imagesTr")

    # Channel order must match dataset.json: 0: FLAIR, 1: T1CE, 2: T2
    channel_paths = [
        images_dir / f"{case_id}_0000.nii.gz",  # FLAIR
        images_dir / f"{case_id}_0001.nii.gz",  # T1CE
        images_dir / f"{case_id}_0002.nii.gz",  # T2
    ]

    output_dir = Path(
        args.output_dir
        or (
            "nnUNet_results/Dataset503_CP/"
            "nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres/"
            "fold_0/latent_features"
        )
    )

    checkpoint_name = args.checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    logger.info(f"Model folder: {model_folder}")
    logger.info("Using channels:")
    for p in channel_paths:
        logger.info(f"  {p}")
        if not p.exists():
            raise FileNotFoundError(f"Channel file not found: {p}")

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

    # --- STEP 2: Register hook on bottleneck ---
    latent_features = {}

    def get_activation(name):
        def hook(module, inp, out):
            latent_features[name] = out.detach().cpu()

        return hook

    bottleneck_module = network.encoder.stages[-1]
    handle = bottleneck_module.register_forward_hook(get_activation("bottleneck"))
    logger.info(f"Hook registered on: {bottleneck_module}")

    # --- STEP 3: Read and preprocess the case ---
    # Use predictor's data iterator (standard nnUNet approach)
    list_of_lists = [[str(p) for p in channel_paths]]  # Wrap in list for single case

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

    logger.info(f"Input tensor shape: {input_tensor.shape}")
    # Debug: Check if dimensions are divisible by network requirements
    _, c, d, h, w = input_tensor.shape
    logger.info(f"Tensor dimensions - C:{c}, D:{d}, H:{h}, W:{w}")

    # Try to make dimensions compatible by padding if needed
    # nnUNet typically requires dimensions divisible by powers of 2
    def make_divisible(size, divisor=32):
        return int(np.ceil(size / divisor) * divisor)

    target_d = make_divisible(d)
    target_h = make_divisible(h)
    target_w = make_divisible(w)

    if target_d != d or target_h != h or target_w != w:
        logger.info(
            f"Padding tensor from ({d},{h},{w}) to ({target_d},{target_h},{target_w})"
        )
        pad_d = target_d - d
        pad_h = target_h - h
        pad_w = target_w - w

        # Pad: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        input_tensor = torch.nn.functional.pad(
            input_tensor, padding, mode="constant", value=0
        )
        logger.info(f"Padded tensor shape: {input_tensor.shape}")

    # --- STEP 4: Forward pass to trigger hook ---
    with torch.no_grad():
        try:
            _ = network(input_tensor)
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            handle.remove()
            raise

    # Check if hook was triggered
    if "bottleneck" not in latent_features:
        handle.remove()
        raise RuntimeError("Hook was not triggered - check bottleneck module selection")

    raw_features = latent_features["bottleneck"]  # (1, C, D, H, W)
    logger.info(f"Raw bottleneck shape: {raw_features.shape}")

    # Global average pooling over spatial dims
    pooled_features = torch.mean(raw_features, dim=(2, 3, 4)).squeeze()  # (C,)
    logger.info(f"Final feature vector shape: {pooled_features.shape}")

    # Clean up hook
    handle.remove()

    # Convert to numpy for LASSO feature vector
    lasso_input_vector = pooled_features.numpy()
    logger.info("Feature vector extracted successfully")
    logger.info(
        f"Feature vector stats - min: {lasso_input_vector.min():.4f}, "
        f"max: {lasso_input_vector.max():.4f}, "
        f"mean: {lasso_input_vector.mean():.4f}"
    )

    # Save the feature vector
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}.npy"
    np.save(output_path, lasso_input_vector)
    logger.info(f"Saved feature vector to: {output_path}")


if __name__ == "__main__":
    main()
