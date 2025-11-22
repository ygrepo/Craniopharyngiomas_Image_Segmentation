#!/usr/bin/env python3
"""
Extract latent vectors from nnU-Net v2 bottleneck layer.
"""
import argparse
import sys
import torch
import numpy as np
from pathlib import Path

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

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
    io = SimpleITKIO()

    # Pass all 3 channels in correct order
    images, props = io.read_images([str(p) for p in channel_paths])

    # Use the preprocessor already configured by the predictor
    # Depending on version, preprocess_single_case can return:
    #   (data, seg, props)  OR a list-like where [0] is data.
    preprocessor_class = predictor.configuration_manager.preprocessor_class
    preprocessor = preprocessor_class(verbose=predictor.verbose)

    logger.info(f"Preprocessor class: {preprocessor_class}")
    logger.info(f"Preprocessor class name: {preprocessor_class.__name__}")
    data, seg, props = preprocessor.run_case(
        [str(p) for p in channel_paths],  # File paths as strings
        None,  # No initial properties needed
        predictor.plans_manager,
        predictor.configuration_manager.configuration,
        predictor.dataset_json,
    )

    # Convert to tensor (C, D, H, W)
    img_np = data.astype(np.float32)
    # If shape is (C, Z, Y, X), this is correct; add batch dimension:
    input_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1, C, D, H, W)

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
