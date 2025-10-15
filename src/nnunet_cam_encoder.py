#!/usr/bin/env python3
"""
Grad-CAM / LayerCAM on nnU-Net v2 (BraTS2017) encoder layers.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
    load_model_from_results,
    pick_target_layer,
    load_volume,
    downsample_multiples,
    pad_to_multiples,
    unpad_3d,
    save_npy,
)

logger = get_logger(__name__)


# ---------------------------
# Targets for segmentation
# ---------------------------
class SegmentationClassAveragedTarget:
    def __init__(self, class_idx: int, mask: Optional[torch.Tensor] = None):
        """
        mask: (D,H,W) or (1,D,H,W) boolean. Will be reshaped to match.
        """
        self.class_idx = int(class_idx)
        self.mask = mask  # can be None

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        # Accept (C,D,H,W) or (N,C,D,H,W)
        if model_output.ndim == 5:
            # batch size should be 1 in our usage; take the first item
            logger.info("Using batch size 1")
            model_output = model_output[0]
        elif model_output.ndim != 4:
            raise AssertionError(f"Expected 4D or 5D output, got {model_output.ndim}D")

        # model_output: (C, D, H, W)
        logits_cdhw = model_output
        score_map = logits_cdhw[self.class_idx]  # (D,H,W)

        if self.mask is not None:
            logger.info("Using mask to spatially restrict the objective")
            mask = self.mask
            # Normalize mask to (D,H,W) on correct device/dtype
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]
            assert (
                mask.shape == score_map.shape
            ), f"Mask shape {mask.shape} != class map {score_map.shape}"
            mask = mask.to(score_map.device)
            return (score_map * mask.float()).sum() / (mask.sum().clamp_min(1.0))
        else:
            logger.info("Using mean activation as objective")
            return score_map.mean()


def overlay_and_save_pngs(
    cam_3d: np.ndarray,
    img_3d: np.ndarray,
    out_dir: Path,
    prefix: str,
    zs: List[int],
    alpha: float = 0.45,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    # For grayscale base, use one modality channel (e.g., T1CE ~ channel index 2 in many setups). We’ll pick channel 2 if exists else 0.
    base = img_3d
    base = base - np.percentile(base, 1)
    base = base / (np.percentile(base, 99) - 1e-6)
    base = np.clip(base, 0, 1)

    for z in zs:
        if not (0 <= z < cam_3d.shape[0]):
            continue
        sl_img = base[z]
        sl_cam = cam_3d[z]
        rgb = np.stack([sl_img, sl_img, sl_img], axis=-1)
        overlay = show_cam_on_image(rgb, sl_cam, use_rgb=True, image_weight=(1 - alpha))

        out_png = out_dir / f"{prefix}_z{z:03d}.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info(f"[ok] Saved overlay: {out_png}")


# ---------------------------
# Core Grad-CAM runner
# ---------------------------
def run_cam(
    model: nn.Module,
    vol: torch.Tensor,  # (1,C,D,H,W)
    target_layer: nn.Module,
    class_idx: int,
    method: str = "gradcam",
    use_pred_mask: bool = True,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    vol = vol.to(device)

    with torch.no_grad():
        logits = model(vol)  # (1, C, D, H, W)

    mask = None
    if use_pred_mask:
        logger.info("Using predicted mask to spatially restrict the objective")
        pred = logits.argmax(dim=1)  # (1, D, H, W)
        mask = (pred == class_idx)[0]  # (D, H, W)  <-- drop batch dim

    targets = [SegmentationClassAveragedTarget(class_idx, mask=mask)]
    cam_cls = GradCAM if method.lower() == "gradcam" else LayerCAM
    cam = cam_cls(
        model=model,
        target_layers=[target_layer],
    )

    cam_map = cam(input_tensor=vol, targets=targets, eigen_smooth=False)[
        0
    ]  # (D,H,W), numpy
    # Normalize to 0..1
    cam_min, cam_max = cam_map.min(), cam_map.max()
    cam_norm = (cam_map - cam_min) / (cam_max - cam_min + 1e-8)
    return cam_norm


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        type=Path,
        default="nnUNet_results/Dataset501_BraTS2017_4ch/nnUNetTrainer__nnUNetPlans__3d_fullres/",
        help="Path to model dir (e.g., nnUNet_results/nnUNetTrainer__nnUNetPlans__3d_fullres/501_BraTS2017_4ch)",
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        required=False,
        default="nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres",
        help="Path to data dir (e.g., nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres)",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        required=False,
        default="output/cam",
        help="Path to output dir",
    )
    ap.add_argument(
        "--case",
        type=str,
        required=False,
        default="Brats17_CBICA_AAG_1",
        help="Case stem (e.g., 'Brats17_CBICA_AAG_1')",
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
        "--class_idx",
        type=int,
        default=3,
        help="Target output channel (verify your mapping!)",
    )
    ap.add_argument(
        "--method", type=str, default="layercam", choices=["gradcam", "layercam"]
    )
    ap.add_argument("--use_pred_mask", type=bool, default=True, help="1=True, 0=False")
    ap.add_argument(
        "--log_file",
        type=Path,
        default="logs/nnunet_gradcam_encoder.log",
        help="Log file path (in addition to console).",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    ap.add_argument(
        "--z_slices",
        type=str,
        default="40,60,80",
        help="Comma list of axial slice indices for PNGs",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")
    model, meta = load_model_from_results(
        model_dir=args.model_dir.resolve(),
        fold=args.fold,
        checkpoint_name="checkpoint_best.pth",
        trainer=None,
        compile_network=False,
    )
    data_dir = args.data_dir.resolve()
    case_stem = args.case  # e.g., 'Brats17_CBICA_AAG_1'

    vol_t, props = load_volume(data_dir, case_stem)
    # Pick target layer
    target_layer = pick_target_layer(model, args.layer_regex, target_idx=-1)

    cfg = meta["configuration_manager"]
    mult = downsample_multiples(cfg)  # e.g., (32, 32, 32)
    vol_pad, pads = pad_to_multiples(vol_t, mult)

    # Run CAM
    cam_3d = run_cam(
        model=model,
        vol=vol_pad,
        target_layer=target_layer,
        class_idx=args.class_idx,
        method=args.method,
        use_pred_mask=args.use_pred_mask,
    )  # (D,H,W) in [0,1]
    cam_3d = unpad_3d(cam_3d, pads)

    # Save outputs
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npy(
        cam_3d,
        out_dir / f"{case_stem}_class{args.class_idx}_{args.method}_layercam.npy",
    )


if __name__ == "__main__":
    main()
