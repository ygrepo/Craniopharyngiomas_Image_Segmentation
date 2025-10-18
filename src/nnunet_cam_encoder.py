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
    largest_cc_bool,
)

logger = get_logger(__name__)


# ---------------------------
# Targets for segmentation
# ---------------------------
class SegmentationClassAveragedTarget:
    def __init__(
        self, class_idx: int, mask: Optional[torch.Tensor] = None, lam_bg: float = 0.25
    ):
        self.class_idx = int(class_idx)
        self.mask = mask
        self.lam_bg = lam_bg

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 5:
            model_output = model_output[0]
        if model_output.ndim != 4:
            raise AssertionError(f"Expected 4D/5D, got {model_output.ndim}D")
        s = model_output[self.class_idx]  # (D,H,W)
        if self.mask is None:
            return s.mean()
        m = self.mask
        if m.ndim == 4 and m.shape[0] == 1:
            m = m[0]
        if m.shape != s.shape:
            raise AssertionError(f"Mask {m.shape} != score {s.shape}")
        m = m.to(s.device).float()
        inv = 1.0 - m
        fg = (s * m).sum() / m.sum().clamp_min(1)
        bg = (s * inv).sum() / inv.sum().clamp_min(1)
        return fg - self.lam_bg * bg


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
    vol = vol.to(device).contiguous()

    with torch.no_grad():
        logits = model(vol)  # (1, C, D, H, W)

    mask = None
    if use_pred_mask:
        logger.info(
            "Using ground truth or predicted mask to spatially restrict the objective"
        )
        pred = logits.argmax(dim=1)  # (1, D, H, W)
        mask = (pred == class_idx)[0]  # (D, H, W)  <-- drop batch dim
        mask = largest_cc_bool(mask)

    targets = [SegmentationClassAveragedTarget(class_idx, mask=mask, lam_bg=0.25)]
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
        out_dir / f"{case_stem}_class{args.class_idx}_{args.method}.npy",
    )


if __name__ == "__main__":
    main()
