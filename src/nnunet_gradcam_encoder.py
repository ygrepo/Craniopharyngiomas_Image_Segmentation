#!/usr/bin/env python3
"""
Grad-CAM / LayerCAM on nnU-Net v2 (BraTS2017) encoder layers.

- Loads model from $nnUNet_results/Dataset{ID}_{NAME}/{TR}__{PLANS}__{CFG}/fold_{FOLD}/checkpoint_best.pth
- Loads one preprocessed case (.npz) from $nnUNet_preprocessed/Dataset{ID}_{NAME}/imagesTr
- Computes 3D CAM for a chosen class channel on a chosen encoder layer (regex match)
- Saves: CAM .npy and a few PNG overlays

Usage (example):
  python nnunet_gradcam_encoder.py \
    --dataset_id 501 \
    --dataset_name BraTS2017_4ch \
    --cfg 3d_fullres \
    --trainer nnUNetTrainer \
    --plans_id nnUNetPlans \
    --fold 0 \
    --case_npz imagesTr/BraTS17_001.npz \
    --class_idx 1 \
    --layer_regex "encoder|down|context|stem" \
    --method gradcam \
    --use_pred_mask 1 \
    --out_dir cam_out

Class index mapping to verify for YOUR checkpoint:
  0 = background
  1 = enhancing tumor (ET)
  2 = peritumoral edema (ED)
  3 = non-enhancing/necrotic core (NCR/NET)
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

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
)

logger = get_logger(__name__)


# ---------------------------
# Targets for segmentation
# ---------------------------
class SegmentationClassAveragedTarget:
    """
    For 3D semantic segmentation:
    - model_output: (N, C, D, H, W)
    - class_idx: channel index to emphasize
    - mask: optional boolean/float mask (1,D,H,W) or (N,D,H,W) to spatially restrict the objective
    """

    def __init__(self, class_idx: int, mask: Optional[torch.Tensor] = None):
        self.class_idx = class_idx
        self.mask = mask

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        assert model_output.ndim == 5
        class_map = model_output[
            :, self.class_idx : self.class_idx + 1, ...
        ]  # (N,1,D,H,W)
        if self.mask is None:
            return class_map.mean()
        m = self.mask
        if m.ndim == 4:
            m = m[:, None, ...]  # (N,1,D,H,W)
        m = m.to(class_map).float()
        denom = torch.clamp(m.sum(), min=1.0)
        return (class_map * m).sum() / denom


# ---------------------------
# I/O helpers
# ---------------------------
# def load_case_npz(prep_dir: Path, case_npz: str) -> np.ndarray:
#     """
#     Load preprocessed nnU-Net .npz (C,D,H,W) from $nnUNet_preprocessed/.../imagesTr
#     Returns float32 array normalized per-channel (z-score).
#     """
#     npz_path = prep_dir / case_npz
#     if not npz_path.exists():
#         raise FileNotFoundError(f"Case not found: {npz_path}")
#     d = np.load(npz_path)
#     # nnU-Net v2 stores under key 'data' typically
#     if "data" not in d:
#         # Some pipelines save as unnamed array; fallback
#         arr = list(d.values())[0]
#     else:
#         arr = d["data"]
#     arr = arr.astype(np.float32)  # (C,D,H,W)
#     # Simple per-channel z-score (your pipeline may already be normalized; if so, skip)
#     for c in range(arr.shape[0]):
#         mu, sd = arr[c].mean(), arr[c].std()
#         arr[c] = (arr[c] - mu) / (sd + 1e-6)
#     return arr


def save_cam_npy(cam_3d: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, cam_3d)
    print(f"[ok] Saved CAM volume: {out_path}")


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
        print(f"[ok] Saved overlay: {out_png}")


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
        logits = model(vol)  # (1,C,D,H,W)

    mask = None
    if use_pred_mask:
        logger.info("Using predicted mask to spatially restrict the objective")
        pred = logits.argmax(dim=1)  # (1,D,H,W)
        mask = pred == class_idx

    targets = [SegmentationClassAveragedTarget(class_idx, mask=mask)]
    cam_cls = GradCAM if method.lower() == "gradcam" else LayerCAM
    cam = cam_cls(
        model=model, target_layers=[target_layer], use_cuda=device.type == "cuda"
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
        default="nnUNet_results/Dataset502_BraTS2017_4ch/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/",
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
        "--method", type=str, default="gradcam", choices=["gradcam", "layercam"]
    )
    ap.add_argument("--use_pred_mask", type=bool, default=True, help="1=True, 0=False")
    ap.add_argument("--out_dir", type=str, default="cam_out")
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
    )
    data_dir = args.data_dir.resolve()
    case_stem = args.case  # e.g., 'Brats17_CBICA_AAG_1'

    vol_t, props = load_volume(data_dir, case_stem)

    # # Resolve env paths
    # nnUNet_preprocessed = Path(os.environ["nnUNet_preprocessed"]).resolve()

    # prep_dir = nnUNet_preprocessed / f"Dataset{args.dataset_id}_{args.dataset_name}"
    # case_npz_path = prep_dir / args.case_npz
    # if not case_npz_path.exists():
    #     raise FileNotFoundError(f"Case not found: {case_npz_path}")
    # # Load case data
    # case_arr = load_case_npz(prep_dir, args.case_npz)  # (C,D,H,W)
    # C, D, H, W = case_arr.shape
    # vol_t = torch.from_numpy(case_arr)[None, ...]  # (1,C,D,H,W)

    # Pick target layer
    target_layer = pick_target_layer(model, args.layer_regex)

    # Run CAM
    cam_3d = run_cam(
        model=model,
        vol=vol_t,
        target_layer=target_layer,
        class_idx=args.class_idx,
        method=args.method,
        use_pred_mask=args.use_pred_mask,
    )  # (D,H,W) in [0,1]

    # Save outputs
    # out_dir = Path(args.out_dir)
    # out_dir.mkdir(parents=True, exist_ok=True)
    # base = Path(args.case_npz).stem
    # save_cam_npy(
    #     cam_3d, out_dir / f"{base}_class{args.class_idx}_{args.method}_cam.npy"
    # )

    # # For PNG overlays, pick a visualization channel (T1CE often at index 2; fallback to 0 if absent)
    # vis_ch = 2 if C > 2 else 0
    # img_3d = case_arr[vis_ch]  # (D,H,W)

    # z_list = [int(z) for z in args.z_slices.split(",") if z.strip().isdigit()]
    # overlay_and_save_pngs(
    #     cam_3d,
    #     img_3d,
    #     out_dir,
    #     prefix=f"{base}_c{args.class_idx}_{args.method}",
    #     zs=z_list,
    #     alpha=0.45,
    # )

    # # Optional: print some available conv names to help you refine layer_regex
    # print(
    #     "\n[hint] A few conv layer names that matched your regex (or try printing all):"
    # )
    # matched = [n for n, _ in list_conv_layers(model, args.layer_regex)]
    # for n in matched[:10]:
    #     print("  ", n)
    # if len(matched) > 10:
    #     print(f"  ... (+{len(matched)-10} more)")


if __name__ == "__main__":
    main()
