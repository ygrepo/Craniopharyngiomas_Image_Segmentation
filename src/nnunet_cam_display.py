#!/usr/bin/env python3
"""
Grad-CAM / LayerCAM overlay renderer for nnU-Net v2 (BraTS2017).

Loads a 3D CAM (as .npy) and a 3D image volume (as .npy), makes axial overlays for selected z-slices.
- CAM is expected as (D, H, W) in [0, 1] (we'll clamp if needed).
- Image can be (D, H, W), (C, D, H, W), or (1, C, D, H, W). We'll pick a visualization channel (T1CE≈2 if present).
"""

import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import blosc2

from pytorch_grad_cam.utils.image import show_cam_on_image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
    load_npy,  # expects a .npy file and returns a numpy array
)

logger = get_logger(__name__)


def overlay_and_save_pngs(
    cam_3d: np.ndarray,
    img_3d: np.ndarray,
    out_dir: Path,
    prefix: str,
    suffix: str,
    zs: List[int],
    alpha: float = 0.45,
) -> None:
    """Write overlays for given z-slices.

    cam_3d: (D,H,W), ideally in [0,1]
    img_3d: (D,H,W), we normalize 1–99 percentile to [0,1] for display
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize grayscale base to [0,1] using robust percentiles
    base = img_3d.astype(np.float32, copy=False)
    lo = np.percentile(base, 1)
    hi = np.percentile(base, 99)
    base = (base - lo) / max(hi - lo, 1e-6)
    base = np.clip(base, 0.0, 1.0)

    # Ensure CAM is in [0,1]
    cam_3d = cam_3d.astype(np.float32, copy=False)
    cam_min, cam_max = float(cam_3d.min()), float(cam_3d.max())
    if cam_max > cam_min + 1e-8:
        cam_3d = (cam_3d - cam_min) / (cam_max - cam_min)
    cam_3d = np.clip(cam_3d, 0.0, 1.0)

    for z in zs:
        if not (0 <= z < cam_3d.shape[0]):
            logger.warning(
                f"Requested z={z} out of bounds [0, {cam_3d.shape[0]-1}] — skipping"
            )
            continue
        sl_img = base[z]  # (H,W), in [0,1]
        sl_cam = cam_3d[z]  # (H,W), in [0,1]
        rgb = np.stack([sl_img, sl_img, sl_img], axis=-1)  # (H,W,3), float32

        overlay = show_cam_on_image(rgb, sl_cam, use_rgb=True, image_weight=(1 - alpha))

        out_png = out_dir / f"{prefix}_{suffix}_z{z:03d}.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info(f"[ok] Saved overlay: {out_png}")


def _pick_vis_channel(volume: np.ndarray) -> np.ndarray:
    """Return (D,H,W) grayscale volume from input shaped as:
    - (D,H,W)           → return as-is
    - (C,D,H,W)         → choose channel 2 if C>2 else 0
    - (1,C,D,H,W)       → squeeze batch → (C,D,H,W) then proceed
    - (D,H,W,C)         → move channels to front then pick
    """
    arr = volume
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]  # -> (C,D,H,W)
    if arr.ndim == 4:
        # Could be (C,D,H,W) or (D,H,W,C)
        if arr.shape[0] in (1, 2, 3, 4, 5):
            C = arr.shape[0]
            vis_ch = 2 if C > 2 else 0
            return arr[vis_ch]
        elif arr.shape[-1] in (1, 2, 3, 4, 5):
            # (D,H,W,C) → (C,D,H,W)
            arr = np.moveaxis(arr, -1, 0)
            C = arr.shape[0]
            vis_ch = 2 if C > 2 else 0
            return arr[vis_ch]
    if arr.ndim == 3:
        return arr  # already (D,H,W)
    raise ValueError(
        f"Unsupported image shape {arr.shape}; expected (D,H,W) or (C,D,H,W) or (1,C,D,H,W)"
    )


def _coerce_same_shape(
    cam_3d: np.ndarray, img_3d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Make CAM and image the same (D,H,W) by safe cropping to the minimum extents if needed."""
    if cam_3d.shape == img_3d.shape:
        return cam_3d, img_3d
    D = min(cam_3d.shape[0], img_3d.shape[0])
    H = min(cam_3d.shape[1], img_3d.shape[1])
    W = min(cam_3d.shape[2], img_3d.shape[2])
    if (D, H, W) != img_3d.shape or (D, H, W) != cam_3d.shape:
        logger.warning(
            f"Shape mismatch cam={cam_3d.shape} vs img={img_3d.shape}; "
            f"cropping both to ({D},{H},{W})"
        )
    return cam_3d[:D, :H, :W], img_3d[:D, :H, :W]


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Render Grad-CAM overlays for selected axial slices."
    )
    ap.add_argument(
        "--input_dir",
        type=Path,
        required=False,
        default=Path("output/cam"),
        help="Directory containing the CAM .npy (alternative to --cam_path).",
    )
    ap.add_argument(
        "--input_fn",
        type=Path,
        required=False,
        default=Path("Brats17_CBICA_AAG_1_class3_layercam.npy"),
        #        default=Path("Brats17_CBICA_AAG_1_class3_gradcam_cam.npy"),
        help="File name of CAM .npy inside --input_dir (alternative to --cam_path).",
    )
    ap.add_argument(
        "--image_path",
        type=Path,
        required=False,
        default=Path(
            "nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres/Brats17_CBICA_AAG_1.b2nd"
        ),
        help="Path to 3D image .npy (D,H,W) or (C,D,H,W) or (1,C,D,H,W) for base overlay.",
    )

    ap.add_argument(
        "--output_dir",
        type=Path,
        required=False,
        default=Path("output/cam"),
        help="Output directory for PNGs.",
    )
    ap.add_argument(
        "--z_slices",
        type=str,
        default="40,60,80",
        help="Comma-separated axial slice indices to render.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="CAM opacity (0..1). 0 = image only, 1 = CAM only.",
    )

    # Logging (optional)
    ap.add_argument(
        "--log_file", type=Path, default=None, help="Optional log file path."
    )
    ap.add_argument(
        "--log_level", type=str, default="INFO", help="Log level (e.g., INFO, DEBUG)."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")

    # Resolve CAM path
    cam_path = (args.input_dir / args.input_fn).resolve()
    # Load arrays
    cam_3d = load_npy(
        cam_path
    )  # expected (D,H,W), but we'll tolerate more and squeeze later
    img_arr = blosc2.open(str(args.image_path))  # Blosc2 NDArray handle
    img_arr = np.asarray(img_arr)

    # Squeeze CAM to (D,H,W) if possible
    if cam_3d.ndim == 4 and cam_3d.shape[0] == 1:
        cam_3d = cam_3d[0]
    if cam_3d.ndim != 3:
        raise ValueError(
            f"CAM array must be 3D (D,H,W) or 4D with batch=1; got {cam_3d.shape}"
        )

    # Pick visualization channel → (D,H,W)
    img_3d = _pick_vis_channel(img_arr)

    # Align shapes by cropping (common if CAM was computed on padded volume)
    cam_3d, img_3d = _coerce_same_shape(cam_3d, img_3d)

    # Parse slices
    z_list = [int(z) for z in args.z_slices.split(",") if z.strip().isdigit()]
    if not z_list:
        logger.warning("No valid z-slices parsed; defaulting to [40, 60, 80].")
        z_list = [40, 60, 80]

    # Output dir & prefix
    output_dir = args.output_dir.resolve()
    prefix = cam_path.stem

    overlay_and_save_pngs(
        cam_3d=cam_3d,
        img_3d=img_3d,
        out_dir=output_dir,
        prefix=prefix,
        suffix="layercam",
        zs=z_list,
        alpha=float(args.alpha),
    )

    logger.info("[ok] All overlays written.")


if __name__ == "__main__":
    main()
