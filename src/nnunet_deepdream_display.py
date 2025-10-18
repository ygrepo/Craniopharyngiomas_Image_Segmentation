#!/usr/bin/env python3
"""
DeepDream overlay renderer for nnU-Net v2 (BraTS2017).

Inputs:
- dream.npy  : (1, C, D, H, W) dreamed volume  (optional)
- delta.npy  : (1, C, D, H, W) difference vs init (preferred)
- image_path : .b2nd (or .npy) with original (1,C,D,H,W) / (C,D,H,W) / (D,H,W)

Output:
- Axial PNG overlays for selected z-slices.

Default visualization:
- mode='abs' → overlay |delta| (robust-normalized to [0,1]) on grayscale MRI.
- mode='signed' → red/blue diverging heat overlay (negative/positive deltas).

Tip:
- For MRI-friendly display we robust-normalize the base (1–99 pct) and delta (1–99 pct of |delta|).
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import matplotlib.cm as cm

import numpy as np
import blosc2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
    load_npy,  # returns numpy array
    load_props_from_pkl,
    get_spacing_origin_from_props,
    affine_from_spacing_origin,
    save_nifti_3d,
    build_heat_and_mask,
    normalize_robust,
    pick_vis_channel,
    pick_channel_like,
    normalize_heat_abs,
    coerce_same_shape,
    load_volume_npy_b2nd,
)
from src.plot_util import save_image

logger = get_logger(__name__)


# ---------------------------
# Helpers
# ---------------------------


def _render_signed_overlay(
    gray: np.ndarray, delta: np.ndarray, q: float = 99.0, alpha: float = 0.5
):
    """Create a red/blue signed overlay (no external show_cam_on_image dependency)."""

    # Normalize base
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    # Symmetric clipping for signed delta
    lim = np.percentile(np.abs(delta), q)
    lim = lim if lim > 1e-12 else 1e-6
    norm = np.clip(delta / lim, -1.0, 1.0)

    # Map to colormap (bwr)
    cmap = cm.get_cmap("bwr")
    heat = cmap((norm + 1.0) / 2.0)[..., :3].astype(np.float32)  # drop alpha

    # Alpha blend
    out = (1 - alpha) * rgb + alpha * heat
    out = np.clip(out, 0.0, 1.0)
    return out


# ---------------------------
# Core
# ---------------------------
def overlay_deepdream(
    image_arr: np.ndarray,  # base MRI: (1,C,D,H,W) / (C,D,H,W) / (D,H,W) / (D,H,W,C)
    dream_arr: Optional[
        np.ndarray
    ],  # (1,C,D,H,W) dreamed; can be None if delta provided
    delta_arr: Optional[
        np.ndarray
    ],  # (1,C,D,H,W) delta; if None, computed as dream - image (same channel)
    z_slices: List[int],
    out_dir: Path,
    prefix: str,
    alpha: float = 0.45,
    mode: str = "abs",  # 'abs' or 'signed'
    abs_pct: float = 99.0,  # robust pct for |delta|
    signed_pct: float = 99.0,  # robust pct for signed normalization
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base grayscale (D,H,W)
    img_3d_all, vis_ch = pick_vis_channel(image_arr)
    img_3d = normalize_robust(img_3d_all)

    # Delta (D,H,W)
    if delta_arr is not None:
        delta_3d = pick_channel_like(delta_arr, vis_ch)
    elif dream_arr is not None:
        dream_3d = pick_channel_like(dream_arr, vis_ch)
        delta_3d = dream_3d - img_3d_all  # both are raw intensities for same channel
    else:
        raise ValueError("Provide at least --delta_path or --dream_path.")

    # Align shapes (if dream/delta came from padded/unpadded passes)
    delta_3d, img_3d = coerce_same_shape(delta_3d, img_3d)

    for z in z_slices:
        if not (0 <= z < img_3d.shape[0]):
            logger.warning(f"z={z} out of bounds [0,{img_3d.shape[0]-1}] — skipping")
            continue
        sl_gray = img_3d[z]
        sl_delta = delta_3d[z]

        if mode == "abs":
            # Use |delta| heat like CAM in [0,1]
            heat = normalize_heat_abs(sl_delta, pct=abs_pct)
            # manual lightweight overlay (no Grad-CAM util): colorize heat with 'jet'
            import matplotlib.cm as cm

            cmap = cm.get_cmap("jet")
            heat_rgb = cmap(heat)[..., :3].astype(np.float32)

            rgb = np.stack([sl_gray, sl_gray, sl_gray], axis=-1)
            overlay = (1 - alpha) * rgb + alpha * heat_rgb
            overlay = np.clip(overlay, 0.0, 1.0)
            out_png = out_dir / f"{prefix}_deepdream_abs_z{z:03d}.png"
            save_image(overlay, out_png)

        elif mode == "signed":
            overlay = _render_signed_overlay(
                sl_gray, sl_delta, q=signed_pct, alpha=alpha
            )
            out_png = out_dir / f"{prefix}_deepdream_signed_z{z:03d}.png"
            save_image(overlay, out_png)
        else:
            raise ValueError("mode must be 'abs' or 'signed'")


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Render DeepDream overlays for axial slices."
    )
    ap.add_argument(
        "--dream_path", type=Path, default=None, help="Path to dream.npy (1,C,D,H,W)."
    )
    ap.add_argument(
        "--delta_path", type=Path, default=None, help="Path to delta.npy (1,C,D,H,W)."
    )
    ap.add_argument(
        "--case",
        type=str,
        default="Brats17_CBICA_AAG_1",
        help="Case stem, e.g., 'Brats17_CBICA_AAG_1'.",
    )
    ap.add_argument(
        "--image_path",
        type=Path,
        default=Path(
            "nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres/Brats17_CBICA_AAG_1.b2nd"
        ),
        help="Base MRI volume: .b2nd (preferred) or .npy.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/deepdream"),
        help="Output directory for PNGs.",
    )
    ap.add_argument(
        "--z_slices",
        type=str,
        default="40,60,80",
        help="Comma-separated axial slice indices.",
    )
    ap.add_argument(
        "--objective",
        type=str,
        default="logit",
        choices=["logit", "feature"],
        help="Maximize a class logit or an internal feature channel.",
    )
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity (0..1).")
    ap.add_argument(
        "--mode",
        type=str,
        default="abs",
        choices=["abs", "signed"],
        help="Overlay mode.",
    )
    ap.add_argument(
        "--abs_pct",
        type=float,
        default=99.0,
        help="Robust percentile for |delta| normalization.",
    )
    ap.add_argument(
        "--signed_pct",
        type=float,
        default=99.0,
        help="Robust percentile for signed normalization.",
    )
    ap.add_argument(
        "--props_path",
        type=Path,
        required=False,
        default=Path(
            "nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres/Brats17_CBICA_AAG_1.pkl"
        ),
        help="Path to the nnU-Net props .pkl for this case (for spacing/origin).",
    )
    ap.add_argument(
        "--save_slicer",
        type=int,
        default=1,
        help="Save NIfTI exports for 3D Slicer (1/0).",
    )
    ap.add_argument(
        "--mask_pct",
        type=float,
        default=97.5,
        help="Percentile on |delta| to make a binary mask (set e.g. 95–99). Use -1 to skip.",
    )

    ap.add_argument("--log_file", type=Path, default=None, help="Optional log file.")
    ap.add_argument("--log_level", type=str, default="INFO")
    # in parse_args()
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")

    # Load base MRI
    img_arr = load_volume_npy_b2nd(args.image_path)

    # Load dream/delta
    fn = args.dream_path / f"{args.case}_{args.objective}_dream.npy"
    dream_arr = load_npy(fn) if args.dream_path is not None else None
    fn = args.delta_path / f"{args.case}_{args.objective}_delta.npy"
    delta_arr = load_npy(fn) if args.delta_path is not None else None

    # Squeeze possible leading batch dim in dream/delta handled in _pick_same_channel
    z_list = [int(z) for z in args.z_slices.split(",") if z.strip().isdigit()]
    if not z_list:
        logger.warning("No valid z-slices parsed; defaulting to [40,60,80].")
        z_list = [40, 60, 80]

    out_dir = args.output_dir.resolve()
    prefix = (args.delta_path or args.dream_path or Path("deepdream")).stem

    overlay_deepdream(
        image_arr=img_arr,
        dream_arr=dream_arr,
        delta_arr=delta_arr,
        z_slices=z_list,
        out_dir=out_dir,
        prefix=prefix,
        alpha=float(args.alpha),
        mode=args.mode,
        abs_pct=float(args.abs_pct),
        signed_pct=float(args.signed_pct),
    )
    logger.info("[ok] All overlays written.")
    if args.save_slicer:
        # 1) Load props and build affine
        props = load_props_from_pkl(args.props_path)
        spacing, origin = get_spacing_origin_from_props(props)
        affine = affine_from_spacing_origin(spacing, origin)

        # 2) Recompute the same base/heat/mask we used for overlays
        #    (reuse internal helpers)
        img_arr = load_volume_npy_b2nd(args.image_path)
        base_all, vis_ch = pick_vis_channel(img_arr)  # (D,H,W), raw intensities

        dream_arr = load_npy(args.dream_path) if args.dream_path is not None else None
        delta_arr = load_npy(args.delta_path) if args.delta_path is not None else None
        if delta_arr is not None:
            delta_3d = pick_channel_like(delta_arr, vis_ch)
        elif dream_arr is not None:
            dream_3d = pick_channel_like(dream_arr, vis_ch)
            delta_3d = dream_3d - base_all
        else:
            raise ValueError("Provide --delta_path or --dream_path to export NIfTI.")

        # Align shapes if needed
        delta_3d, base_all = coerce_same_shape(delta_3d, base_all)

        # 3) Build heat + optional mask
        heat_3d, mask_3d = build_heat_and_mask(
            delta_3d,
            abs_pct=float(args.abs_pct),
            bin_pct=(None if float(args.mask_pct) < 0 else float(args.mask_pct)),
        )

        # 4) Write NIfTI files (float32 for image/heat; uint8 for mask)
        out_dir = args.output_dir.resolve()
        prefix = (args.delta_path or args.dream_path or Path("deepdream")).stem

        # Base image (raw intensities of the chosen channel)
        save_nifti_3d(
            base_all, affine, out_dir / f"{prefix}_image.nii.gz", dtype=np.float32
        )

        # DeepDream heat (0..1)
        save_nifti_3d(
            heat_3d,
            affine,
            out_dir / f"{prefix}_deepdream_heat_abs.nii.gz",
            dtype=np.float32,
        )

        # Binary mask (optional)
        if mask_3d is not None:
            pct_int = int(round(float(args.mask_pct)))
            save_nifti_3d(
                mask_3d,
                affine,
                out_dir / f"{prefix}_deepdream_mask_p{pct_int}.nii.gz",
                dtype=np.uint8,
            )

        # Dream volume (optional, if --dream_path)
        if dream_arr is not None:
            dream_3d, _ = coerce_same_shape(
                pick_channel_like(dream_arr, vis_ch), base_all
            )
            save_nifti_3d(
                dream_3d, affine, out_dir / f"{prefix}_dream.nii.gz", dtype=np.float32
            )

        logger.info("[ok] Slicer-ready NIfTI exports written to %s", out_dir)


if __name__ == "__main__":
    main()
