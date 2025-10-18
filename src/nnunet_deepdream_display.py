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
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
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
)

logger = get_logger(__name__)


# ---------------------------
# Helpers
# ---------------------------
def _pick_vis_channel(volume: np.ndarray) -> np.ndarray:
    """Return (D,H,W) grayscale volume from input shaped as:
    - (D,H,W)           → return as-is
    - (C,D,H,W)         → choose channel 2 if C>2 else 0  (BraTS: T1CE≈2)
    - (1,C,D,H,W)       → squeeze batch then choose channel
    - (D,H,W,C)         → move channels to front then choose
    """
    arr = volume
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]  # -> (C,D,H,W)
    if arr.ndim == 4:
        if arr.shape[0] in (1, 2, 3, 4, 5):
            C = arr.shape[0]
            vis_ch = 2 if C > 2 else 0
            return arr[vis_ch]
        elif arr.shape[-1] in (1, 2, 3, 4, 5):
            arr = np.moveaxis(arr, -1, 0)
            C = arr.shape[0]
            vis_ch = 2 if C > 2 else 0
            return arr[vis_ch]
    if arr.ndim == 3:
        return arr
    raise ValueError(
        f"Unsupported image shape {arr.shape}; expected (D,H,W) or (C,D,H,W) or (1,C,D,H,W)"
    )


def _pick_same_channel(tensor_1C_DHW: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Ensures delta/dream channel choice matches the base image channel choice.
    If tensor is (1,C,D,H,W) or (C,D,H,W) or (D,H,W), returns (D,H,W) with same channel index rule as base.
    """
    arr = tensor_1C_DHW
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]  # (C,D,H,W)
    if arr.ndim == 4:
        # pick channel like base: 2 if >2 else 0
        C = arr.shape[0]
        vis_ch = 2 if C > 2 else 0
        return arr[vis_ch]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported delta/dream shape {arr.shape}")


def _coerce_same_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop both to the minimum (D,H,W)."""
    if a.shape == b.shape:
        return a, b
    D = min(a.shape[0], b.shape[0])
    H = min(a.shape[1], b.shape[1])
    W = min(a.shape[2], b.shape[2])
    if (D, H, W) != a.shape or (D, H, W) != b.shape:
        logger.warning(
            f"Shape mismatch {a.shape} vs {b.shape}; cropping both to ({D},{H},{W})"
        )
    return a[:D, :H, :W], b[:D, :H, :W]


def _normalize_img_robust(img_3d: np.ndarray) -> np.ndarray:
    base = img_3d.astype(np.float32, copy=False)
    lo, hi = np.percentile(base, 1), np.percentile(base, 99)
    if hi <= lo:
        hi = lo + 1e-6
    base = (base - lo) / (hi - lo)
    return np.clip(base, 0.0, 1.0)


def _normalize_heat_abs(delta_3d: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """Normalize |delta| to [0,1] using robust percentile."""
    a = np.abs(delta_3d).astype(np.float32)
    hi = np.percentile(a, pct)
    if hi <= 1e-12:
        hi = 1e-6
    a = np.clip(a / hi, 0.0, 1.0)
    return a


def _render_signed_overlay(
    gray: np.ndarray, delta: np.ndarray, q: float = 99.0, alpha: float = 0.5
):
    """Create a red/blue signed overlay (no external show_cam_on_image dependency)."""
    import matplotlib.cm as cm

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


def _save_png(img: np.ndarray, path: Path):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


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
    img_3d_all = _pick_vis_channel(image_arr)
    img_3d = _normalize_img_robust(img_3d_all)

    # Delta (D,H,W)
    if delta_arr is not None:
        delta_3d = _pick_same_channel(delta_arr, img_3d_all)
    elif dream_arr is not None:
        dream_3d = _pick_same_channel(dream_arr, img_3d_all)
        delta_3d = dream_3d - img_3d_all  # both are raw intensities for same channel
    else:
        raise ValueError("Provide at least --delta_path or --dream_path.")

    # Align shapes (if dream/delta came from padded/unpadded passes)
    delta_3d, img_3d = _coerce_same_shape(delta_3d, img_3d)

    for z in z_slices:
        if not (0 <= z < img_3d.shape[0]):
            logger.warning(f"z={z} out of bounds [0,{img_3d.shape[0]-1}] — skipping")
            continue
        sl_gray = img_3d[z]
        sl_delta = delta_3d[z]

        if mode == "abs":
            # Use |delta| heat like CAM in [0,1]
            heat = _normalize_heat_abs(sl_delta, pct=abs_pct)
            # manual lightweight overlay (no Grad-CAM util): colorize heat with 'jet'
            import matplotlib.cm as cm

            cmap = cm.get_cmap("jet")
            heat_rgb = cmap(heat)[..., :3].astype(np.float32)

            rgb = np.stack([sl_gray, sl_gray, sl_gray], axis=-1)
            overlay = (1 - alpha) * rgb + alpha * heat_rgb
            overlay = np.clip(overlay, 0.0, 1.0)
            out_png = out_dir / f"{prefix}_deepdream_abs_z{z:03d}.png"
            _save_png(overlay, out_png)
            logger.info(f"[ok] Saved {out_png}")

        elif mode == "signed":
            overlay = _render_signed_overlay(
                sl_gray, sl_delta, q=signed_pct, alpha=alpha
            )
            out_png = out_dir / f"{prefix}_deepdream_signed_z{z:03d}.png"
            _save_png(overlay, out_png)
            logger.info(f"[ok] Saved {out_png}")
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


def _load_volume_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return load_npy(path)
    # assume blosc2 .b2nd
    arr = blosc2.open(str(path))
    return np.asarray(arr)


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")

    # Load base MRI
    img_arr = _load_volume_any(args.image_path)

    # Load dream/delta
    dream_arr = load_npy(args.dream_path) if args.dream_path is not None else None
    delta_arr = load_npy(args.delta_path) if args.delta_path is not None else None

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
        img_arr = _load_volume_any(args.image_path)
        base_all = _pick_vis_channel(img_arr)  # (D,H,W), raw intensities

        dream_arr = load_npy(args.dream_path) if args.dream_path is not None else None
        delta_arr = load_npy(args.delta_path) if args.delta_path is not None else None
        if delta_arr is not None:
            delta_3d = _pick_same_channel(delta_arr, base_all)
        elif dream_arr is not None:
            dream_3d = _pick_same_channel(dream_arr, base_all)
            delta_3d = dream_3d - base_all
        else:
            raise ValueError("Provide --delta_path or --dream_path to export NIfTI.")

        # Align shapes if needed
        delta_3d, base_all = _coerce_same_shape(delta_3d, base_all)

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
            dream_3d, _ = _coerce_same_shape(
                _pick_same_channel(dream_arr, base_all), base_all
            )
            save_nifti_3d(
                dream_3d, affine, out_dir / f"{prefix}_dream.nii.gz", dtype=np.float32
            )

        logger.info("[ok] Slicer-ready NIfTI exports written to %s", out_dir)


if __name__ == "__main__":
    main()
