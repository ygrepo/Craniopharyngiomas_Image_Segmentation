#!/usr/bin/env python3
"""
Export nnU-Net v2 .b2nd (preprocessed) image & seg to NIfTI for 3D Slicer.

Usage:
  python export_b2nd_to_nifti.py \
      --root nnUNet_preprocessed/Dataset503_CP/nnUNetPlans_3d_fullres \
      --case 75062101 \
      --out out_nifti

Requires: blosc2, SimpleITK, numpy
  pip install blosc2 SimpleITK
"""
import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import blosc2 as b2
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def load_spacing_from_pkl(pkl_path: Path):
    """
    Try to fetch spacing (z, y, x) from the nnU-Net v2 case .pkl.
    Falls back to None if not found (we’ll default to (1,1,1)).
    """
    try:
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)

        # Common keys you may find:
        # 'spacing' (z,y,x) after resampling/cropping for this case
        # (Other datasets may store under slightly different keys.)
        if isinstance(meta, dict):
            for key in ["spacing", "spacing_after_resampling", "resampled_spacing"]:
                if key in meta and meta[key] is not None:
                    sp = tuple(float(v) for v in meta[key])
                    if len(sp) == 3:
                        return sp  # (z, y, x)
    except Exception as e:
        logger.info(f"Warning: could not read spacing from {pkl_path} ({e})")

    return None


def numpy_to_sitk(arr_zyx: np.ndarray, spacing_zyx=None, is_label=False) -> sitk.Image:
    """
    Convert (Z, Y, X) numpy array to SimpleITK image.
    spacing_zyx: (z, y, x). SITK expects (x, y, z), so we reverse.
    For label maps, use an integer pixel type.
    """
    # Ensure C-contiguous
    arr_zyx = np.ascontiguousarray(arr_zyx)

    # Choose pixel type
    if is_label:
        # Ensure non-negative and integer type (Slicer labelmap-friendly)
        if arr_zyx.dtype.kind != "i" and arr_zyx.dtype.kind != "u":
            arr_zyx = arr_zyx.astype(np.int16)
    else:
        # Float32 is safe for images
        if arr_zyx.dtype != np.float32 and arr_zyx.dtype != np.float64:
            arr_zyx = arr_zyx.astype(np.float32)

    img = sitk.GetImageFromArray(arr_zyx)  # SITK creates (z,y,x) correctly

    if spacing_zyx is not None:
        z, y, x = spacing_zyx
        img.SetSpacing((float(x), float(y), float(z)))  # SITK uses (x, y, z)

    # Leave origin/direction as defaults (0, identity).
    # (You can set them if you have them in the pkl.)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, help="Folder containing the case .b2nd/.pkl"
    )
    ap.add_argument("--case", required=True, help="Case ID, e.g., 75062101")
    ap.add_argument("--out", required=True, help="Output folder for NIfTI")
    args = ap.parse_args()
    setup_logging(None, "DEBUG")

    root = Path(args.root)
    case = args.case
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    img_b2_path = root / f"{case}.b2nd"
    seg_b2_path = root / f"{case}_seg.b2nd"
    pkl_path = root / f"{case}.pkl"

    if not img_b2_path.exists():
        raise FileNotFoundError(f"Missing image file: {img_b2_path}")
    if not seg_b2_path.exists():
        raise FileNotFoundError(f"Missing seg file: {seg_b2_path}")

    # Load arrays
    img_nd = np.asarray(b2.open(img_b2_path)[:])  # shape (C, Z, Y, X)
    seg_nd = np.asarray(b2.open(seg_b2_path)[:])  # shape (1, Z, Y, X) ints incl. -1

    if img_nd.ndim != 4:
        raise ValueError(f"Expected image shape (C,Z,Y,X), got {img_nd.shape}")
    if seg_nd.ndim != 4 or seg_nd.shape[0] != 1:
        logger.info(f"Warning: expected seg shape (1,Z,Y,X), got {seg_nd.shape}")

    C, Z, Y, X = img_nd.shape
    logger.info(f"Image shape: (C,Z,Y,X)=({C},{Z},{Y},{X}) dtype={img_nd.dtype}")
    logger.info(f"Seg shape: {seg_nd.shape} dtype={seg_nd.dtype}")
    uniques = np.unique(seg_nd)
    logger.info(f"Seg uniques before fix: {uniques}")

    # Map ignore (-1) to background (0) for a clean labelmap in Slicer
    seg_fixed = seg_nd.copy()
    seg_fixed[seg_fixed < 0] = 0
    seg_fixed = seg_fixed.astype(np.uint8)  # labelmaps fine as uint8
    logger.info(f"Seg uniques after fix: {np.unique(seg_fixed)}")

    # Try to get spacing (z,y,x) from pkl (fall back to 1mm)
    spacing_zyx = load_spacing_from_pkl(pkl_path)
    if spacing_zyx is None:
        spacing_zyx = (1.0, 1.0, 1.0)
        logger.info("Spacing not found in pkl; defaulting to (1.0,1.0,1.0) mm (z,y,x).")
    else:
        logger.info(f"Using spacing from pkl (z,y,x): {spacing_zyx}")

    # Save each image channel as its own NIfTI (good for Slicer)
    for c in range(C):
        vol_c = img_nd[c]  # (Z,Y,X)
        img_sitk = numpy_to_sitk(vol_c, spacing_zyx=spacing_zyx, is_label=False)
        out_path = outdir / f"{case}_000{c}_preproc.nii.gz"
        sitk.WriteImage(img_sitk, str(out_path))
        logger.info(f"Saved image channel {c}: {out_path}")

    # Save seg labelmap
    seg_sitk = numpy_to_sitk(seg_fixed[0], spacing_zyx=spacing_zyx, is_label=True)
    seg_out_path = outdir / f"{case}_seg_preproc.nii.gz"
    sitk.WriteImage(seg_sitk, str(seg_out_path))
    logger.info(f"Saved segmentation: {seg_out_path}")


if __name__ == "__main__":
    main()
