"""
Craniopharyngioma MRI preprocessing:
N4 bias correction → resample to isotropic spacing → ROI crop (sellar/suprasellar) → z-score normalization.

Usage (example):
    python craniopharyngioma_preprocess.py \
        --in_dir /path/to/raw_nifti \
        --out_dir /path/to/preproc \
        --modalities T1w T1wCE T2w FLAIR \
        --spacing 1.0 \
        --roi_size_mm 96 96 96 \
        --inferior_offset_mm 20 \
        --save_mask

Notes
-----
- ROI strategy (no atlas): we place a box around the image center, then shift inferiorly by `inferior_offset_mm` along the superior-inferior axis.
  This captures the sellar/suprasellar region for most head MRIs without brittle skull stripping or template registration.
- If you later want an atlas-assisted ROI, you can add a rigid registration step and compute the ROI center from template landmarks.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import SimpleITK as sitk


def read_image(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    return img


def write_image(img: sitk.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def n4_bias_correct(
    img: sitk.Image,
    mask: Optional[sitk.Image] = None,
    shrink_factor: int = 2,
    conv: int = 50,
) -> sitk.Image:
    """
    N4 bias field correction. If no mask is given, we derive one using Otsu on a blurred magnitude image.
    """
    if mask is None:
        # robust mask: Otsu on Gaussian-smoothed image; also erode to avoid background
        sm = sitk.SmoothingRecursiveGaussian(img, 1.0)
        # use absolute (magnitude) in case of signed types
        sm_abs = sitk.Cast(sitk.Abs(sm), sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(sm_abs, 0, 1, 200)  # 0/1 mask
        mask = sitk.BinaryMorphologicalOpening(mask, (1, 1, 1))

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([conv])  # single level; increase if needed
    corrector.SetConvergenceThreshold(1e-6)
    # Work in a lower resolution for speed
    img_shrunk = sitk.Shrink(img, [shrink_factor] * 3)
    mask_shrunk = sitk.Shrink(mask, [shrink_factor] * 3)
    corrected_shrunk = corrector.Execute(img_shrunk, mask_shrunk)
    # Apply estimated field to full-res:
    log_field = corrector.GetLogBiasFieldAsImage(img)
    corrected = sitk.Exp(sitk.Log(img) - log_field)
    # Handle numerical issues: cast back to original pixel type range if possible
    corrected = sitk.Cast(corrected, img.GetPixelID())
    return corrected


def resample_isotropic(
    img: sitk.Image, out_spacing: float = 1.0, interpolator=sitk.sitkBSpline
) -> sitk.Image:
    """
    Resample image to isotropic spacing (out_spacing in mm). Keeps original orientation & origin.
    """
    original_spacing = np.array(list(img.GetSpacing()), dtype=float)
    original_size = np.array(list(img.GetSize()), dtype=int)

    new_spacing = np.array([out_spacing, out_spacing, out_spacing], dtype=float)
    new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0)

    # Identity transform (we're just changing sampling grid)
    resampler.SetTransform(sitk.Transform())
    out = resampler.Execute(img)
    return out


def zscore(
    img: sitk.Image, mask: Optional[sitk.Image] = None, eps: float = 1e-8
) -> sitk.Image:
    """
    Z-score normalize within mask (if given) or within robust foreground determined by Otsu.
    """
    im = sitk.Cast(img, sitk.sitkFloat32)
    if mask is None:
        m = sitk.OtsuThreshold(im, 0, 1, 128)
    else:
        m = mask

    arr = sitk.GetArrayFromImage(im)
    marr = sitk.GetArrayFromImage(m).astype(bool)
    vals = arr[marr]
    if vals.size == 0:
        mean, std = float(np.mean(arr)), float(np.std(arr))
    else:
        # robust: clip extremes before z-score
        v = np.clip(vals, np.percentile(vals, 1), np.percentile(vals, 99))
        mean, std = float(np.mean(v)), float(np.std(v) + eps)

    z = (arr - mean) / (std + eps)
    out = sitk.GetImageFromArray(z)
    out.CopyInformation(im)
    return sitk.Cast(out, sitk.sitkFloat32)


def compute_roi_center_world(
    img: sitk.Image, inferior_offset_mm: float = 20.0
) -> Tuple[float, float, float]:
    """
    Compute ROI center in world coordinates (LPS) near image center with an inferior shift.
    Positive Z is usually towards superior in LPS (depends on orientation), so we detect axis using direction cosines.
    We move 'inferior_offset_mm' along the negative of the dominant axis of the superior-inferior direction.
    """
    # Image center index:
    size = np.array(img.GetSize(), dtype=float)
    center_idx = (size - 1) / 2.0

    # Map to physical:
    center_world = np.array(
        img.TransformIndexToPhysicalPoint([int(round(c)) for c in center_idx])
    )

    # Determine SI axis from direction matrix: columns are direction cosines of axes in physical space
    D = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.array(img.GetSpacing(), dtype=float)
    # Superior-Inferior direction (heuristic): the axis with the largest |Z| component in direction matrix
    si_axis = np.argmax(
        np.abs(D[2, :])
    )  # which image axis contributes most to physical Z
    si_sign = np.sign(D[2, si_axis])  # +1 if increasing index goes towards superior

    # Move towards inferior: subtract along SI axis direction
    # We construct a unit vector in physical coordinates for the chosen image axis:
    axis_dir = D[:, si_axis]  # unit vector in physical space
    shift_vec = -si_sign * axis_dir * inferior_offset_mm
    roi_center = center_world + shift_vec
    return tuple(roi_center.tolist())


def crop_roi_world(
    img: sitk.Image,
    center_world: Tuple[float, float, float],
    size_mm: Tuple[float, float, float],
) -> sitk.Image:
    """
    Crop a rectangular ROI centered at 'center_world' with given physical size (mm).
    Works in index space by converting world coordinates to indices & mm to voxels.
    """
    spacing = np.array(img.GetSpacing(), dtype=float)
    size_vox = np.maximum(1, np.round(np.array(size_mm, dtype=float) / spacing)).astype(
        int
    )

    # Compute start index by transforming world center to index
    center_idx = np.array(img.TransformPhysicalPointToIndex(center_world), dtype=float)
    start_idx = np.round(center_idx - size_vox / 2.0).astype(int)
    end_idx = start_idx + size_vox

    # Clamp to image bounds
    start_idx = np.maximum(start_idx, 0)
    end_idx = np.minimum(end_idx, np.array(img.GetSize(), dtype=int))

    # Extract
    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetSize([int(s) for s in (end_idx - start_idx)])
    extractor.SetIndex([int(s) for s in start_idx])
    out = extractor.Execute(img)
    return out


def ensure_4d_stack(mods: List[sitk.Image]) -> sitk.Image:
    """
    Stack modalities along a 4th dim so you can save/check them together if needed.
    """
    arrs = [sitk.GetArrayFromImage(m) for m in mods]
    arr4 = np.stack(arrs, axis=0)  # [C, Z, Y, X]
    img = sitk.GetImageFromArray(arr4)
    # Copy spatial info from first modality:
    img.CopyInformation(mods[0])
    return img


def find_case_files(case_dir: Path, modalities: List[str]) -> List[Path]:
    out = []
    for m in modalities:
        # Accept both "*-T1w.nii.gz" & "*_t1.nii.gz" styles
        patterns = [
            f"*{m}.nii.gz",
            f"*{m}.nii",
            f"*{m.lower()}.nii.gz",
            f"*{m.lower()}.nii",
        ]
        found = None
        for p in patterns:
            cand = list(case_dir.glob(p))
            if cand:
                found = cand[0]
                break
        if found is None:
            raise FileNotFoundError(f"Missing modality {m} in {case_dir}")
        out.append(found)
    return out


def preprocess_case(
    case_dir: Path,
    out_dir: Path,
    modalities: List[str],
    spacing: float = 1.0,
    roi_size_mm: Tuple[int, int, int] = (96, 96, 96),
    inferior_offset_mm: float = 20.0,
    save_mask: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load modalities
    paths = find_case_files(case_dir, modalities)
    imgs = [read_image(p) for p in paths]

    # N4 bias correction (per modality) with a shared mask derived from first modality for consistency
    msk = None
    first_for_mask = sitk.Cast(imgs[0], sitk.sitkFloat32)
    sm = sitk.SmoothingRecursiveGaussian(first_for_mask, 1.0)
    sm_abs = sitk.Cast(sitk.Abs(sm), sitk.sitkFloat32)
    msk = sitk.OtsuThreshold(sm_abs, 0, 1, 200)
    msk = sitk.BinaryMorphologicalOpening(msk, (1, 1, 1))

    imgs_n4 = [n4_bias_correct(im, mask=msk) for im in imgs]

    # Resample all to isotropic spacing
    imgs_rs = [resample_isotropic(im, out_spacing=spacing) for im in imgs_n4]
    msk_rs = resample_isotropic(
        msk, out_spacing=spacing, interpolator=sitk.sitkNearestNeighbor
    )

    # Compute ROI center & crop
    roi_center = compute_roi_center_world(
        imgs_rs[0], inferior_offset_mm=float(inferior_offset_mm)
    )
    imgs_crop = [crop_roi_world(im, roi_center, roi_size_mm) for im in imgs_rs]
    msk_crop = crop_roi_world(msk_rs, roi_center, roi_size_mm)

    # Z-score normalize per modality in ROI mask (use mask to avoid background)
    imgs_norm = [zscore(im, mask=msk_crop) for im in imgs_crop]

    # Save each modality back with conventional names
    name = case_dir.name
    for img, src_path in zip(imgs_norm, paths):
        # normalize suffix to *_<mod>.nii.gz (e.g., _T1w.nii.gz -> _t1.nii.gz)
        src = src_path.name
        # crude modality tag extraction
        tag = None
        for m in modalities:
            if m in src or m.lower() in src.lower():
                tag = m.lower()
                break
        tag = tag.replace("t1wce", "t1ce")  # common alias
        out_path = out_dir / name / f"{name}_{tag}.nii.gz"
        write_image(img, out_path)

    if save_mask:
        write_image(msk_crop, out_dir / name / f"{name}_roi_mask.nii.gz")


def main():
    ap = argparse.ArgumentParser(
        description="Craniopharyngioma MRI preprocessing pipeline."
    )
    ap.add_argument(
        "--in_dir",
        type=Path,
        required=True,
        help="Input directory with one folder per case.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for preprocessed cases.",
    )
    ap.add_argument(
        "--modalities",
        nargs="+",
        default=["T1w", "T1wCE", "T2w", "FLAIR"],
        help="List of modality tags to search per case (e.g., T1w T1wCE T2w FLAIR).",
    )
    ap.add_argument(
        "--spacing", type=float, default=1.0, help="Isotropic spacing (mm)."
    )
    ap.add_argument(
        "--roi_size_mm",
        nargs=3,
        type=int,
        default=[96, 96, 96],
        help="ROI size (mm) in X Y Z.",
    )
    ap.add_argument(
        "--inferior_offset_mm",
        type=float,
        default=20.0,
        help="Inferior shift from center (mm).",
    )
    ap.add_argument(
        "--save_mask", action="store_true", help="Save ROI mask used for z-scoring."
    )
    args = ap.parse_args()

    cases = [d for d in args.in_dir.iterdir() if d.is_dir()]
    if not cases:
        raise RuntimeError(f"No case folders found in {args.in_dir}")

    for c in sorted(cases):
        try:
            preprocess_case(
                c,
                args.out_dir,
                modalities=args.modalities,
                spacing=args.spacing,
                roi_size_mm=tuple(args.roi_size_mm),
                inferior_offset_mm=args.inferior_offset_mm,
                save_mask=args.save_mask,
            )
            print(f"[OK] {c.name}")
        except Exception as e:
            print(f"[FAIL] {c.name}: {e}")


if __name__ == "__main__":
    main()
