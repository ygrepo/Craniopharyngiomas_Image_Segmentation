"""
Craniopharyngioma MRI preprocessing (NRRD -> NIfTI):
N4 bias correction → resample to isotropic spacing → ROI crop (sellar/suprasellar) → z-score normalization.

Usage (example):
    python craniopharyngioma_preprocess.py \
        --in_dir /path/to/raw_nrrd \
        --out_dir /path/to/preproc \
        --modalities T1w T1wCE T2w FLAIR \
        --spacing 1.0 \
        --roi_size_mm 96 96 96 \
        --inferior_offset_mm 20 \
        --save_mask

Notes
-----
- Input can be NRRD/NHDR (recommended) or NIfTI; outputs are always NIfTI (.nii.gz).
- ROI strategy (no atlas): we place a box around the image center, then shift inferiorly by
  `inferior_offset_mm` along the superior-inferior axis to capture sellar/suprasellar.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)

# ---------- I/O ----------


def read_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))  # NRRD/NHDR/NIfTI auto-detected


def write_image(img: sitk.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def find_case_files(case_dir: Path, modalities: List[str]) -> List[Path]:
    """
    Find one file per modality inside `case_dir`.
    Priority order per modality: NRRD/NHDR (including gz) first, then NIfTI.
    """
    out = []
    for m in modalities:
        patterns = [
            # NRRD/NHDR (common + gz)
            f"*{m}.nrrd",
            f"*{m}.nhdr",
            f"*{m}.nrrd.gz",
            f"*{m}.nhdr.gz",
            f"*{m.lower()}.nrrd",
            f"*{m.lower()}.nhdr",
            f"*{m.lower()}.nrrd.gz",
            f"*{m.lower()}.nhdr.gz",
            # NIfTI (fallback / mixed sets)
            f"*{m}.nii.gz",
            f"*{m}.nii",
            f"*{m.lower()}.nii.gz",
            f"*{m.lower()}.nii",
        ]
        found = None
        for p in patterns:
            cand = list(case_dir.glob(p))
            if cand:
                # if multiple matches, take the first in sorted order for determinism
                found = sorted(cand)[0]
                break
        if found is None:
            raise FileNotFoundError(f"Missing modality {m} in {case_dir}")
        out.append(found)
    return out


def find_mask_file(case_dir: Path, mask_tag: str) -> Optional[Path]:
    """
    Look for a provided tumor/lesion segmentation (labelmap).
    Example mask_tag: 'Tumor.seg' matches '*Tumor.seg.nrrd', '*Tumor.seg.nhdr', etc.
    """
    stems = [mask_tag, mask_tag.lower()]
    exts = [".nrrd", ".nhdr", ".nrrd.gz", ".nhdr.gz", ".nii.gz", ".nii"]
    for s in stems:
        for e in exts:
            cand = sorted(case_dir.glob(f"*{s}{e}"))
            if cand:
                return cand[0]
    return None


# ---------- PROCESSING ----------


def n4_bias_correct(
    img: sitk.Image,
    mask: sitk.Image | None = None,
    shrink_factor: int = 2,
    conv: int = 50,
) -> sitk.Image:
    """
    Force float32 image + uint8 mask for N4.
    """
    # N4 requires float input
    img_f = sitk.Cast(img, sitk.sitkFloat32)

    # Build/convert mask -> UInt8
    if mask is None:
        sm = sitk.SmoothingRecursiveGaussian(img_f, 1.0)
        sm_abs = sitk.Abs(sm)
        mask = sitk.OtsuThreshold(sm_abs, 0, 1, 200)
        mask = sitk.BinaryMorphologicalOpening(mask, (1, 1, 1))
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([conv])
    corrector.SetConvergenceThreshold(1e-6)

    # Speed-up pass (still float image + uint8 mask)
    img_shrunk = sitk.Shrink(img_f, [shrink_factor] * 3)
    mask_shrunk = sitk.Shrink(mask_u8, [shrink_factor] * 3)
    _ = corrector.Execute(img_shrunk, mask_shrunk)

    # Apply estimated field to the full-res float image
    log_field = corrector.GetLogBiasFieldAsImage(img_f)
    corrected = sitk.Exp(sitk.Log(img_f) - log_field)

    # Keep float32 downstream (recommended)
    return sitk.Cast(corrected, sitk.sitkFloat32)


def resample_isotropic(
    img: sitk.Image, out_spacing: float = 1.0, interpolator=sitk.sitkBSpline
) -> sitk.Image:
    original_spacing = np.array(list(img.GetSpacing()), dtype=float)
    original_size = np.array(list(img.GetSize()), dtype=int)

    new_spacing = np.array([out_spacing, out_spacing, out_spacing], dtype=float)
    new_size = np.maximum(
        1, np.round(original_size * (original_spacing / new_spacing)).astype(int)
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(img)


def resample_label_like(label: sitk.Image, reference: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(reference.GetSpacing())
    resampler.SetSize(reference.GetSize())
    resampler.SetOutputDirection(reference.GetDirection())
    resampler.SetOutputOrigin(reference.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(label)


def zscore(
    img: sitk.Image, mask: Optional[sitk.Image] = None, eps: float = 1e-8
) -> sitk.Image:
    im = sitk.Cast(img, sitk.sitkFloat32)
    m = (
        sitk.OtsuThreshold(im, 0, 1, 128)
        if mask is None
        else sitk.Cast(mask, sitk.sitkUInt8)
    )

    arr = sitk.GetArrayFromImage(im)
    marr = sitk.GetArrayFromImage(m).astype(bool)
    vals = arr[marr]
    if vals.size == 0:
        mean, std = float(np.mean(arr)), float(np.std(arr))
    else:
        v = np.clip(vals, np.percentile(vals, 1), np.percentile(vals, 99))
        mean, std = float(np.mean(v)), float(np.std(v) + eps)

    z = (arr - mean) / (std + eps)
    out = sitk.GetImageFromArray(z)
    out.CopyInformation(im)
    return sitk.Cast(out, sitk.sitkFloat32)


def mask_centroid_world(mask: sitk.Image) -> Tuple[float, float, float]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    # If multiple labels exist, merge by using all >0; centroid() expects a label id.
    # Use label=1 if present; otherwise take the first label id.
    lbls = list(stats.GetLabels())
    if 1 in lbls:
        c = stats.GetCentroid(1)
    else:
        c = stats.GetCentroid(lbls[0])
    return tuple(c)


def bbox_indices_from_mask(mask: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    arr = sitk.GetArrayFromImage(mask)  # z, y, x
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        # empty mask: whole image
        start = np.array([0, 0, 0])
        end = np.array(arr.shape)
    else:
        zyx_min = coords.min(axis=0)
        zyx_max = coords.max(axis=0) + 1  # end-exclusive
        start, end = zyx_min[::-1], zyx_max[::-1]  # to x,y,z
    return start.astype(int), end.astype(int)


def crop_roi_world(
    img: sitk.Image,
    center_world: Tuple[float, float, float],
    size_mm: Tuple[float, float, float],
) -> sitk.Image:
    spacing = np.array(img.GetSpacing(), float)
    size_vox = np.maximum(1, np.round(np.array(size_mm, float) / spacing)).astype(int)
    center_idx = np.array(img.TransformPhysicalPointToIndex(center_world), float)
    start_idx = np.round(center_idx - size_vox / 2.0).astype(int)
    end_idx = start_idx + size_vox
    start_idx = np.maximum(start_idx, 0)
    end_idx = np.minimum(end_idx, np.array(img.GetSize(), int))
    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetSize([int(s) for s in (end_idx - start_idx)])
    extractor.SetIndex([int(s) for s in start_idx])
    return extractor.Execute(img)


def crop_bbox_with_pad(
    img: sitk.Image, mask: sitk.Image, pad_mm: Tuple[float, float, float]
) -> sitk.Image:
    spacing = np.array(img.GetSpacing(), float)
    pad_vox = np.maximum(0, np.round(np.array(pad_mm, float) / spacing)).astype(int)

    # compute bbox in voxel indices (x,y,z)
    start_xyz, end_xyz = bbox_indices_from_mask(mask)
    start_xyz = np.maximum(0, start_xyz - pad_vox)
    end_xyz = np.minimum(np.array(img.GetSize(), int), end_xyz + pad_vox)

    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetIndex([int(i) for i in start_xyz])
    extractor.SetSize([int(i) for i in (end_xyz - start_xyz)])
    return extractor.Execute(img)


# ---------- Pipeline ----------


def preprocess_case(
    case_dir: Path,
    out_dir: Path,
    modalities: List[str],
    spacing: float = 1.0,
    roi_size_mm: Tuple[int, int, int] = (96, 96, 96),
    roi_from_mask: str = "centroid",  # 'centroid' or 'bbox'
    bbox_pad_mm: Tuple[float, float, float] = (8.0, 8.0, 8.0),
    mask_tag: str = "Tumor.seg",
    save_mask: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load modalities
    img_paths = find_case_files(case_dir, modalities)
    imgs = [read_image(p) for p in img_paths]
    logger.info(f"Found {len(imgs)} modalities: {[p.name for p in img_paths]}")

    # Load provided mask if present
    mask_path = find_mask_file(case_dir, mask_tag)
    provided_mask = read_image(mask_path) if mask_path is not None else None
    if provided_mask is not None:
        # Ensure binary/label type
        provided_mask = sitk.Cast(provided_mask > 0, sitk.sitkUInt8)

    logger.info(f"Found mask: {mask_path}")
    logger.info(f"Mask min: {sitk.GetArrayFromImage(provided_mask).min()}")
    logger.info(f"Mask max: {sitk.GetArrayFromImage(provided_mask).max()}")

    # N4 bias correction: use provided mask if available
    n4_mask = provided_mask
    imgs_n4 = [n4_bias_correct(im, mask=n4_mask) for im in imgs]

    # Resample images to isotropic
    imgs_rs = [resample_isotropic(im, out_spacing=spacing) for im in imgs_n4]

    # Resample mask to match resampled images (use first modality as reference)
    if provided_mask is not None:
        mask_rs = resample_label_like(provided_mask, imgs_rs[0])
    else:
        # If no provided mask, derive a foreground for z-score/cropping later
        sm = sitk.SmoothingRecursiveGaussian(
            sitk.Cast(imgs_rs[0], sitk.sitkFloat32), 1.0
        )
        sm_abs = sitk.Cast(sitk.Abs(sm), sitk.sitkFloat32)
        mask_rs = sitk.OtsuThreshold(sm_abs, 0, 1, 200)

    # Crop strategy
    if roi_from_mask == "bbox" and mask_rs is not None:
        logger.info(f"Cropping from bbox with pad {bbox_pad_mm}")
        imgs_crop = [crop_bbox_with_pad(im, mask_rs, bbox_pad_mm) for im in imgs_rs]
        mask_crop = crop_bbox_with_pad(mask_rs, mask_rs, bbox_pad_mm)
    else:
        logger.info(f"Cropping from centroid with size {roi_size_mm}")
        # default: fixed box centered at mask centroid (or image center if mask empty)
        if (
            sitk.StatisticsImageFilter().Execute(mask_rs)
            or sitk.GetArrayFromImage(mask_rs).any()
        ):
            center_world = mask_centroid_world(mask_rs)
        else:
            # fallback: image center
            size = np.array(imgs_rs[0].GetSize(), float)
            center_idx = (size - 1) / 2.0
            center_world = imgs_rs[0].TransformIndexToPhysicalPoint(
                [int(round(c)) for c in center_idx]
            )
        imgs_crop = [crop_roi_world(im, center_world, roi_size_mm) for im in imgs_rs]
        mask_crop = crop_roi_world(mask_rs, center_world, roi_size_mm)

    # Z-score per modality within mask crop (if provided)
    logger.info("Z-scoring within mask")
    imgs_norm = [zscore(im, mask=mask_crop) for im in imgs_crop]

    # Save as NIfTI
    name = case_dir.name
    for img, src_path in zip(imgs_norm, img_paths):
        # infer tag from filename based on provided modality list
        tag = None
        for m in modalities:
            if m in src_path.name or m.lower() in src_path.name.lower():
                tag = m.upper()
                break
        tag = (tag or "MOD").replace("T1WCE", "T1CE")
        out_path = out_dir / name / f"{name}_{tag}.nii.gz"
        logger.info(f"Saving to {out_path}")
        write_image(img, out_path)

    if save_mask and mask_crop is not None:
        out_mask_path = out_dir / name / f"{name}_mask.nii.gz"
        logger.info(f"Saving mask to {out_mask_path}")
        write_image(sitk.Cast(mask_crop, sitk.sitkUInt8), out_mask_path)


def main():
    ap = argparse.ArgumentParser(
        description="Craniopharyngioma MRI preprocessing (reads NRRD/NHDR/NIfTI; writes NIfTI)."
    )
    ap.add_argument(
        "--in_dir", type=Path, required=True, help="Input dir with one folder per case."
    )
    ap.add_argument(
        "--out_dir", type=Path, required=True, help="Output dir for preprocessed NIfTI."
    )
    ap.add_argument("--modalities", nargs="+", default=["T1w", "T1wCE", "T2w", "FLAIR"])
    ap.add_argument(
        "--spacing", type=float, default=1.0, help="Isotropic spacing (mm)."
    )
    ap.add_argument(
        "--roi_size_mm",
        nargs=3,
        type=float,
        default=[96, 96, 96],
        help="Fixed ROI size (mm) if using centroid mode.",
    )
    ap.add_argument(
        "--roi_from_mask",
        choices=["centroid", "bbox"],
        default="centroid",
        help="ROI strategy using provided mask: 'centroid' uses fixed-size box; 'bbox' uses tight box with padding.",
    )
    ap.add_argument(
        "--bbox_pad_mm",
        nargs=3,
        type=float,
        default=[8.0, 8.0, 8.0],
        help="Padding (mm) added around mask bounding box when roi_from_mask='bbox'.",
    )
    ap.add_argument(
        "--mask_tag",
        type=str,
        default="Tumor.seg",
        help="Substring/tag used to find the provided mask (e.g., 'Tumor.seg').",
    )
    ap.add_argument(
        "--save_mask",
        action="store_true",
        help="Save the (resampled/cropped) mask as NIfTI.",
    )
    ap.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Log file path (in addition to console).",
    )
    args = ap.parse_args()

    setup_logging(Path(args.log_file), args.log_level)

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
                roi_from_mask=args.roi_from_mask,
                bbox_pad_mm=tuple(args.bbox_pad_mm),
                mask_tag=args.mask_tag,
                save_mask=args.save_mask,
            )
            print(f"[OK] {c.name}")
        except Exception as e:
            print(f"[FAIL] {c.name}: {e}")


if __name__ == "__main__":
    main()

# def preprocess_case(
#     case_dir: Path,
#     out_dir: Path,
#     modalities: List[str],
#     spacing: float = 1.0,
#     roi_size_mm: Tuple[int, int, int] = (96, 96, 96),
#     inferior_offset_mm: float = 20.0,
#     save_mask: bool = False,
# ):
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Load modalities (NRRD/NHDR/NIfTI)
#     paths = find_case_files(case_dir, modalities)
#     imgs = [read_image(p) for p in paths]

#     # Shared mask from first modality for consistent N4
#     first_for_mask = sitk.Cast(imgs[0], sitk.sitkFloat32)
#     sm = sitk.SmoothingRecursiveGaussian(first_for_mask, 1.0)
#     sm_abs = sitk.Cast(sitk.Abs(sm), sitk.sitkFloat32)
#     msk = sitk.OtsuThreshold(sm_abs, 0, 1, 200)
#     msk = sitk.BinaryMorphologicalOpening(msk, (1, 1, 1))

#     imgs_n4 = [n4_bias_correct(im, mask=msk) for im in imgs]

#     # Resample to isotropic spacing
#     imgs_rs = [resample_isotropic(im, out_spacing=spacing) for im in imgs_n4]
#     msk_rs = resample_isotropic(
#         msk, out_spacing=spacing, interpolator=sitk.sitkNearestNeighbor
#     )

#     # ROI center & crop
#     roi_center = compute_roi_center_world(
#         imgs_rs[0], inferior_offset_mm=float(inferior_offset_mm)
#     )
#     imgs_crop = [crop_roi_world(im, roi_center, roi_size_mm) for im in imgs_rs]
#     msk_crop = crop_roi_world(msk_rs, roi_center, roi_size_mm)

#     # Z-score per modality (foreground mask)
#     imgs_norm = [zscore(im, mask=msk_crop) for im in imgs_crop]

#     # Save as NIfTI
#     name = case_dir.name
#     for img, src_path in zip(imgs_norm, paths):
#         tag = None
#         src = src_path.name
#         for m in modalities:
#             if m in src or m.lower() in src.lower():
#                 tag = m.lower()
#                 break
#         if tag is None:
#             # fallback if pattern matching fails
#             tag = "mod"
#         tag = tag.replace("t1wce", "t1ce")  # normalize a common alias
#         out_path = out_dir / name / f"{name}_{tag}.nii.gz"
#         write_image(img, out_path)

#     if save_mask:
#         write_image(msk_crop, out_dir / name / f"{name}_roi_mask.nii.gz")


# def main():
#     ap = argparse.ArgumentParser(
#         description="Craniopharyngioma MRI preprocessing pipeline (reads NRRD/NHDR/NIfTI, writes NIfTI)."
#     )
#     ap.add_argument(
#         "--in_dir",
#         type=Path,
#         required=True,
#         help="Input directory with one folder per case (NRRD/NHDR or NIfTI).",
#     )
#     ap.add_argument(
#         "--out_dir",
#         type=Path,
#         required=True,
#         help="Output directory for preprocessed NIfTI cases.",
#     )
#     ap.add_argument(
#         "--modalities",
#         nargs="+",
#         default=["T1w", "T1wCE", "T2w", "FLAIR"],
#         help="List of modality tags to search per case (e.g., T1w T1wCE T2w FLAIR).",
#     )
#     ap.add_argument(
#         "--spacing", type=float, default=1.0, help="Isotropic spacing (mm)."
#     )
#     ap.add_argument(
#         "--roi_size_mm",
#         nargs=3,
#         type=int,
#         default=[96, 96, 96],
#         help="ROI size (mm) in X Y Z.",
#     )
#     ap.add_argument(
#         "--inferior_offset_mm",
#         type=float,
#         default=20.0,
#         help="Inferior shift from center (mm).",
#     )
#     ap.add_argument(
#         "--save_mask", action="store_true", help="Save ROI mask used for z-scoring."
#     )
#     args = ap.parse_args()

#     cases = [d for d in args.in_dir.iterdir() if d.is_dir()]
#     if not cases:
#         raise RuntimeError(f"No case folders found in {args.in_dir}")

#     for c in sorted(cases):
#         try:
#             preprocess_case(
#                 c,
#                 args.out_dir,
#                 modalities=args.modalities,
#                 spacing=args.spacing,
#                 roi_size_mm=tuple(args.roi_size_mm),
#                 inferior_offset_mm=args.inferior_offset_mm,
#                 save_mask=args.save_mask,
#             )
#             print(f"[OK] {c.name}")
#         except Exception as e:
#             print(f"[FAIL] {c.name}: {e}")


# if __name__ == "__main__":
#     main()
