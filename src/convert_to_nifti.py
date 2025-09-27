#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
from typing import List
import re
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
    find_case_files,
    find_mask_file,
    read_image,
    write_image,
    same_geometry,
)

logger = get_logger(__name__)


def convert_to_nifti(
    case_dir: Path, out_root: Path, modalities: List[str], mask_tag: str = "Tumor.seg"
):
    # Load modalities
    img_paths = find_case_files(case_dir, modalities)
    imgs = [read_image(p) for p in img_paths]
    logger.info(f"Found {len(imgs)} modalities: {[p.name for p in img_paths]}")

    # Load provided mask if present
    mask_path = find_mask_file(case_dir, mask_tag)
    provided_mask = read_image(mask_path) if mask_path is not None else None
    if provided_mask is not None:
        provided_mask = sitk.Cast(provided_mask > 0, sitk.sitkUInt8)
        logger.info(f"Found mask: {mask_path}")
        arr = sitk.GetArrayFromImage(provided_mask)
        logger.info(f"Mask min: {float(arr.min())} | max: {float(arr.max())}")
    else:
        logger.info("Found mask: None")

    case_id = case_dir.name
    out_case = out_root / case_id
    out_case.mkdir(parents=True, exist_ok=True)

    # --- detect T1-CE among provided modalities ---
    def is_t1ce_name(name: str) -> bool:
        # common tags for T1-CE
        return bool(re.search(r"(t1[_\- ]?ce|t1ce|post)", name, flags=re.IGNORECASE))

    # If user passed only one modality and it’s T1-CE by intent
    user_wants_t1ce = any(m.upper() in ("T1_CE", "T1CE", "POST") for m in modalities)

    t1ce_img = None
    t1ce_written_path = None

    # --- save each modality as NIfTI next to out_case ---
    for p, img in zip(img_paths, imgs):
        # convert NRRD→NIfTI if needed by changing suffix
        out_img_path = out_case / (
            p.stem + ".nii.gz" if not p.name.endswith(".nii.gz") else p.name
        )
        write_image(img, out_img_path)
        t1ce_native_path = None
        # remember T1-CE (by filename heuristic or by user intent if only one)
        if is_t1ce_name(p.name) or (len(img_paths) == 1 and user_wants_t1ce):
            t1ce_img = img
            # nnU-Net alias for T1ce channel
            t1ce_alias = out_case / f"{case_id}_0002.nii.gz"
            write_image(img, t1ce_alias)
            t1ce_written_path = t1ce_alias
            t1ce_native_path = out_img_path

    # Fallback: if we still did not identify T1-CE, use the first modality as reference
    if t1ce_img is None and imgs:
        t1ce_img = imgs[0]
        t1ce_alias = out_case / f"{case_id}_0002.nii.gz"
        write_image(t1ce_img, t1ce_alias)
        t1ce_written_path = t1ce_alias
        logger.warning(
            f"{case_id}: T1-CE not detected from names; using first modality as T1-CE for nnU-Net alias."
        )

    # Also write a Slicer-friendly alias for T1-CE (for your downstream tools)
    if t1ce_img is not None:
        slicer_alias = out_case / f"{case_id}_T1_CE_3D_AX_ALIGNED.nii.gz"
        if t1ce_native_path is None or slicer_alias != t1ce_native_path:

            write_image(t1ce_img, slicer_alias)

    # --- save mask aligned to T1-CE geometry (if provided) ---
    if provided_mask is not None and t1ce_img is not None:
        # resample if geometry differs (nearest-neighbor for labels)
        if not same_geometry(provided_mask, t1ce_img):
            provided_mask = sitk.Resample(
                provided_mask,
                t1ce_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
        mask_out = out_case / f"{case_id}_mask.nii.gz"
        provided_mask = sitk.Cast(provided_mask, sitk.sitkUInt8)
        write_image(provided_mask, mask_out)
        logger.info(
            f"[ok] {case_id}: wrote mask {mask_out.name} "
            f"(T1-CE ref: {t1ce_written_path.name if t1ce_written_path else 'n/a'})"
        )
    else:
        logger.info(f"[ok] {case_id}: no mask to save.")


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
            convert_to_nifti(
                c,
                args.out_dir,
                args.modalities,
            )
            logger.info(f"[OK] {c.name}")
        except Exception as e:
            logger.error(f"[FAIL] {c.name}: {e}")


if __name__ == "__main__":
    main()
