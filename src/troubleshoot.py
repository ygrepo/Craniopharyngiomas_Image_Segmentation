import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import SimpleITK as sitk
import numpy as np
import blosc2 as b2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
    hd95_mm_from_binary,
    load_mask_bool,
)  # noqa: E402

logger = get_logger(__name__)


def main():
    setup_logging(None, "DEBUG")
    # # Replace with the exact pair that showed ∞ in your sheet
    # path = "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/nnUNet_results/Dataset503_CP/EmaDiceEarlyStopTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/predictions/validation/70900351.nii.gz"
    # pred = sitk.ReadImage(path)
    # path = "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/tmp_503_fold0_val/labelsTr/70900351.nii.gz"
    # gt = sitk.ReadImage(path)
    # logger.info(hd95_mm_from_binary(pred, gt, one_empty_policy="inf"))
    # # Expect ~0.0 (if they fully overlap) or a small positive value
    # # If you get ∞, something is wrong.

    # path = "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/nnUNet_results/Dataset503_CP/EmaDiceEarlyStopTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/predictions/validation/52435303.nii.gz"
    # pred = sitk.ReadImage(path)
    # path = "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/tmp_503_fold0_val/labelsTr/52435303.nii.gz"
    # gt = sitk.ReadImage(path)
    # logger.info(hd95_mm_from_binary(pred, gt, one_empty_policy="inf"))
    # path = "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/nnUNet_results/Dataset503_CP/EmaDiceEarlyStopTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/predictions/validation/75062101.nii.gz"
    # pred = sitk.ReadImage(path)
    # path = "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/tmp_503_fold0_val/labelsTr/75062101.nii.gz"
    # gt = sitk.ReadImage(path)
    # logger.info(hd95_mm_from_binary(pred, gt, one_empty_policy="inf"))

    # path = Path(
    #     "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/data/CP/75062101/75062101_Tumor.seg.nrrd"
    # )
    # mask_bool, img = load_mask_bool(path)

    # voxel_count = int(mask_bool.sum())
    # spacing = img.GetSpacing()  # (sx, sy, sz) in mm
    # size = img.GetSize()  # (nx, ny, nz)
    # origin = img.GetOrigin()

    # vol_mm3 = voxel_count * float(spacing[0] * spacing[1] * spacing[2])
    # vol_ml = vol_mm3 / 1000.0

    # raw_vals = np.unique(sitk.GetArrayFromImage(img))

    # logger.info(f"Mask voxel count: {voxel_count}")
    # logger.info(f"Unique raw values in file: {raw_vals[:10]}")
    # logger.info(f"Size: {size}, Spacing: {spacing}, Origin: {origin}")
    # logger.info(f"Physical volume: {vol_ml:.3f} mL")

    seg_path = Path(
        "../nnUNet_preprocessed/Dataset503_CP/nnUNetPlans_3d_fullres/75062101_seg.b2nd"
    )
    seg_b2 = b2.open(seg_path)  # Blosc2 NDArray
    seg = np.asarray(seg_b2[:])  # to NumPy
    logger.info(f"seg shape: {seg.shape}, dtype: {seg.dtype}")
    logger.info(f"unique: {np.unique(seg)}, nonzero: {np.count_nonzero(seg)}")


if __name__ == "__main__":
    main()
