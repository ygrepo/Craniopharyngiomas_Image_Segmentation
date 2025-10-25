import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging, hd95_mm_from_binary_robust  # noqa: E402

logger = get_logger(__name__)


def main():
    setup_logging(None, "DEBUG")
    logger.info("Hello world!")
    # Replace with the exact pair that showed ∞ in your sheet
    path = "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/nnUNet_results/Dataset503_CP/EmaDiceEarlyStopTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/predictions/validation/70900351.nii.gz"
    pred = sitk.ReadImage(path)
    path = "/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/tmp_503_fold0_val/labelsTr/70900351.nii.gz"
    gt = sitk.ReadImage(path)
    print(hd95_mm_from_binary_robust(pred, gt, one_empty_policy="inf"))
    # Expect ~0.0 (if they fully overlap) or a small positive value
    # If you get ∞, something is wrong.
