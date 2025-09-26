import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging
from src.preprocess import preprocess_case

logger = get_logger(__name__)

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