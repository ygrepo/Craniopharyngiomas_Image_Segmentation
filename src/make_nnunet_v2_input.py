#!/usr/bin/env python3
# import argparse, os, shutil, subprocess, sys, warnings
# from pathlib import Path
import argparse
import shutil
from pathlib import Path
import sys
from typing import List
import os
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def place(src: Path, dst: Path, mode: str, overwrite: bool):
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def build_case(case_dir: Path, dst_root: Path, mode: str, overwrite: bool) -> bool:
    """
    Make nnU-Net v2 4-channel inputs for one CASEID by duplicating _0002.
    Returns True if successful, else False.
    """
    case_id = case_dir.name
    t1ce = case_dir / f"{case_id}_0002.nii.gz"
    if not t1ce.exists():
        # fallback: try the descriptive filename and rely on user prep
        cand = list(case_dir.glob(f"{case_id}_T1_CE_*ALIGNED.nii.gz"))
        if cand:
            t1ce = cand[0]
        else:
            logger.info(
                f"[skip] {case_id}: no {case_id}_0002.nii.gz found", file=sys.stderr
            )
            return False

    out_case = dst_root / case_id
    out_case.mkdir(parents=True, exist_ok=True)

    # 0000=FLAIR, 0001=T1, 0002=T1ce, 0003=T2  (we duplicate T1ce)
    mapping = [
        (t1ce, out_case / f"{case_id}_0000.nii.gz"),
        (t1ce, out_case / f"{case_id}_0001.nii.gz"),
        (t1ce, out_case / f"{case_id}_0002.nii.gz"),  # as-is
        (t1ce, out_case / f"{case_id}_0003.nii.gz"),
    ]
    for src, dst in mapping:
        place(src, dst, mode, overwrite)

    logger.info(f"[ok] {case_id}: inputs at {out_case}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Build nnU-Net v2 inputs (duplicate T1-CE into missing channels) and optionally run prediction."
    )
    ap.add_argument(
        "--src_root",
        type=Path,
        required=True,
        help="Folder with per-case subdirs that contain <CASEID>_0002.nii.gz (your current output/nifti).",
    )
    ap.add_argument(
        "--dst_root",
        type=Path,
        required=True,
        help="Where to write nnU-Net input cases (<CASEID>_0000..0003.nii.gz).",
    )
    ap.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Create symlinks (default) or real copies.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing channel files if they exist.",
    )
    ap.add_argument(
        "--dataset_id",
        "-d",
        type=str,
        default="002",
        help="Dataset ID of the installed BraTS-21 model (logger.infoed by installer; often 002).",
    )
    ap.add_argument(
        "--out_pred",
        type=Path,
        default=Path("output/nnunet_out"),
        help="Output folder for nnU-Net predictions.",
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

    cases = [d for d in sorted(args.src_root.iterdir()) if d.is_dir()]
    if not cases:
        logger.info(f"No case folders under {args.src_root}", file=sys.stderr)
        sys.exit(1)

    ok = 0
    for cd in cases:
        ok += build_case(cd, args.dst_root, args.mode, args.overwrite)

    if ok == 0:
        logger.info("No cases prepared. Nothing to do.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
