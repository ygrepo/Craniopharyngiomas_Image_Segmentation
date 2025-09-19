#!/usr/bin/env python3
import os, json, argparse, glob, sys
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nii_root",
        type=Path,
        required=True,
        help="Root with case subfolders (e.g., output/preprocessed)",
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Where to write per-slice PNGs and metadata",
    )
    ap.add_argument(
        "--prefer",
        type=str,
        default="t1*ce*aligned*.nii.gz",
        help="Glob (case-insensitive) to pick the image per case",
    )
    # logging
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
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)

    cases = sorted([d for d in args.nii_root.iterdir() if d.is_dir()])
    for c in cases:
        pats = [p for p in c.glob("*.nii.gz")] + [p for p in c.glob("*.nii")]
        img_p = None
        patt = args.prefer.lower()
        for p in pats:
            if Path(p).name.lower().find(patt.replace("*", "")) >= 0:
                img_p = p
                break
        if img_p is None and pats:
            img_p = sorted(pats)[0]
        if img_p is None:
            logger.warning(f"[skip] {c.name}: no nifti")
            continue

        out_dir = args.out_root / c.name / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        nii = nib.load(str(img_p))
        vol = nii.get_fdata().astype(np.float32)
        X, Y, Z = vol.shape
        affine = nii.affine
        axcodes = nib.aff2axcodes(affine)  # e.g. ('L','P','S')

        # simple per-slice min–max to [0,1] → uint8 PNG
        for z in range(Z):
            sl = vol[..., z]
            vmin, vmax = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
            sl = (np.clip(sl, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)
            png = (sl * 255).astype(np.uint8)
            png3 = np.stack([png] * 3, axis=-1)  # 3-ch expected at inference
            cv2.imwrite(str(out_dir / f"{z:04d}.png"), png3)

        # store geometry to rebuild
        meta = {
            "nifti_path": str(img_p),
            "shape_xyz": [int(X), int(Y), int(Z)],
            "affine": affine.tolist(),
            "axcodes": list(axcodes),  # for sanity checks
            "export_axis": 2,  # we exported along Z
        }
        (args.out_root / c.name).mkdir(parents=True, exist_ok=True)
        with open(args.out_root / c.name / "meta.json", "w") as f:
            json.dump(meta, f)
        logger.info(f"[ok] {c.name}: {Z} slices → {out_dir}")


if __name__ == "__main__":
    main()
