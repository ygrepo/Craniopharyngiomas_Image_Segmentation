import logging
import sys
from pathlib import Path
from typing import List, Optional
import SimpleITK as sitk
import numpy as np


# ---- One base for everything ----
BASE_LOGGER = "base_logger"
_BASE = logging.getLogger(BASE_LOGGER)  # the only logger we configure here


def setup_logging(log_path: str | Path | None, level: str = "INFO") -> logging.Logger:
    """Configure the base logger once (file + console)."""
    if getattr(_BASE, "_configured", False):
        return _BASE

    _BASE.handlers.clear()
    _BASE.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Optional file handler
    if log_path:
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        _BASE.addHandler(fh)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    _BASE.addHandler(sh)

    # Do not bubble to the *root* logger
    _BASE.propagate = False
    _BASE._configured = True
    return _BASE


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger that inherits the base handlers."""
    return logging.getLogger(BASE_LOGGER if not name else f"{BASE_LOGGER}.{name}")


# Convenience logger for this module
logger = get_logger(__name__)


# ---------- I/O ----------


def read_image(path: Path) -> sitk.Image:
    logger.info(f"Reading image: {path}")
    return sitk.ReadImage(str(path))  # NRRD/NHDR/NIfTI auto-detected


def write_image(img: sitk.Image, path: Path):
    logger.info(f"Writing image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetDirection(), b.GetDirection())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
    )


def find_case_files(case_dir: Path, modalities: List[str]) -> List[Path]:
    """
    Find one file per modality inside `case_dir`.
    Priority order per modality: NRRD/NHDR (including gz) first, then NIfTI.
    """
    out = []
    for m in modalities:
        logger.info(f"Searching for {m} in {case_dir}")
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
    logger.info(f"Searching for mask '{mask_tag}' in {case_dir}")
    stems = [mask_tag, mask_tag.lower()]
    exts = [".nrrd", ".nhdr", ".nrrd.gz", ".nhdr.gz", ".nii.gz", ".nii"]
    for s in stems:
        for e in exts:
            cand = sorted(case_dir.glob(f"*{s}{e}"))
            if cand:
                return cand[0]
    return None


# ---------- Optional N4 ----------
def n4_bias_correct_np(x: np.ndarray, shrink: int = 2, n_iters: int = 50) -> np.ndarray:
    img = sitk.GetImageFromArray(x.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetShrinkFactor(shrink)
    n4.SetMaximumNumberOfIterations([n_iters])
    out = n4.Execute(img, mask)
    return sitk.GetArrayFromImage(out).astype(np.float32)


def strip_ext(p: Path) -> str:
    s = p.name
    if s.endswith(".nii.gz"):
        return s[:-7]
    if s.endswith(".nii"):
        return s[:-4]
    return s
