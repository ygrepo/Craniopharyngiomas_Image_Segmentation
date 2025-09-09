import os

import random
import SimpleITK as sitk

# import splitfolders
from tqdm import tqdm
import nibabel as nib

import glob
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# from sklearn.model_selection import train_test_split
# import shutil
# import time

from dataclasses import dataclass

import torch

# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import torchvision.transforms as transforms
# from torch.cuda import amp

# from torchmetrics import MeanMetric
# from torchmetrics.classification import MulticlassAccuracy

# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchinfo import summary
# import gc

# import segmentation_models_pytorch_3d as smp

# from livelossplot import PlotLosses
# from livelossplot.outputs import MatplotlibPlot, ExtremaPrinter

import splitfolders
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_device():
    gpu_available = torch.cuda.is_available()
    return torch.device("cuda" if gpu_available else "cpu"), gpu_available


@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 5
    EPOCHS: int = 100
    LEARNING_RATE: float = 1e-3
    CHECKPOINT_DIR: str = os.path.join("model_checkpoint", "3D_UNet_Brats2023")
    NUM_WORKERS: int = 4


def show_one_case(
    case_dir: Path = Path("data/BraTS2023-Glioma/BraTS-GLI-00000-000"), show=True
):
    img_flair = nib.load(Path(case_dir, f"{Path(case_dir).name}-t2f.nii")).get_fdata()
    img_t1 = nib.load(Path(case_dir, f"{Path(case_dir).name}-t1n.nii")).get_fdata()
    img_t1ce = nib.load(Path(case_dir, f"{Path(case_dir).name}-t1c.nii")).get_fdata()
    img_t2 = nib.load(Path(case_dir, f"{Path(case_dir).name}-t2w.nii")).get_fdata()

    mask = (
        nib.load(Path(case_dir, f"{Path(case_dir).name}-seg.nii"))
        .get_fdata()
        .astype(np.uint8)
    )

    # pick a slice that actually contains tumor
    z_has_tumor = np.where(mask.any(axis=(0, 1)))[0]
    if z_has_tumor.size == 0:
        logger.info("No nonzero slices in this case (unexpected).")
        return
    n_slice = z_has_tumor[len(z_has_tumor) // 2]  # middle tumor slice

    # nicer display: clip to percentiles per modality
    def norm2d(x):
        lo, hi = np.percentile(x, (1, 99))
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo + 1e-6)

    flair2d = norm2d(img_flair[:, :, n_slice])
    t12d = norm2d(img_t1[:, :, n_slice])
    t1ce2d = norm2d(img_t1ce[:, :, n_slice])
    t22d = norm2d(img_t2[:, :, n_slice])

    combined_x = np.stack(
        [flair2d, t12d, t22d], axis=2
    )  # along the last channel dimension.
    combined_x = combined_x[56:184, 56:184, :]  # keep all 3 channels

    # logger.info(f"Shape of Combined x {combined_x.shape}")
    # combined_x = combined_x[56:184, 56:184, 13:141]
    # logger.info(f"Shape after cropping: {combined_x.shape}")

    # logger.info(f"mask classes: {np.unique(mask)}")
    # logger.info(f"Mask shape before cropping: {mask.shape}")
    # mask = mask[56:184, 56:184, 13:141]
    # logger.info(f"Mask shape after cropping: {mask.shape}")

    mask = mask[:, :, n_slice]

    if not show:
        return combined_x, mask

    # discrete colormap for labels (edit colors as you like)
    # assuming labels {0,1,2,3}; index 0 = transparent/black
    cmap = ListedColormap(["black", "#1f77b4", "#2ca02c", "#d62728"])

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.imshow(flair2d, cmap="gray")
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=3, alpha=0.35, interpolation="nearest")
    plt.title("Overlay")
    plt.axis("off")

    # plt.subplot(231)
    # plt.imshow(flair2d, cmap="gray")
    # plt.title("Image flair")
    # plt.axis("off")

    plt.subplot(232)
    plt.imshow(t12d, cmap="gray")
    plt.title("Image t1")
    plt.axis("off")
    plt.subplot(233)
    plt.imshow(t1ce2d, cmap="gray")
    plt.title("Image t1ce")
    plt.axis("off")
    plt.subplot(234)
    plt.imshow(t22d, cmap="gray")
    plt.title("Image t2")
    plt.axis("off")

    # mask alone (discrete)
    plt.subplot(235)
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    plt.title("Seg Mask")
    plt.axis("off")

    # overlay on one modality
    # plt.subplot(236)
    # plt.imshow(flair2d, cmap="gray")
    # plt.imshow(mask, cmap=cmap, vmin=0, vmax=3, alpha=0.35, interpolation="nearest")
    # plt.title("Overlay")
    # plt.axis("off")

    plt.subplot(236)
    plt.imshow(np.mean(combined_x, axis=2), cmap="gray")
    plt.title("Combined X")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ---------- N4 BIAS FIELD CORRECTION (optional) ----------
def n4_bias_correct_np(
    x: np.ndarray,
    brain_mask: np.ndarray | None = None,
    shrink_factor: int = 2,
    n_iters: int = 50,
) -> np.ndarray:
    """
    Run N4 bias correction on a 3D NumPy array (float32).
    If brain_mask is None, uses Otsu to estimate one.
    """

    img = sitk.GetImageFromArray(x.astype(np.float32))  # zyx orientation inside SITK
    if brain_mask is None:
        # Rough brain mask via Otsu (works well enough for bias correction)
        brain_mask = sitk.OtsuThreshold(img, 0, 1, 200)
    else:
        brain_mask = sitk.GetImageFromArray(brain_mask.astype(np.uint8))

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(
        [n_iters]
    )  # one level is fine for many cases
    corrector.SetShrinkFactor(shrink_factor)  # speed-up; 2 or 4 are common

    corrected = corrector.Execute(img, brain_mask)
    out = sitk.GetArrayFromImage(corrected).astype(np.float32)
    return out


# ---------- your existing robust normalization helpers ----------
def robust_scale_volume(x: np.ndarray, lo_p=1, hi_p=99, per_slice=False) -> np.ndarray:
    x = x.astype(np.float32)
    if per_slice:
        zdim = x.shape[2]
        out = np.empty_like(x, dtype=np.float32)
        for z in range(zdim):
            sl = x[:, :, z]
            lo, hi = np.percentile(sl, (lo_p, hi_p))
            sl = np.clip(sl, lo, hi)
            out[:, :, z] = (sl - lo) / (hi - lo + 1e-6)
        return out
    else:
        lo, hi = np.percentile(x, (lo_p, hi_p))
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo + 1e-6)


def zscore_in_mask(
    x: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
    clip_sigma: float | None = 3.0,
    remap01: bool = True,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = mask.astype(bool)
    if not np.any(m):
        mu = float(np.mean(x))
        sd = float(np.std(x) + eps)
    else:
        mu = float(np.mean(x[m]))
        sd = float(np.std(x[m]) + eps)
    z = (x - mu) / sd
    if clip_sigma is not None:
        z = np.clip(z, -clip_sigma, clip_sigma)
    if remap01:
        if clip_sigma is None:
            z_min, z_max = float(z.min()), float(z.max())
            z = (z - z_min) / (z_max - z_min + eps)
        else:
            z = (z + clip_sigma) / (2 * clip_sigma)
    return z.astype(np.float32)


# ---------- main preprocess with optional N4 ----------
def preprocess(
    data_path: Path = Path("data/BraTS2023-Glioma"),
    output_path: Path = Path("BraTS2023_Preprocessed"),
    *,
    do_n4: bool = False,  # toggle this ON to enable N4
    n4_shrink: int = 2,
    n4_iters: int = 50,
    lo_p: int = 1,
    hi_p: int = 99,
    per_slice: bool = False,
    do_zscore: bool = True,
    clip_sigma: float | None = 3.0,
    remap01_after_z: bool = True,
):
    t1ce_list = sorted(glob.glob(f"{data_path}/*/*t1c.nii"))
    t2_list = sorted(glob.glob(f"{data_path}/*/*t2w.nii"))
    flair_list = sorted(glob.glob(f"{data_path}/*/*t2f.nii"))
    mask_list = sorted(glob.glob(f"{data_path}/*/*seg.nii"))

    logger.info(f"t1ce list: {len(t1ce_list)}")
    logger.info(f"t2 list: {len(t2_list)}")
    logger.info(f"flair list: {len(flair_list)}")
    logger.info(f"Mask list: {len(mask_list)}")

    out_img_dir = output_path / Path("input_data_3channels/images")
    out_msk_dir = output_path / Path("input_data_3channels/masks")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(
        range(len(t2_list)), desc="Preparing to N4/normalize/crop/save", unit="file"
    ):
        # --- load ---
        t1ce = nib.load(t1ce_list[idx]).get_fdata().astype(np.float32)
        t2 = nib.load(t2_list[idx]).get_fdata().astype(np.float32)
        flair = nib.load(flair_list[idx]).get_fdata().astype(np.float32)
        seg = nib.load(mask_list[idx]).get_fdata().astype(np.uint8)

        # --- optional N4 bias correction (per modality) ---
        if do_n4:
            # You can pass a rough mask for speed/robustness; here we let Otsu estimate it.
            t1ce = n4_bias_correct_np(
                t1ce, brain_mask=None, shrink_factor=n4_shrink, n_iters=n4_iters
            )
            t2 = n4_bias_correct_np(
                t2, brain_mask=None, shrink_factor=n4_shrink, n_iters=n4_iters
            )
            flair = n4_bias_correct_np(
                flair, brain_mask=None, shrink_factor=n4_shrink, n_iters=n4_iters
            )

        # --- robust percentile scaling to [0,1] ---
        t1ce = robust_scale_volume(t1ce, lo_p=lo_p, hi_p=hi_p, per_slice=per_slice)
        t2 = robust_scale_volume(t2, lo_p=lo_p, hi_p=hi_p, per_slice=per_slice)
        flair = robust_scale_volume(flair, lo_p=lo_p, hi_p=hi_p, per_slice=per_slice)

        # --- brain mask from union of nonzero voxels after clipping ---
        brain_mask = (flair > 0) | (t1ce > 0) | (t2 > 0)

        # --- z-score within brain mask, then (optionally) clip & map to [0,1] ---
        if do_zscore:
            t1ce = zscore_in_mask(
                t1ce, brain_mask, clip_sigma=clip_sigma, remap01=remap01_after_z
            )
            t2 = zscore_in_mask(
                t2, brain_mask, clip_sigma=clip_sigma, remap01=remap01_after_z
            )
            flair = zscore_in_mask(
                flair, brain_mask, clip_sigma=clip_sigma, remap01=remap01_after_z
            )

        # --- stack channels last: (H, W, Z, 3) in order [FLAIR, T1CE, T2] ---
        X = np.stack([flair, t1ce, t2], axis=3).astype(np.float32)

        # --- crop spatial + Z; KEEP channels ---
        X = X[56:184, 56:184, 13:141, :]
        seg = seg[56:184, 56:184, 13:141]

        # --- skip mostly-empty masks (<=1% foreground) safely ---
        vals, counts = np.unique(seg, return_counts=True)
        total = int(counts.sum())
        bg = int(counts[vals.tolist().index(0)]) if 0 in vals else 0
        fg_ratio = 1 - (bg / total) if total > 0 else 0.0
        if fg_ratio <= 0.01:
            continue

        # --- save ---
        np.save(out_img_dir / f"image_{idx}.npy", X)  # (128,128,128,3) float32
        np.save(out_msk_dir / f"mask_{idx}.npy", seg)  # (128,128,128)  uint8 {0..3}


def split_preprocessed_data(
    input_folder: Path = Path("data/BraTS2023_Preprocessed/input_data_3channels/"),
    output_folder: Path = Path("data/BraTS2023_Preprocessed/input_data_128/"),
    seed: int = 42,
    split_ratio: tuple = (0.75, 0.25),
    strict_check: bool = True,
):
    """
    Split BraTS preprocessed data into train/val(/test) while ensuring that
    'images/' and 'masks/' subfolders are aligned (image_x ↔ mask_x).

    Parameters
    ----------
    input_folder : Path
        Path to the preprocessed input data folder (must contain 'images/' and 'masks/').
    output_folder : Path
        Path where the split dataset will be saved.
    seed : int
        Random seed for reproducibility.
    split_ratio : tuple
        Ratio for splitting (train, val) or (train, val, test).
        Examples:
            (0.75, 0.25)   → train 75%, val 25%
            (0.7, 0.2, 0.1) → train 70%, val 20%, test 10%
    strict_check : bool
        If True, raise an error when image/mask filenames don't match.
        If False, just warn and continue.
    """

    img_dir = input_folder / "images"
    msk_dir = input_folder / "masks"

    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(
            f"Expected 'images/' and 'masks/' subfolders in {input_folder}"
        )

    # --- 1. Collect filenames (without extensions) ---
    img_files = {f.stem.replace("image_", "") for f in img_dir.glob("*.npy")}
    msk_files = {f.stem.replace("mask_", "") for f in msk_dir.glob("*.npy")}

    # --- 2. Check alignment ---
    only_imgs = img_files - msk_files
    only_msks = msk_files - img_files

    if only_imgs or only_msks:
        msg = f"Mismatched pairs found!\nOnly images: {sorted(only_imgs)}\nOnly masks: {sorted(only_msks)}"
        if strict_check:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    else:
        logger.info(f"All {len(img_files)} image/mask pairs are aligned.")

    # --- 3. Run split ---
    splitfolders.ratio(
        input_folder,
        output=output_folder,
        seed=seed,
        ratio=split_ratio,
        group_prefix=None,  # keeps pairs together
        move=False,  # copy instead of move
    )
    logger.info(f"✅ Dataset split saved in: {output_folder}")


# Example usage:
# split_preprocessed_data()


if __name__ == "__main__":
    seed_everything(42)
    # show_one_case()
    # preprocess()
    split_preprocessed_data()
