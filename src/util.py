import os

import random

# import splitfolders
# from tqdm import tqdm
import nibabel as nib

# import glob
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


from matplotlib.colors import ListedColormap


def show_one_case(case_dir="data/BraTS2023-Glioma/BraTS-GLI-00000-000"):
    img_flair = nib.load(Path(case_dir, f"{Path(case_dir).name}-t2f.nii")).get_fdata()
    img_t1 = nib.load(Path(case_dir, f"{Path(case_dir).name}-t1n.nii")).get_fdata()
    img_t1ce = nib.load(Path(case_dir, f"{Path(case_dir).name}-t1c.nii")).get_fdata()
    img_t2 = nib.load(Path(case_dir, f"{Path(case_dir).name}-t2w.nii")).get_fdata()
    mask = (
        nib.load(Path(case_dir, f"{Path(case_dir).name}-seg.nii"))
        .get_fdata()
        .astype(np.uint8)
    )

    print("mask classes:", np.unique(mask))

    # pick a slice that actually contains tumor
    z_has_tumor = np.where(mask.any(axis=(0, 1)))[0]
    if z_has_tumor.size == 0:
        print("No nonzero slices in this case (unexpected).")
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
    m2d = mask[:, :, n_slice]

    # discrete colormap for labels (edit colors as you like)
    # assuming labels {0,1,2,3}; index 0 = transparent/black
    cmap = ListedColormap(["black", "#1f77b4", "#2ca02c", "#d62728"])

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.imshow(flair2d, cmap="gray")
    plt.title("Image flair")
    plt.axis("off")
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
    plt.imshow(m2d, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    plt.title("Seg Mask")
    plt.axis("off")

    # overlay on one modality
    plt.subplot(236)
    plt.imshow(flair2d, cmap="gray")
    plt.imshow(m2d, cmap=cmap, vmin=0, vmax=3, alpha=0.35, interpolation="nearest")
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    seed_everything(42)
    show_one_case()
    # investigate_data()
