from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import SimpleITK as sitk
import numpy as np
import pickle
import nibabel as nib
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
    inv_ornt_aff,
)

from typing import Dict, Any
import torch
from torch.serialization import add_safe_globals
import nnunetv2
import json
import torch.nn as nn

import random
import re
import blosc2
import scipy.ndimage as ndi
import math

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunetv2.utilities.label_handling.label_handling import LabelManager

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)

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


def save_npy(ar: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, ar)
    logger.info(f"[ok] Saved: {out_path}")


def load_volume_npy_b2nd(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return load_npy(path)
    # assume blosc2 .b2nd
    arr = blosc2.open(str(path))
    return np.asarray(arr)


def load_npy(path: Path) -> np.ndarray:
    logger.info(f"Loading: {path}")
    return np.load(path)


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetDirection(), b.GetDirection())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
    )


def pick_vis_channel(volume: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Returns:
      vol3d: (D,H,W) selected grayscale channel
      vis_ch: integer channel index used (0-based); -1 if input was already (D,H,W)
    """
    arr = volume
    # Squeeze leading batch if present
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]  # -> (C,D,H,W)

    # Already single-channel
    if arr.ndim == 3:
        return arr, -1

    # Handle channel-first (C,D,H,W) or channel-last (D,H,W,C)
    if arr.ndim == 4:
        # Prefer explicit detection of channel axis
        # Heuristic: channel dim is the one with the smallest size among the 4 dims if <= 16
        # (works for medical volumes where D/H/W are large)
        dims = arr.shape
        candidate_axes = [i for i, s in enumerate(dims) if s <= 16]
        if candidate_axes:
            ch_axis = min(candidate_axes, key=lambda i: dims[i])
        else:
            # Fallback: if first dim looks like channels (<=16), assume (C,D,H,W),
            # else if last dim looks like channels, assume (D,H,W,C),
            # else default to channel-first.
            ch_axis = 0 if dims[0] <= 16 else (3 if dims[-1] <= 16 else 0)

        # Move channels to axis 0
        if ch_axis != 0:
            arr = np.moveaxis(arr, ch_axis, 0)  # -> (C,D,H,W)

        C = arr.shape[0]
        vis_ch = 2 if C > 2 else 0
        if vis_ch >= C:
            vis_ch = C - 1  # guard

        return arr[vis_ch], vis_ch

    raise ValueError(
        f"Unsupported shape {arr.shape}; expected (D,H,W), (C,D,H,W), (1,C,D,H,W), or (D,H,W,C)"
    )


def pick_channel_like(arr: np.ndarray, vis_ch: int) -> np.ndarray:
    """
    Pick the same channel as the base image.
    arr: (1,C,D,H,W) or (C,D,H,W) or (D,H,W)
    vis_ch: channel index used for the base image
    Returns (D,H,W)
    """
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]  # (C,D,H,W)
    if arr.ndim == 4:
        C = arr.shape[0]
        if not (0 <= vis_ch < C):
            raise ValueError(f"vis_ch={vis_ch} out of range for C={C}")
        return arr[vis_ch]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported shape {arr.shape}")


def normalize_heat_abs(delta_3d: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """Normalize |delta| to [0,1] using robust percentile."""
    a = np.abs(delta_3d).astype(np.float32)
    hi = np.percentile(a, pct)
    if hi <= 1e-12:
        hi = 1e-6
    a = np.clip(a / hi, 0.0, 1.0)
    return a


# --- add helpers (place near other helpers) ---
def load_props_from_pkl(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        logger.info(f"Loading props from {pkl_path}")
        props = pickle.load(f)
    return props


def get_spacing_origin_from_props(
    props: dict,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # nnU-Net v2 common keys; fallbacks are safe
    spacing = tuple(
        props.get("spacing_after_resampling")
        or props.get("spacing")
        or props.get("itk_spacing")
        or (1.0, 1.0, 1.0)
    )
    origin = tuple(props.get("origin") or props.get("itk_origin") or (0.0, 0.0, 0.0))
    # ensure 3 floats
    logger.info(f"Origin: {origin} → {origin[:3]}")
    logger.info(f"Spacing: {spacing} → {spacing[:3]}")
    spacing = tuple(float(x) for x in spacing[:3])
    origin = tuple(float(x) for x in origin[:3])
    return spacing, origin


def affine_from_spacing_origin(
    spacing: tuple[float, float, float], origin: tuple[float, float, float]
) -> np.ndarray:
    # RAS-ish diagonal; if your pipeline uses a specific orientation, swap signs/axes here
    aff = np.eye(4, dtype=np.float32)
    aff[0, 0], aff[1, 1], aff[2, 2] = spacing
    aff[0, 3], aff[1, 3], aff[2, 3] = origin
    logger.info(f"Affine:\n{aff}")
    return aff


def save_nifti_3d(
    vol_DHW: np.ndarray, affine: np.ndarray, out_path: Path, dtype=np.float32
):
    vol = np.asarray(vol_DHW, dtype=dtype, order="F")  # F-order is fine for nibabel
    img = nib.Nifti1Image(vol, affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {out_path}")
    nib.save(img, str(out_path))


def normalize_robust(img_3d: np.ndarray) -> np.ndarray:
    base = img_3d.astype(np.float32, copy=False)
    lo, hi = np.percentile(base, 1), np.percentile(base, 99)
    if hi <= lo:
        hi = lo + 1e-6
    base = (base - lo) / (hi - lo)
    return np.clip(base, 0.0, 1.0)


def build_heat_and_mask(delta_3d: np.ndarray, abs_pct: float, bin_pct: float | None):
    """Return (heat_0_1, mask_or_None). heat is |delta| normalized to [0,1] by abs_pct."""
    heat = np.abs(delta_3d).astype(np.float32)
    hi = np.percentile(heat, abs_pct)
    hi = hi if hi > 1e-12 else 1e-6
    heat = np.clip(heat / hi, 0.0, 1.0)
    mask = None
    if bin_pct is not None:
        thr = np.percentile(np.abs(delta_3d), bin_pct)
        mask = (np.abs(delta_3d) >= thr).astype(np.uint8)
    logger.info(f"Built heat (pct={abs_pct}) and mask (pct={bin_pct})")
    return heat, mask


def coerce_same_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop both to the minimum (D,H,W)."""
    if a.shape == b.shape:
        return a, b
    D = min(a.shape[0], b.shape[0])
    H = min(a.shape[1], b.shape[1])
    W = min(a.shape[2], b.shape[2])
    if (D, H, W) != a.shape or (D, H, W) != b.shape:
        logger.warning(
            f"Shape mismatch {a.shape} vs {b.shape}; cropping both to ({D},{H},{W})"
        )
    return a[:D, :H, :W], b[:D, :H, :W]


def reorient_like(
    data_3d: np.ndarray,
    affine: np.ndarray,
    ref_img: Optional[nib.spatialimages.SpatialImage] = None,
    target_orient: Optional[str] = None,  # e.g., "RAS" or "LPS"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorient (D,H,W) and affine to match either a reference NIfTI's orientation
    or an explicit target orientation string.

    Returns: (data_reoriented, affine_reoriented)
    """
    logger.info(f"Reorienting {data_3d.shape} to match reference nifti {ref_img.shape}")
    # Current orientation from our affine
    cur_ornt = io_orientation(affine)

    if ref_img is not None:
        tgt_ornt = io_orientation(ref_img.affine)
    elif target_orient:
        if len(target_orient) != 3 or any(
            c not in "RLPAIS" for c in target_orient.upper()
        ):
            raise ValueError(
                f"target_orient must be a 3-letter code like 'RAS' or 'LPS', got {target_orient}"
            )
        tgt_ornt = axcodes2ornt(tuple(target_orient.upper()))
    else:
        # No target → keep as-is
        return data_3d, affine

    # If already matching, skip
    if np.allclose(cur_ornt, tgt_ornt):
        logger.info(f"Already matching {cur_ornt} == {tgt_ornt}")
        return data_3d, affine

    # Build orientation transform and apply to data
    xform = ornt_transform(cur_ornt, tgt_ornt)
    data_re = apply_orientation(data_3d, xform)
    # Update affine so that new voxel indices map correctly to the same world space
    aff_re = affine @ inv_ornt_aff(xform, data_3d.shape)
    return data_re, aff_re


# Normalize intensities for better visualization in Slicer
def normalize_for_slicer(nii_img: nib.Nifti1Image, name="image") -> nib.Nifti1Image:
    data = nii_img.get_fdata()
    logger.info(
        f"{name} before normalization: min={data.min():.3f}, max={data.max():.3f}"
    )

    # Clip extreme outliers (optional)
    p1, p99 = np.percentile(data[data != 0], [1, 99])  # Ignore zeros
    data_clipped = np.clip(data, p1, p99)

    # Normalize to 0-255 range for better Slicer display
    if data_clipped.max() > data_clipped.min():
        data_normalized = (
            (data_clipped - data_clipped.min())
            / (data_clipped.max() - data_clipped.min())
            * 255
        )
    else:
        data_normalized = data_clipped

    logger.info(
        f"{name} after normalization: min={data_normalized.min():.3f}, max={data_normalized.max():.3f}"
    )

    return nib.Nifti1Image(
        data_normalized.astype(np.float32), nii_img.affine, nii_img.header
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


def safe_torch_load(path: str, map_location: torch.device | str = "cpu"):
    """
    Robust checkpoint loader across PyTorch>=2.6 (weights_only=True by default)
    and older checkpoints that pickle numpy scalar types.
    """
    # 1) Try weights_only=True with allowlisted numpy scalar
    try:
        add_safe_globals([np._core.multiarray.scalar])  # allow old numpy scalar pickles
    except Exception:
        # older torch versions may not have add_safe_globals; that's fine
        pass

    # Try modern safe path first
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch<2.6: weights_only kw not supported -> fall back to classic load
        return torch.load(path, map_location=map_location)
    except Exception as e_safe:
        # 2) Fallback: explicitly allow full pickle if you trust the source
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
        except Exception as e_full:
            raise RuntimeError(
                f"Failed to load checkpoint '{path}'. "
                f"weights_only=True error: {e_safe!r} | weights_only=False error: {e_full!r}"
            )


def load_model_from_results(
    model_dir: Path,
    fold: int,
    trainer: str | None,  # not strictly required; taken from checkpoint if None
    checkpoint_name: str = "checkpoint_final.pth",  # or 'checkpoint_best.pth'
    strict: bool = True,
    compile_network: bool = True,  # True/False to force; None -> respect env nnUNet_compile
    device_str: str | None = None,  # e.g. "cuda:0" / "cpu"; None -> auto
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a trained nnU-Net v2 network from nnUNet_results in a way consistent with the Predictor.
    Supports single or multi-fold. For multi-fold, returns a single network instance with the first
    fold’s weights loaded AND a 'list_of_parameters' you can use for ensemble inference.

    Returns:
        network (torch.nn.Module): model on device in eval mode (first fold weights)
        meta (dict): {
            'plans_manager', 'configuration_manager', 'label_manager', 'dataset_json',
            'trainer_name', 'configuration_name', 'allowed_mirroring_axes',
            'list_of_parameters' (for ensemble), 'model_dir', 'folds'
        }
    """
    model_dir = Path(model_dir)
    plans_path = model_dir / "plans.json"
    dataset_json_path = model_dir / "dataset.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"Missing plans.json: {plans_path}")
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"Missing dataset.json: {dataset_json_path}")

    plans = json.loads(plans_path.read_text())
    dataset_json = json.loads(dataset_json_path.read_text())
    plans_manager = PlansManager(plans)
    ckpt_file = model_dir / f"fold_{fold}" / checkpoint_name

    # ----- Load checkpoints (and sniff trainer/config on first fold) -----
    list_of_parameters = []
    checkpoint = safe_torch_load(str(ckpt_file), map_location=torch.device("cpu"))
    trainer_name = checkpoint.get("trainer_name", trainer or None)
    logger.info(f"Trainer name: {trainer_name}")
    configuration_name = checkpoint.get("init_args", {}).get("configuration")
    logger.info(f"Configuration name: {configuration_name}")
    allowed_mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes", None)
    logger.info(f"Allowed mirroring axes: {allowed_mirroring_axes}")

    weights = checkpoint.get(
        "network_weights",
        checkpoint.get("network_state_dict", checkpoint.get("state_dict", None)),
    )
    if weights is None:
        raise KeyError(f"Could not find weights in checkpoint {ckpt_file}")
    list_of_parameters.append(weights)

    if configuration_name is None:
        raise RuntimeError(
            "Could not determine configuration name (cfg). Provide 'cfg' or use a proper v2 checkpoint."
        )

    configuration_manager = plans_manager.get_configuration(configuration_name)
    label_manager: LabelManager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )
    num_output_channels = (
        label_manager.num_segmentation_heads
    )  # equals number of region/label heads

    logger.info(f"num_input_channels: {num_input_channels}")
    logger.info(f"num_output_channels: {num_output_channels}")

    # ----- Rebuild exact network architecture via trainer class -----
    if trainer_name is None:
        # Try to parse from directory name if missing
        # dir format: <trainer>__<plans>__<cfg>
        try:
            trainer_name = model_dir.name.split("__", 1)[0]
        except Exception:
            raise RuntimeError(
                "Unable to determine trainer_name from checkpoint or directory."
            )
    logger.info(f"Trainer name: {trainer_name}")
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer_class is None:
        raise RuntimeError(
            f"Unable to locate trainer class '{trainer_name}' in nnunetv2.training.nnUNetTrainer. "
            f"Make sure your custom trainer is placed there."
        )

    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=False,
    )

    # ----- Load first fold weights into the network -----
    network.load_state_dict(list_of_parameters[0], strict=strict)

    # ----- Device, eval, optional compile -----
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Device: {device}")
    network = network.to(device).eval()

    if compile_network:
        try:
            logger.info("Compiling network with torch.compile()")
            network = torch.compile(network)  # type: ignore[attr-defined]
        except Exception:
            # don’t hard fail on compile issues
            logger.warning("Failed to torch.compile() the network. Ignoring.")
            pass

    meta = {
        "plans_manager": plans_manager,
        "configuration_manager": configuration_manager,
        "label_manager": label_manager,
        "dataset_json": dataset_json,
        "trainer_name": trainer_name,
        "configuration_name": configuration_name,
        "allowed_mirroring_axes": allowed_mirroring_axes,
        "list_of_parameters": list_of_parameters,  # for ensemble inference across folds
        "model_dir": str(model_dir),
        "fold": fold,
        "checkpoint_name": checkpoint_name,
    }
    logger.info(
        f"[info] Loaded {model_dir.name} | cfg={configuration_name} | fold={fold} "
        f"| in={num_input_channels} out={num_output_channels} on {device}"
    )
    return network, meta


def load_volume(data_dir: Path, case_stem: str):
    """
    Load nnU-Net v2 preprocessed case (.b2nd + .pkl) via Blosc2.
    Returns (torch tensor (1, C, D, H, W), props)
    """
    b2nd = data_dir / f"{case_stem}.b2nd"
    pkl = data_dir / f"{case_stem}.pkl"
    if not b2nd.exists() or not pkl.exists():
        raise FileNotFoundError(f"Missing .b2nd or .pkl for {case_stem} in {data_dir}")

    # props (spacing, crop bbox, etc.), useful for later but not needed to decode b2nd
    with open(pkl, "rb") as f:
        props = pickle.load(f)

    # --- load compressed array (shape + dtype are stored inside) ---
    nd = blosc2.open(str(b2nd))  # Blosc2 NDArray handle
    arr = np.asarray(nd)  # materialize to NumPy

    # Normalize to (C, D, H, W)
    if arr.ndim == 3:
        # single-channel volume (D, H, W)
        arr = arr[None, ...]  # -> (1, D, H, W)
    elif arr.ndim == 4:
        # either (C, D, H, W) or (D, H, W, C)
        if arr.shape[0] in (1, 2, 3, 4, 5):
            pass  # already (C, D, H, W)
        elif arr.shape[-1] in (1, 2, 3, 4, 5):
            arr = np.moveaxis(arr, -1, 0)  # (D,H,W,C) -> (C,D,H,W)
        else:
            raise ValueError(f"Ambiguous channel axis for shape {arr.shape}")
    else:
        raise ValueError(f"Unexpected ndim {arr.ndim} for {b2nd.name}")

    arr = arr.astype(np.float32, copy=False)
    vol_t = torch.from_numpy(arr)[None, ...]  # (1, C, D, H, W)
    logger.info(f"Loaded {b2nd.name} with shape {vol_t.shape}")
    return vol_t, props


# ---------------------------
# Layer discovery utilities
# ---------------------------
def list_conv_layers(
    model: nn.Module, name_filter: Optional[str] = None
) -> List[Tuple[str, nn.Module]]:
    layers = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if name_filter is None or re.search(name_filter, n):
                layers.append((n, m))
    return layers


def pick_target_layer(
    model: nn.Module,
    layer_regex: str,
    *,
    target_idx: int = 0,
) -> nn.Module:
    candidates = list_conv_layers(model, name_filter=layer_regex)
    if not candidates:
        all_convs = list_conv_layers(model)
        raise RuntimeError(
            f"No conv layer matched regex '{layer_regex}'. "
            f"{len(all_convs)} convs exist; try a looser regex or print them."
        )

    # Python-style negative indexing
    if target_idx < 0:
        target_idx += len(candidates)

    if not (0 <= target_idx < len(candidates)):
        raise IndexError(
            f"target_idx={target_idx} out of bounds (have {len(candidates)} matches)"
        )

    name, layer = candidates[target_idx]
    logger.info(f"[info] Using target layer: {name}")
    return layer


def downsample_multiples(cfg: Any) -> tuple[int, int, int]:
    # cfg.pool_op_kernel_sizes is a list like [(2,2,2), (2,2,2), (2,2,2), (2,2,2), (2,2,2)]
    sizes = cfg.pool_op_kernel_sizes
    mult_d = math.prod(k[0] for k in sizes)
    mult_h = math.prod(k[1] for k in sizes)
    mult_w = math.prod(k[2] for k in sizes)
    logger.info(f"Downsample multiples: {mult_d}x{mult_h}x{mult_w}")
    return mult_d, mult_h, mult_w


def pad_to_multiples(
    x: torch.Tensor, mult: tuple[int, int, int]
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    _, _, D, H, W = x.shape
    md, mh, mw = mult
    logger.info(f"Padding {x.shape} to multiples of {mult}")
    padD = (md - D % md) % md
    padH = (mh - H % mh) % mh
    padW = (mw - W % mw) % mw
    # F.pad expects (W_right, W_left, H_right, H_left, D_right, D_left)
    x_pad = torch.nn.functional.pad(x, (0, padW, 0, padH, 0, padD))
    return x_pad, (padD, padH, padW)


def pad_to_multiples_dynamic(
    x: torch.Tensor, cfg: Optional[Any] = None
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    mult = downsample_multiples(cfg) if cfg is not None else (32, 32, 32)
    x_pad, pads = pad_to_multiples(x, mult)
    return x_pad, pads


def unpad_3d(arr: np.ndarray, pads: tuple[int, int, int]) -> np.ndarray:
    """
    Remove trailing padding (pad-at-the-end) from the last three dims.
    Works for shapes (..., D, H, W) including (D,H,W), (C,D,H,W), (N,C,D,H,W).
    """
    logger.info(f"Unpadding {arr.shape} by {pads}")
    padD, padH, padW = pads
    *lead, D, H, W = arr.shape
    slicer = [slice(None)] * len(lead) + [
        slice(0, D - padD if padD else D),
        slice(0, H - padH if padH else H),
        slice(0, W - padW if padW else W),
    ]
    return arr[tuple(slicer)]


def jitter3d(x: torch.Tensor, vox: int = 1) -> torch.Tensor:
    if vox <= 0:
        return x
    dz = random.randint(-vox, vox)
    dy = random.randint(-vox, vox)
    dx = random.randint(-vox, vox)
    return torch.roll(x, shifts=(dz, dy, dx), dims=(-3, -2, -1))


def largest_cc_bool(mask_3d: torch.Tensor) -> torch.Tensor:
    lab, n = ndi.label(mask_3d.cpu().numpy().astype(np.uint8))
    if n == 0:
        logger.info("No CCs found")
        return mask_3d
    sizes = ndi.sum(mask_3d.cpu().numpy(), lab, index=range(1, n + 1))
    keep = 1 + int(np.argmax(sizes))
    out = lab == keep
    logger.info(f"Kept CC {keep} out of {n} (size={sizes[keep - 1]})")
    return torch.from_numpy(out).to(mask_3d.device)


def resolve_class_idx(meta, class_name, fallback_idx):
    lm = meta["label_manager"]
    dsj = meta["dataset_json"]
    if hasattr(lm, "foreground_labels") and lm.foreground_labels:  # label-based
        if class_name:
            # dataset_json['labels'] maps name->id, you may also invert
            inv = {v: k for k, v in dsj["labels"].items()}
            key = class_name.lower()
            # simple aliases
            alias = {
                "necrotic": "necrotic/non-enhancing",
                "enhancing": "enhancing",
                "edema": "edema",
            }
            key = alias.get(key, key)
            if key in inv:
                return int(inv[key])
        return int(fallback_idx)
    else:  # region-based
        regions = [r.name.lower() for r in lm.foreground_regions]
        if class_name:
            key = class_name.lower()
            alias = {"wt": "whole tumor", "tc": "tumor core", "et": "enhancing tumor"}
            key = alias.get(key, key)
            for i, n in enumerate(regions):
                if key in n:
                    return i
        return min(fallback_idx, len(regions) - 1)
