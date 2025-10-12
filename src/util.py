from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import List, Optional
import SimpleITK as sitk
import numpy as np

from typing import Dict, Any
import torch
from torch.serialization import add_safe_globals
import nnunetv2
import json


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
