#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple, Dict, Optional, Any

import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import json
from tqdm import tqdm
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    load_model_from_results,
    setup_logging,
    load_volume,
    pick_target_layer,
    save_b2nd_to_nifti_for_slicer,
)

logger = get_logger(__name__)


def load_nnunet_preprocessed_case(
    preprocessed_dir: Path, case_id: str, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, Dict]:
    """
    Load preprocessed nnU-Net data (.pkl files).

    Args:
        preprocessed_dir: Path to nnUNet_preprocessed/DatasetXXX_BraTS2017/nnUNetTrainer__nnUNetPlans__3d_fullres
        case_id: Case identifier (e.g., 'Brats17_TCIA_001_1')

    Returns:
        Tensor of shape [1, 4, D, H, W] and metadata
    """
    # 2) Load volume (preprocessed b2nd) → (1,C,D,H,W) torch
    vol_t, props = load_volume(
        preprocessed_dir, case_id
    )  # returns torch tensor normalized like training
    vol_t = vol_t.to(device)

    metadata = {
        "case_id": case_id,
        "shape": vol_t.shape,
        "properties": props,
        "preprocessed": True,
    }

    logger.info(f"Loaded preprocessed case {case_id}: shape {vol_t.shape}")

    return vol_t, metadata


def visualize_results(
    original: torch.Tensor,
    dreamed: torch.Tensor,
    slice_idx: Optional[int] = None,
    modality_idx: int = 0,
    save_path: Optional[Path] = None,
):
    """Visualize deep dream results."""

    # Convert to numpy and remove batch dimension
    orig_np = original[0, modality_idx].cpu().numpy()
    dream_np = dreamed[0, modality_idx].cpu().numpy()

    if orig_np.ndim == 3:  # 3D volume
        if slice_idx is None:
            slice_idx = orig_np.shape[0] // 2
        orig_slice = orig_np[slice_idx]
        dream_slice = dream_np[slice_idx]
    else:  # 2D image
        orig_slice = orig_np
        dream_slice = dream_np

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_slice, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(dream_slice, cmap="gray")
    axes[1].set_title("Deep Dream")
    axes[1].axis("off")

    # Difference
    diff = dream_slice - orig_slice
    im = axes[2].imshow(diff, cmap="RdBu_r")
    axes[2].set_title("Difference")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def _next_multiple(n: int, m: int) -> int:
    return int(math.ceil(n / m) * m)


def _pad_to_multiples(x: torch.Tensor, factors: tuple[int, ...], nd: int):
    spatial = list(x.shape[-nd:])
    target = [_next_multiple(s, f) for s, f in zip(spatial, factors)]
    pads = []
    for s, t in zip(spatial[::-1], target[::-1]):  # F.pad uses reverse spatial order
        diff = t - s
        left = diff // 2
        right = diff - left
        pads.extend([left, right])
    if any(pads):
        x = F.pad(x, pads, mode="reflect")
    return x, pads


def _crop_from_pads(x: torch.Tensor, pads: list[int], nd: int):
    if not any(pads):
        return x
    slc = [slice(None), slice(None)]
    if nd == 2:
        wL, wR, hL, hR = pads
        slc += [slice(hL, x.shape[-2] - hR), slice(wL, x.shape[-1] - wR)]
    else:
        wL, wR, hL, hR, dL, dR = pads
        slc += [
            slice(dL, x.shape[-3] - dR),
            slice(hL, x.shape[-2] - hR),
            slice(wL, x.shape[-1] - wR),
        ]
    return x[tuple(slc)]


class DeepDreamBraTS:
    """Deep Dream implementation for BraTS2017 using nnU-Net models."""

    def __init__(self, model: nn.Module, meta: Dict[str, Any]):
        self.model = model
        self.meta = meta
        self.device = next(model.parameters()).device

    def compute_score(
        self,
        input_tensor: torch.Tensor,
        layer: nn.Module,
        filter_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute score for deep dream based on layer activations."""

        # Hook to capture activations
        activations = {}

        def hook_fn(module, input, output):
            activations["target"] = output

        handle = layer.register_forward_hook(hook_fn)

        try:
            # Forward pass
            _ = self.model(input_tensor)

            if "target" not in activations:
                raise RuntimeError("Target layer activation not captured")

            activation = activations["target"]

            if filter_index is not None:
                # Focus on specific filter/channel
                if filter_index >= activation.shape[1]:
                    filter_index = activation.shape[1] - 1
                score = activation[:, filter_index].mean()
            else:
                # Use all activations
                score = activation.norm()

        finally:
            handle.remove()

        return score

    def gradient_ascent_step(
        self,
        input_tensor: torch.Tensor,
        layer: nn.Module,
        filter_index: Optional[int] = None,
        step_size: float = 0.01,
    ) -> torch.Tensor:
        """Perform one gradient ascent step."""
        x = input_tensor.detach().requires_grad_(True)
        # Compute score
        score = self.compute_score(x, layer, filter_index)
        (-score).backward()  # descent on (-score) == ascent on score

        # Gradient ascent step
        with torch.no_grad():
            g = x.grad
            g = g / (torch.sqrt(torch.mean(g**2)) + 1e-5)
            x += step_size * g  # ascent on score
            x.clamp_(-3, 3)
        return x.detach()

    def deep_dream_loop(
        self,
        input_tensor: torch.Tensor,
        layer: nn.Module,
        *,
        iterations: int = 20,
        step_size: float = 0.01,
        filter_index: Optional[int] = None,
        log_every: int = 5,
        show_progress: bool = True,
        leave_bar: bool = False,
    ) -> torch.Tensor:
        """Run the deep dream optimization loop."""

        input_tensor = input_tensor.clone().detach()
        pbar = tqdm(
            range(iterations),
            desc="DeepDream",
            disable=not show_progress,
            leave=leave_bar,
        )

        for i in pbar:
            input_tensor = self.gradient_ascent_step(
                input_tensor, layer, filter_index, step_size
            )

            if i % log_every == 0:
                score_t = self.compute_score(input_tensor, layer, filter_index)
                score = -float(score_t.detach().item())
                loss = -score
                if show_progress:
                    pbar.set_postfix(score=f"{score:.4f}")
                logger.info(f"Iter {i}, Score={score:.4f}, Loss={loss:.4f}")

        return input_tensor

    def run_deep_dream(
        self,
        input_data: torch.Tensor,
        layer_regex: str = r"encoder",
        target_idx: int = 0,
        iterations: int = 20,
        step_size: float = 0.01,
        filter_index: Optional[int] = None,
        octave_scale: float = 1.4,
        num_octaves: int = 3,
    ) -> torch.Tensor:

        logger.info("Running deep dream...")
        logger.info(
            f"target_idx: {target_idx}, filter_index: {filter_index}, num_octaves: {num_octaves}, octave_scale: {octave_scale}"
        )
        # 1) pick target layer
        target_layer = pick_target_layer(self.model, layer_regex, target_idx=target_idx)

        # 2) shape/device bookkeeping
        if input_data.dim() == 4:  # [B, C, H, W]
            original_shape = input_data.shape[-2:]
            is_3d = False
        elif input_data.dim() == 5:  # [B, C, D, H, W]
            original_shape = input_data.shape[-3:]
            is_3d = True
        else:
            raise ValueError(f"Unsupported input dimensions: {input_data.shape}")

        logger.info(f"Input shape: {input_data.shape}, is_3d={is_3d}")
        input_tensor = input_data.to(self.device)
        cfg = self.meta["configuration_manager"]
        pm = self.meta["plans_manager"]
        cfg_name = self.meta["configuration_name"]

        # 1) Get pool kernels (prefer cfg, else pull from plans)
        pool_ks = getattr(cfg, "pool_op_kernel_sizes", None)
        logger.info(f"pool_ks: {pool_ks}")
        if pool_ks is None:
            # nnU-Net v2 stores this under plans -> configurations -> <cfg>
            pool_ks = pm.plans["configurations"][cfg_name]["pool_op_kernel_sizes"]

        # 2) Number of spatial dims (2 for 2D, 3 for 3D)
        nd = len(pool_ks[0])  # or: nd = len(getattr(cfg, "patch_size", pool_ks[0]))
        logger.info(f"nd: {nd}")
        # 4) Interp mode from nd
        interp_mode = "trilinear" if nd == 3 else "bilinear"
        logger.info(f"interp_mode: {interp_mode}")

        # 3) build octaves: [largest (=original), smaller, smallest]
        octaves = []
        for i in range(num_octaves):
            if i == 0:
                octaves.append(input_tensor)
            else:
                logger.info(f"Downsampling octave {i}")
                scale = octave_scale ** (-i)
                new_spatial = [max(1, int(s * scale)) for s in original_shape]
                scaled = F.interpolate(
                    input_tensor,
                    size=new_spatial,
                    mode=interp_mode,
                    align_corners=False,
                )
                octaves.append(scaled)

        # 4) process smallest -> largest
        #    IMPORTANT: init detail to smallest shape to avoid first-iter mismatch
        detail = torch.zeros_like(octaves[-1])
        pool_ks = cfg.pool_op_kernel_sizes
        # 3) Total down/up factor per axis = product over stages
        factors = tuple(
            int(np.prod([stage[d] for stage in pool_ks])) for d in range(nd)
        )

        for i, octave_base in enumerate(
            tqdm(reversed(octaves), desc="Processing octaves")
        ):
            logger.info(f"Upsampling and dreaming octave {i}")

            # resize detail to current octave if needed (robust guard)
            cur_spatial = octave_base.shape[-nd:]
            if detail.shape[-nd:] != cur_spatial:
                detail = F.interpolate(
                    detail, size=cur_spatial, mode=interp_mode, align_corners=False
                )

            # add detail and dream at this scale
            input_octave = octave_base + detail
            input_padded, pads = _pad_to_multiples(input_octave, factors, nd)
            dreamed = self.deep_dream_loop(
                input_tensor=input_padded,
                layer=target_layer,
                iterations=iterations,
                step_size=step_size,
                filter_index=filter_index,
            )
            dreamed = _crop_from_pads(dreamed, pads, nd)

            # update detail contributed at this scale
            detail = dreamed - octave_base

        # 5) upsample accumulated detail back to original size and add
        if detail.shape[-nd:] != tuple(original_shape):
            detail = F.interpolate(
                detail, size=original_shape, mode=interp_mode, align_corners=False
            )

        result = input_tensor + detail
        return result


# -------------------------- main -------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()

    # Model arguments
    ap.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to nnU-Net results directory",
    )
    ap.add_argument("--fold", type=int, default=0, help="Model fold to use")
    ap.add_argument(
        "--checkpoint", type=str, default="checkpoint_final.pth", help="Checkpoint name"
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default="output/deepdream_keras",
        help="Where to save dream, delta, and traces.",
    )
    ap.add_argument(
        "--preprocessed_dir",
        type=Path,
        required=True,
        help="Path to nnU-Net preprocessed data (e.g., nnUNet_preprocessed/Dataset001_BraTS2017)",
    )
    ap.add_argument(
        "--case_id",
        type=str,
        required=True,
        help="BraTS case ID (e.g., Brats17_TCIA_001_1)",
    )
    # Deep dream arguments
    ap.add_argument(
        "--layer_regex",
        type=str,
        default=r"encoder|down|context",
        help="Regex to pick target layer",
    )
    ap.add_argument(
        "--target_idx", type=int, default=0, help="Layer index if multiple matches"
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of gradient ascent iterations",
    )
    ap.add_argument(
        "--step_size", type=float, default=0.01, help="Gradient ascent step size"
    )
    ap.add_argument(
        "--filter_index",
        type=int,
        default=None,
        help="Specific filter to enhance (None for all)",
    )
    ap.add_argument(
        "--num_octaves",
        type=int,
        default=3,
        help="Number of octaves for multi-scale processing",
    )
    ap.add_argument(
        "--octave_scale", type=float, default=1.4, help="Scale factor between octaves"
    )
    ap.add_argument(
        "--slice_idx", type=int, default=None, help="Slice index for visualization"
    )
    ap.add_argument(
        "--modality_idx",
        type=int,
        default=0,
        help="Modality index for visualization (0=FLAIR)",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="logs/nnunet_deepdream_keras.log",
        help="Log file path (in addition to console).",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")
    preprocessed_data_dir = args.preprocessed_dir.resolve()
    logger.info(f"Preprocessed data dir: {preprocessed_data_dir}")

    # Load model
    logger.info("Loading nnU-Net model...")
    model, meta = load_model_from_results(
        args.model_dir, args.fold, None, args.checkpoint, compile_network=False
    )
    device = next(model.parameters()).device
    logger.info(f"Model on device: {device}")

    # Load data
    input_tensor, data_meta = load_nnunet_preprocessed_case(
        preprocessed_data_dir, args.case_id, device
    )

    # Save results
    output_dir = args.output_dir.resolve()

    save_b2nd_to_nifti_for_slicer(
        input_tensor,
        data_meta["properties"],
        output_dir,
        args.case_id,
        save_4d=False,
    )

    # Initialize deep dream
    deep_dream = DeepDreamBraTS(model, meta)
    # Run deep dream
    dreamed_tensor = deep_dream.run_deep_dream(
        input_tensor,
        layer_regex=args.layer_regex,
        target_idx=args.target_idx,
        iterations=args.iterations,
        step_size=args.step_size,
        filter_index=args.filter_index,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves,
    )

    dreamed_np = dreamed_tensor[0].cpu().numpy()  # Remove batch dimension

    # Save as NIfTI
    nii_img = nib.Nifti1Image(dreamed_np.transpose(1, 2, 3, 0), affine=np.eye(4))
    nib.save(nii_img, str(output_dir / f"{args.case_id}_deep_dream.nii.gz"))
    logger.info(
        f"Saved deep dream result to: {output_dir / f'{args.case_id}_deep_dream.nii.gz'}"
    )

    # Visualize
    viz_path = args.output_dir / f"{args.case_id}_visualization.png"
    visualize_results(
        input_tensor,
        dreamed_tensor,
        slice_idx=args.slice_idx,
        modality_idx=args.modality_idx,
        save_path=viz_path,
    )

    # Save parameters
    params = {
        "case_id": args.case_id,
        "model_dir": str(args.model_dir),
        "fold": args.fold,
        "layer_regex": args.layer_regex,
        "target_idx": args.target_idx,
        "iterations": args.iterations,
        "step_size": args.step_size,
        "filter_index": args.filter_index,
        "num_octaves": args.num_octaves,
        "octave_scale": args.octave_scale,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(dreamed_tensor.shape),
    }

    params_path = args.output_dir / f"{args.case_id}_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    logger.info("Deep dream completed successfully!")
