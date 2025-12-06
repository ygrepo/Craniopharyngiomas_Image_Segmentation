#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple, Dict, Optional, Any, Union, List

import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import json
from tqdm import tqdm
import math
import pandas as pd  # NEW: to read feature importance CSV

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
        preprocessed_dir: Path to nnUNet_preprocessed/DatasetXXX/...
        case_id: Case identifier (e.g., 'Brats17_TCIA_001_1' or CP case)

    Returns:
        Tensor of shape [1, C, D, H, W] and metadata
    """
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


# Utility function to generate slice lists
def generate_slice_list(
    volume_shape: tuple, method: str = "evenly_spaced", num_slices: int = 5, **kwargs
) -> List[int]:
    """
    Generate a list of slice indices for visualization.

    Args:
        volume_shape: Shape of the 3D volume (D, H, W)
        method: Method to select slices
            - "evenly_spaced": Evenly distributed slices
            - "middle_range": Slices around the middle
            - "custom": Custom list provided in kwargs['slices']
        num_slices: Number of slices to select
        **kwargs: Additional parameters for specific methods

    Returns:
        List of slice indices
    """

    depth = volume_shape[0]

    if method == "evenly_spaced":
        return list(np.linspace(0, depth - 1, num_slices, dtype=int))

    elif method == "middle_range":
        middle = depth // 2
        half_range = num_slices // 2
        start = max(0, middle - half_range)
        end = min(depth, middle + half_range + 1)
        return list(range(start, end))

    elif method == "custom":
        return kwargs.get("slices", [depth // 2])

    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_results(
    original: torch.Tensor,
    dreamed: torch.Tensor,
    slice_idx: Optional[Union[int, List[int]]] = None,
    modality_idx: int = 0,
    save_path: Optional[Path] = None,
    max_cols: int = 4,
):
    """
    Visualize deep dream results for single slice or multiple slices.

    Args:
        original: Original tensor [B, C, D, H, W]
        dreamed: Deep dream result tensor [B, C, D, H, W]
        slice_idx: Single slice index, list of indices, or None for middle slice
        modality_idx: Which modality/channel to visualize
        save_path: Path to save the visualization
        max_cols: Maximum number of columns in multi-slice view
    """

    # Convert to numpy and remove batch dimension
    orig_np = original[0, modality_idx].cpu().numpy()
    dream_np = dreamed[0, modality_idx].cpu().numpy()

    if orig_np.ndim == 2:  # 2D image
        _visualize_single_slice(orig_np, dream_np, None, save_path)
        return

    # 3D volume
    if slice_idx is None:
        slice_idx = orig_np.shape[0] // 2

    if isinstance(slice_idx, int):
        _visualize_single_slice(orig_np, dream_np, slice_idx, save_path)
    else:
        _visualize_multiple_slices(orig_np, dream_np, slice_idx, save_path, max_cols)


def _visualize_single_slice(
    orig_np: np.ndarray,
    dream_np: np.ndarray,
    slice_idx: Optional[int],
    save_path: Optional[Path],
):
    """Visualize a single slice (original vs deep dream vs difference)."""

    if orig_np.ndim == 3 and slice_idx is not None:
        orig_slice = orig_np[slice_idx]
        dream_slice = dream_np[slice_idx]
        title_suffix = f" (Slice {slice_idx})"
    else:
        orig_slice = orig_np
        dream_slice = dream_np
        title_suffix = ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_slice, cmap="gray")
    axes[0].set_title(f"Original{title_suffix}")
    axes[0].axis("off")

    axes[1].imshow(dream_slice, cmap="gray")
    axes[1].set_title(f"Deep Dream{title_suffix}")
    axes[1].axis("off")

    diff = dream_slice - orig_slice
    im = axes[2].imshow(diff, cmap="RdBu_r")
    axes[2].set_title(f"Difference{title_suffix}")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def _visualize_multiple_slices(
    orig_np: np.ndarray,
    dream_np: np.ndarray,
    slice_indices: list[int],
    save_path: Optional[Path],
    max_cols: int = 4,
    diff_clip: float = 2.5,
):
    """
    Grid per slice: Original (top) | Edges (middle) | Difference (bottom).
    """

    def edge_mag(img2d: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import gaussian_filter, sobel

            x = gaussian_filter(img2d.astype(float), sigma=1.0)
            gx = sobel(x, axis=1)
            gy = sobel(x, axis=0)
            e = np.hypot(gx, gy)
        except Exception:
            gy, gx = np.gradient(img2d.astype(float))
            e = np.hypot(gx, gy)
        return e

    max_slice = orig_np.shape[0] - 1
    valid_slices = [s for s in slice_indices if 0 <= s <= max_slice]
    if not valid_slices:
        print(
            f"No valid slice indices. Volume has {orig_np.shape[0]} slices (0-{max_slice})"
        )
        return
    if len(valid_slices) != len(slice_indices):
        invalid = [s for s in slice_indices if s not in valid_slices]
        print(f"Warning: Invalid slice indices {invalid} ignored. Using {valid_slices}")
    num_slices = len(valid_slices)

    cols = min(max_cols, num_slices)
    rows = (num_slices + cols - 1) // cols

    fig, axes = plt.subplots(
        rows * 3,
        cols,
        figsize=(cols * 5.2, rows * 11.5),
        sharex="col",
        sharey="col",
        constrained_layout=True,
    )

    if rows * 3 == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows * 3 == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    edges_all = []
    for sidx in valid_slices:
        edges_all.append(edge_mag(orig_np[sidx]))
    p995 = np.percentile(np.concatenate([e.ravel() for e in edges_all]), 99.5)
    edge_vmin, edge_vmax = 0.0, float(p995 if p995 > 0 else np.max(edges_all))

    for i, sidx in enumerate(valid_slices):
        r = i // cols
        c = i % cols
        r0, r1, r2 = r * 3, r * 3 + 1, r * 3 + 2

        orig = orig_np[sidx]
        dream = dream_np[sidx]
        diff = dream - orig

        dstd = float(np.std(diff))
        lim = diff_clip * dstd if dstd > 0 else np.max(np.abs(diff)) + 1e-8
        vmin, vmax = -lim, +lim

        ax = axes[r0, c]
        ax.imshow(orig, cmap="gray", aspect="equal")
        ax.set_title(f"Original (Slice {sidx})")
        ax.axis("off")

        ax = axes[r1, c]
        e = edges_all[i]
        ax.imshow(e, cmap="gist_gray", vmin=edge_vmin, vmax=edge_vmax, aspect="equal")
        ax.set_title(f"Edges (Slice {sidx})")
        ax.axis("off")

        ax = axes[r2, c]
        im = ax.imshow(diff, cmap="jet", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"Difference (Slice {sidx})")
        ax.axis("off")

    total_used = num_slices * 3
    total_slots = rows * cols * 3
    for flat_idx in range(total_used, total_slots):
        rr = flat_idx // cols
        cc = flat_idx % cols
        axes[rr, cc].axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def _next_multiple(n: int, m: int) -> int:
    return int(math.ceil(n / m) * m)


def _pad_to_multiples(x: torch.Tensor, factors: tuple[int, ...], nd: int):
    spatial = list(x.shape[-nd:])
    target = [_next_multiple(s, f) for s, f in zip(spatial, factors)]
    pads = []
    for s, t in zip(spatial[::-1], target[::-1]):
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
    """Deep Dream implementation for nnU-Net models."""

    def __init__(self, model: nn.Module, meta: Dict[str, Any], use_bottleneck=False):
        self.model = model
        self.meta = meta
        self.device = next(model.parameters()).device
        self.use_bottleneck = use_bottleneck

    def compute_score(
        self,
        input_tensor: torch.Tensor,
        layer: nn.Module,
        filter_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute score for deep dream based on layer activations."""
        activations = {}

        def hook_fn(module, input, output):
            activations["target"] = output

        handle = layer.register_forward_hook(hook_fn)

        try:
            _ = self.model(input_tensor)
            if "target" not in activations:
                raise RuntimeError("Target layer activation not captured")

            activation = activations["target"]
            if filter_index is not None:
                if filter_index >= activation.shape[1]:
                    logger.warning(
                        f"filter_index {filter_index} >= number of channels "
                        f"{activation.shape[1]}; clipping to last channel."
                    )
                    filter_index = activation.shape[1] - 1
                score = activation[:, filter_index].mean()
            else:
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
        """Perform one gradient ascent step on the chosen feature channel."""
        x = input_tensor.detach().requires_grad_(True)
        score = self.compute_score(x, layer, filter_index)
        (-score).backward()  # gradient ascent on score

        with torch.no_grad():
            g = x.grad
            g = g / (torch.sqrt(torch.mean(g**2)) + 1e-5)
            x += step_size * g
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
                score = float(score_t.detach().item())
                if show_progress:
                    pbar.set_postfix(score=f"{score:.4f}")
                logger.info(f"Iter {i}, Score={score:.4f}")

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

        # Decide whether to hook bottleneck or use regex
        use_bottleneck = getattr(self, "use_bottleneck", False)
        if use_bottleneck:
            # This matches how you extracted latents: bottleneck_module = network.encoder.stages[-1]
            try:
                target_layer = self.model.encoder.stages[-1]
                logger.info(
                    "Using encoder bottleneck layer: model.encoder.stages[-1] "
                    f"(intended for latent_channel_idx={filter_index})"
                )
            except AttributeError as e:
                raise RuntimeError(
                    "Model does not have encoder.stages[-1]. Check architecture and "
                    "disable --use_bottleneck if incompatible."
                ) from e
        else:
            target_layer = pick_target_layer(
                self.model, layer_regex, target_idx=target_idx
            )
            logger.info(
                f"Using layer from regex '{layer_regex}' with target_idx={target_idx}"
            )

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

        pool_ks = getattr(cfg, "pool_op_kernel_sizes", None)
        logger.info(f"pool_ks: {pool_ks}")
        if pool_ks is None:
            pool_ks = pm.plans["configurations"][cfg_name]["pool_op_kernel_sizes"]

        nd = len(pool_ks[0])
        logger.info(f"nd: {nd}")
        interp_mode = "trilinear" if nd == 3 else "bilinear"
        logger.info(f"interp_mode: {interp_mode}")

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

        detail = torch.zeros_like(octaves[-1])
        pool_ks = cfg.pool_op_kernel_sizes
        factors = tuple(
            int(np.prod([stage[d] for stage in pool_ks])) for d in range(nd)
        )

        for i, octave_base in enumerate(
            tqdm(reversed(octaves), desc="Processing octaves")
        ):
            logger.info(f"Upsampling and dreaming octave {i}")

            cur_spatial = octave_base.shape[-nd:]
            if detail.shape[-nd:] != cur_spatial:
                detail = F.interpolate(
                    detail, size=cur_spatial, mode=interp_mode, align_corners=False
                )

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

            detail = dreamed - octave_base

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
        help="Path to nnU-Net preprocessed data (e.g., nnUNet_preprocessed/Dataset503_CP/...)",
    )
    ap.add_argument(
        "--case_id",
        type=str,
        required=True,
        help="Case ID (e.g., Brats17_TCIA_001_1 or CP case_id)",
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
        help="Specific feature channel to enhance (if not using latent_rank).",
    )
    ap.add_argument(
        "--filter_type",
        type=str,
        default="max",
        help="Filter type (placeholder; currently unused).",
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
        "--slice_idx",
        type=str,
        default="40,60,80",
        help="Comma-separated axial slice indices.",
    )
    ap.add_argument(
        "--modality_idx",
        type=int,
        default=0,
        help="Modality index for visualization (0=first channel, e.g., FLAIR)",
    )
    ap.add_argument(
        "--only_visualize",
        type=bool,
        default=False,
        help="Only visualize existing deep-dream result (no optimization).",
    )

    # NEW: latent feature integration
    ap.add_argument(
        "--feature_importance_csv",
        type=Path,
        default=None,
        help=(
            "CSV from compute_feature_importance.py. Must contain columns "
            "'group', 'latent_channel_idx', and 'rank_global'. "
            "If provided with --latent_rank, overrides --filter_index."
        ),
    )
    ap.add_argument(
        "--latent_rank",
        type=int,
        default=None,
        help=(
            "Global rank of latent feature to visualize (rank_global among latent rows). "
            "Used together with --feature_importance_csv to determine filter_index."
        ),
    )
    ap.add_argument(
        "--use_bottleneck",
        type=int,
        default=0,
        help="If 1, ignore layer_regex/target_idx and hook encoder.stages[-1] (the bottleneck).",
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

    preprocessed_data_dir = Path(args.preprocessed_dir).resolve()
    logger.info(f"Preprocessed data dir: {preprocessed_data_dir}")

    # Load model
    logger.info("Loading nnU-Net model...")
    model, meta = load_model_from_results(
        args.model_dir, args.fold, None, args.checkpoint, compile_network=False
    )
    model.eval()
    device = next(model.parameters()).device
    logger.info(f"Model on device: {device}")

    # Load data
    input_tensor, data_meta = load_nnunet_preprocessed_case(
        preprocessed_data_dir, args.case_id, device
    )

    # Prepare output dir
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original (for Slicer)
    save_b2nd_to_nifti_for_slicer(
        input_tensor.detach().cpu(),
        data_meta["properties"],
        output_dir,
        args.case_id,
        save_4d=False,
    )

    # Parse slice indices once
    slice_idx = [int(z) for z in str(args.slice_idx).split(",") if z.strip().isdigit()]

    if args.only_visualize:
        logger.info("Only visualizing results. Exiting after visualization.")
        dream_path = output_dir / f"{args.case_id}_deep_dream.nii.gz"
        img = nib.load(str(dream_path))
        dd = np.asanyarray(img.dataobj)  # (Z, Y, X, C) or (Z, Y, X)
        if dd.ndim == 3:
            dd = dd[..., None]
        dreamed_tensor = torch.from_numpy(dd.transpose(3, 0, 1, 2)).unsqueeze(0).float()

        viz_path = output_dir / f"{args.case_id}_visualization_2.png"
        visualize_results(
            original=input_tensor.detach().cpu(),
            dreamed=dreamed_tensor,
            slice_idx=slice_idx,
            modality_idx=args.modality_idx,
            save_path=viz_path,
        )
        sys.exit(0)

    # Resolve filter_index from latent feature importance if requested
    resolved_filter_index = args.filter_index
    selected_latent_row: Optional[Dict[str, Any]] = None

    if args.feature_importance_csv is not None and args.latent_rank is not None:
        df = pd.read_csv(args.feature_importance_csv)

        required_cols = {"group", "latent_channel_idx", "rank_global"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"feature_importance_csv is missing required columns: {missing}"
            )

        df_latent = df[df["group"] == "latent"].copy()
        row = df_latent[df_latent["rank_global"] == args.latent_rank]
        if row.empty:
            raise ValueError(
                f"No latent feature with rank_global == {args.latent_rank} "
                f"in {args.feature_importance_csv}"
            )
        row = row.iloc[0]
        resolved_filter_index = int(row["latent_channel_idx"])
        selected_latent_row = {
            "feature_name": row.get("feature_name", None),
            "group": row.get("group", None),
            "importance": float(row.get("importance", np.nan)),
            "rank_global": int(row.get("rank_global", -1)),
            "latent_channel_idx": resolved_filter_index,
        }
        logger.info(
            "Using latent feature from CSV: "
            f"feature_name='{selected_latent_row['feature_name']}', "
            f"group='{selected_latent_row['group']}', "
            f"importance={selected_latent_row['importance']:.4f}, "
            f"rank_global={selected_latent_row['rank_global']}, "
            f"latent_channel_idx={resolved_filter_index}"
        )

    if resolved_filter_index is None:
        logger.warning(
            "No filter_index provided and no latent_rank/feature_importance_csv set. "
            "DeepDream will optimize all channels jointly (activation norm)."
        )

    # Initialize and run DeepDream
    deep_dream = DeepDreamBraTS(model, meta, use_bottleneck=args.use_bottleneck)
    dreamed_tensor = deep_dream.run_deep_dream(
        input_tensor,
        layer_regex=args.layer_regex,
        target_idx=args.target_idx,
        iterations=args.iterations,
        step_size=args.step_size,
        filter_index=resolved_filter_index,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves,
    )

    # Save DeepDream as NIfTI, preserving geometry
    dreamed_np = dreamed_tensor.detach().cpu().numpy()[0]  # (C, Z, Y, X)
    dd_zyxc = dreamed_np.transpose(1, 2, 3, 0)  # (Z, Y, X, C)

    props = data_meta.get("properties", {})
    affine = props.get("original_affine", np.eye(4))
    nii_img = nib.Nifti1Image(dd_zyxc, affine=affine)
    deep_dream_path = output_dir / f"{args.case_id}_deep_dream.nii.gz"
    nib.save(nii_img, str(deep_dream_path))
    logger.info(f"Saved deep dream result to: {deep_dream_path}")

    # Visualize
    viz_path = output_dir / f"{args.case_id}_visualization_2.png"
    visualize_results(
        original=input_tensor.detach().cpu(),
        dreamed=dreamed_tensor.detach().cpu(),
        slice_idx=slice_idx,
        modality_idx=args.modality_idx,
        save_path=viz_path,
    )

    # Save parameters and latent feature metadata
    params: Dict[str, Any] = {
        "case_id": args.case_id,
        "model_dir": str(args.model_dir),
        "fold": args.fold,
        "layer_regex": args.layer_regex,
        "target_idx": args.target_idx,
        "iterations": args.iterations,
        "step_size": args.step_size,
        "filter_index": resolved_filter_index,
        "num_octaves": args.num_octaves,
        "octave_scale": args.octave_scale,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(dreamed_tensor.shape),
        "feature_importance_csv": (
            str(args.feature_importance_csv) if args.feature_importance_csv else None
        ),
        "latent_rank": args.latent_rank,
        "selected_latent_feature": selected_latent_row,
    }
    params_path = output_dir / f"{args.case_id}_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    logger.info("Deep dream completed successfully!")
