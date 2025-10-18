#!/usr/bin/env python3
"""
DeepDream-style activation maximization for nnU-Net v2 (BraTS2017).

Two objectives:
  (A) 'logit'   : maximize a segmentation class logit within a (predicted) mask
  (B) 'feature' : maximize a chosen encoder/decoder feature channel via forward hook

MRI-aware regularization:
  - 3D Total Variation (TV)
  - High-frequency energy penalty in k-space (limits edgy hallucinations)
  - Intensity anchoring toward the original scan (keeps histogram realistic)

Outputs:
  - dream.npy : dreamed volume tensor (1,C,D,H,W) after optimization
  - delta.npy : (dream - init) to visualize what changed
  - objective_trace.npy : per-iteration objective/regularizer values
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.util import (
    get_logger,
    setup_logging,
    load_model_from_results,
    pick_target_layer,
    load_volume,
    downsample_multiples,
    pad_to_multiples,
    unpad_3d,
    save_npy,
    largest_cc_bool,
)

logger = get_logger(__name__)


# ---------------------------
# Helpers
# ---------------------------
def tv3d(z: torch.Tensor) -> torch.Tensor:
    # z: (1, C, D, H, W)
    dz = (z[:, :, 1:, :, :] - z[:, :, :-1, :, :]).abs().mean()
    dy = (z[:, :, :, 1:, :] - z[:, :, :, :-1, :]).abs().mean()
    dx = (z[:, :, :, :, 1:] - z[:, :, :, :, :-1]).abs().mean()
    return dx + dy + dz


def radial_freq_energy(z: torch.Tensor, frac: float = 0.35) -> torch.Tensor:
    """
    Penalize energy beyond a radial cutoff in k-space to discourage unrealistic high-freq texture.
    z: (1, C, D, H, W) – assumes channels share same spatial dims.
    """
    # FFT over spatial dims only
    Z = torch.fft.fftn(z, dim=(-3, -2, -1))
    Zs = torch.fft.fftshift(Z, dim=(-3, -2, -1))
    D, H, W = z.shape[-3:]
    yy, xx, zz = torch.meshgrid(
        torch.linspace(-1, 1, D, device=z.device),
        torch.linspace(-1, 1, H, device=z.device),
        torch.linspace(-1, 1, W, device=z.device),
        indexing="ij",
    )
    r = (xx**2 + yy**2 + zz**2).sqrt()
    mask = (r > frac).float()  # high-frequency shell
    return (Zs * mask).abs().pow(2).mean()


def intensity_anchor(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    Pull current intensities toward the original (keeps histogram/contrast sane).
    """
    return (x - x0).pow(2).mean()


class FeatureHook:
    def __init__(self, layer: nn.Module):
        self.feat = None
        self.h = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.feat = out  # (B, K, D', H', W')

    def remove(self):
        self.h.remove()


# ---------------------------
# Dream objective
# ---------------------------
def seg_logit_objective(
    logits: torch.Tensor,
    class_idx: int,
    mask: Optional[torch.Tensor] = None,
    lam_bg: float = 0.25,
) -> torch.Tensor:
    """
    logits: (1, C, D, H, W)
    mask  : (D, H, W) or (1, D, H, W) boolean/float
    """
    s = logits[0, class_idx]  # (D,H,W)
    if mask is None:
        return s.mean()
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    mask = mask.to(s.device).float()
    inv = 1.0 - mask
    fg = (s * mask).sum() / mask.sum().clamp_min(1)
    bg = (s * inv).sum() / inv.sum().clamp_min(1)
    return fg - lam_bg * bg


def feature_channel_objective(feat: torch.Tensor, channel_idx: int) -> torch.Tensor:
    """
    feat: (1, K, D', H', W'), maximize mean of one channel
    """
    return feat[0, channel_idx].mean()


# ---------------------------
# Core runner
# ---------------------------
def run_deepdream(
    model: nn.Module,
    vol_t: torch.Tensor,  # (1, C, D, H, W), already normalized like training
    objective: str,  # "logit" or "feature"
    class_idx: int,
    use_pred_mask: bool,
    layer_regex: Optional[str],
    channel_idx: Optional[int],
    steps: int,
    lr: float,
    w_tv: float,
    w_hf: float,
    w_anchor: float,
    clamp_to_init: bool,
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device

    # Pad to multiples expected by nnU-Net
    cfg = None
    if hasattr(model, "configuration_manager"):
        cfg = model.configuration_manager
    # if meta packed model differently, the caller sends cfg; but we try helper:
    try:
        # Try to fetch cfg from model attr (nnU-Net v2 stores it in trainer, but load util returns meta)
        from types import SimpleNamespace  # noqa: F401
    except Exception:
        pass

    # If caller provided downsample_multiples via meta, they'll precompute and pass pads.
    # Here, we recompute to be safe using util.
    mult = downsample_multiples(cfg) if cfg is not None else (32, 32, 32)
    vol_pad, pads = pad_to_multiples(vol_t, mult)  # keep torch

    x0 = vol_pad.to(device).contiguous()  # reference image
    x = x0.clone().detach().requires_grad_(True)

    mask = None
    with torch.no_grad():
        logits0 = model(x0)  # (1,C,D,H,W)
        if use_pred_mask and objective == "logit":
            pred = logits0.argmax(dim=1)  # (1,D,H,W)
            mask = (pred == class_idx)[0]
            mask = largest_cc_bool(mask)  # prune flyaways

    # Set up feature hook if needed
    f_hook = None
    if objective == "feature":
        if layer_regex is None:
            raise ValueError("For 'feature' objective, --layer_regex must be provided.")
        layer = pick_target_layer(model, layer_regex, target_idx=-1)
        f_hook = FeatureHook(layer)

    opt = torch.optim.Adam([x], lr=lr)

    # For logging
    hist = {"obj": [], "tv": [], "hf": [], "anchor": []}

    for t in range(1, steps + 1):
        opt.zero_grad()

        # Forward
        logits = model(x)  # (1,C,D,H,W)

        if objective == "logit":
            obj = seg_logit_objective(logits, class_idx, mask=mask, lam_bg=0.25)
        else:
            # Ensure hook fired
            _ = logits  # no-op; forward already done
            if f_hook is None or f_hook.feat is None:
                # Some nnU-Net variants compute encoder before decoder; a second call is harmless if needed.
                logits = model(x)
            feat = f_hook.feat
            if feat is None:
                raise RuntimeError(
                    "Feature hook did not capture activations. Check --layer_regex."
                )
            if channel_idx is None or channel_idx >= feat.shape[1]:
                raise ValueError(
                    f"--channel_idx out of range (got {channel_idx}, feature has {feat.shape[1]} channels)."
                )
            obj = feature_channel_objective(feat, channel_idx)

        # Regularizers
        r_tv = tv3d(x)
        r_hf = radial_freq_energy(x, frac=0.35)
        r_anchor = intensity_anchor(x, x0)

        loss = -(obj) + w_tv * r_tv + w_hf * r_hf + w_anchor * r_anchor
        loss.backward()
        opt.step()

        if clamp_to_init:
            # Keep intensities within [min, max] of the padded original
            x.data.clamp_(x0.min().item(), x0.max().item())

        # Log
        hist["obj"].append(float(obj.detach().cpu()))
        hist["tv"].append(float(r_tv.detach().cpu()))
        hist["hf"].append(float(r_hf.detach().cpu()))
        hist["anchor"].append(float(r_anchor.detach().cpu()))

        if t % max(1, steps // 10) == 0:
            logger.info(
                f"[{t:04d}/{steps}] obj={hist['obj'][-1]:.4f} | "
                f"tv={hist['tv'][-1]:.4e} hf={hist['hf'][-1]:.4e} anchor={hist['anchor'][-1]:.4e}"
            )

    # Unpad to original shape
    dream = unpad_3d(x.detach().cpu().numpy(), pads)  # (1,C,D,H,W)
    init = unpad_3d(x0.detach().cpu().numpy(), pads)
    delta = dream - init
    return {"dream": dream, "delta": delta, "trace": hist}


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        type=Path,
        default="nnUNet_results/Dataset501_BraTS2017_4ch/nnUNetTrainer__nnUNetPlans__3d_fullres/",
        help="Path to model dir (trainer folder).",
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default="nnUNet_preprocessed/Dataset501_BraTS2017_4ch/nnUNetPlans_3d_fullres",
        help="Path to preprocessed data dir (b2nd+pkl).",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default="output/deepdream",
        help="Where to save dream, delta, and traces.",
    )
    ap.add_argument(
        "--case",
        type=str,
        default="Brats17_CBICA_AAG_1",
        help="Case stem, e.g., 'Brats17_CBICA_AAG_1'.",
    )
    ap.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold (0..4) for 5-fold; -1 for ensemble (if supported by your loader).",
    )
    ap.add_argument(
        "--objective",
        type=str,
        default="logit",
        choices=["logit", "feature"],
        help="Maximize a class logit or an internal feature channel.",
    )
    ap.add_argument(
        "--class_idx",
        type=int,
        default=3,
        help="Segmentation class index for 'logit' objective (verify mapping!).",
    )
    ap.add_argument(
        "--layer_regex",
        type=str,
        default=r"encoder|down|context|stem",
        help="Regex to pick a conv layer (used for 'feature' objective).",
    )
    ap.add_argument(
        "--channel_idx",
        type=int,
        default=16,
        help="Feature channel index to maximize (used for 'feature' objective).",
    )
    ap.add_argument(
        "--use_pred_mask",
        type=int,
        default=1,
        help="Restrict logit objective to predicted mask (1/0).",
    )
    ap.add_argument("--steps", type=int, default=250, help="Optimization steps.")
    ap.add_argument("--lr", type=float, default=0.07, help="Adam learning rate.")
    ap.add_argument("--w_tv", type=float, default=1e-3, help="TV weight.")
    ap.add_argument(
        "--w_hf", type=float, default=1e-5, help="High-frequency penalty weight."
    )
    ap.add_argument(
        "--w_anchor",
        type=float,
        default=5e-4,
        help="Intensity anchor (to original) weight.",
    )
    ap.add_argument(
        "--clamp_to_init",
        type=int,
        default=1,
        help="Clamp intensities to the original min/max (1/0).",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default="logs/nnunet_deepdream.log",
        help="Log file path (also logs to console).",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")

    # 1) Load model
    model, meta = load_model_from_results(
        model_dir=args.model_dir.resolve(),
        fold=args.fold,
        checkpoint_name="checkpoint_best.pth",
        trainer=None,
        compile_network=False,
    )
    device = next(model.parameters()).device
    logger.info(f"Model on device: {device}")

    # 2) Load volume (preprocessed b2nd) → (1,C,D,H,W) torch
    vol_t, props = load_volume(
        args.data_dir.resolve(), args.case
    )  # returns torch tensor normalized like training
    vol_t = vol_t.to(device)

    # 3) Run DeepDream
    out = run_deepdream(
        model=model,
        vol_t=vol_t,
        objective=args.objective.lower(),
        class_idx=int(args.class_idx),
        use_pred_mask=bool(args.use_pred_mask),
        layer_regex=args.layer_regex if args.objective == "feature" else None,
        channel_idx=int(args.channel_idx) if args.objective == "feature" else None,
        steps=int(args.steps),
        lr=float(args.lr),
        w_tv=float(args.w_tv),
        w_hf=float(args.w_hf),
        w_anchor=float(args.w_anchor),
        clamp_to_init=bool(args.clamp_to_init),
    )

    # 4) Save outputs
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npy(out["dream"], out_dir / f"{args.case}_{args.objective}_dream.npy")
    save_npy(out["delta"], out_dir / f"{args.case}_{args.objective}_delta.npy")
    # also save trace
    trace = np.stack(
        [
            np.array(out["trace"]["obj"]),
            np.array(out["trace"]["tv"]),
            np.array(out["trace"]["hf"]),
            np.array(out["trace"]["anchor"]),
        ],
        axis=1,
    )
    save_npy(trace, out_dir / f"{args.case}_{args.objective}_trace.npy")
    logger.info(f"Saved dream/delta/trace to {out_dir}")


if __name__ == "__main__":
    main()
