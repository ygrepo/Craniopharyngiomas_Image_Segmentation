#!/usr/bin/env python3
"""
predict_cp_sam3d.py
Batch 3D auto-segmentation with a fine-tuned SAM checkpoint — no masks needed.
- Expects test_dir/CASE_ID/*.nii.gz (at least one image per case).
- Writes predict_dir/CASE_ID/pred_mask.nii.gz

Requirements (same as the notebook):
  pip install torch torchvision torchaudio
  pip install torchio simpleitk nibabel scikit-image opencv-python

This script assumes your training repo exposes:
  - sam_model_registry
  - a fine-tuned SAM loading path that accepts (args, checkpoint)

If your project uses different import paths, edit `build_sam_model()` accordingly.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
import nibabel as nib
import SimpleITK as sitk
import cv2


# -----------------------------
# Model loading (edit if needed)
# -----------------------------
def build_sam_model(args_json: Path, checkpoint_path: Path, device: torch.device):
    """
    Build the fine-tuned SAM model from args.json + checkpoint.

    If your repo exposes a different API, adjust this function.
    Common pattern in finetune-SAM:
        from segment_anything.modeling import sam_model_registry
        sam = sam_model_registry[args['arch']](args, checkpoint=checkpoint_path)
    """
    with open(args_json, "r") as f:
        args = json.load(f)

    arch = args.get("arch", "vit_b")
    normalize_type = args.get("normalize_type", "sam")  # "sam" or "medsam"
    finetune_type = args.get("finetune_type", "adapter")  # optional

    try:
        # Typical registry:
        from segment_anything.modeling import sam_model_registry

        sam = sam_model_registry[arch](args, checkpoint=str(checkpoint_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate SAM model with arch={arch}. "
            f"Edit build_sam_model() to match your repo API.\n{e}"
        )

    sam.to(device)
    sam.eval()
    return sam, normalize_type, arch, finetune_type


# -----------------------------
# I/O helpers
# -----------------------------
def find_image_path(
    case_dir: Path, prefer_patterns: Optional[List[str]] = None
) -> Optional[Path]:
    """
    Return an image path for a case. If multiple NIfTIs exist, optionally prefer patterns.
    """
    cands = sorted(case_dir.glob("*.nii")) + sorted(case_dir.glob("*.nii.gz"))
    if not cands:
        return None
    if prefer_patterns:
        low = [
            p
            for p in cands
            if any(p.name.lower().find(pat) >= 0 for pat in prefer_patterns)
        ]
        if low:
            return low[0]
    return cands[0]


def read_volume_tio(img_path: Path) -> tio.ScalarImage:
    return tio.ScalarImage(str(img_path))


def save_mask_like(ref_img: tio.ScalarImage, mask_zyx: np.ndarray, out_path: Path):
    """
    Save a binary mask (Z,Y,X) with same affine/header as ref image.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    affine = ref_img.affine
    # TorchIO uses channels-first (C,Z,Y,X) internally; here we already have Z,Y,X
    nii = nib.Nifti1Image(mask_zyx.astype(np.uint8), affine)
    nib.save(nii, str(out_path))


# -----------------------------
# Geometry / orientation utils
# -----------------------------
def maybe_flip_lr_for_R_orientation(
    vol: tio.ScalarImage, arr_zyx: np.ndarray
) -> np.ndarray:
    """
    Some pipelines flip if orientation starts with 'R' (RAS/LPS handling).
    If TorchIO captured orientation and its first label is 'R', we LR-flip the array.
    """
    try:
        ori = getattr(vol, "orientation", None)
        if ori is not None and len(ori) >= 1 and ori[0] == "R":
            return arr_zyx[:, :, ::-1]
    except Exception:
        pass
    return arr_zyx


# -----------------------------
# Pre/Post-processing
# -----------------------------
def normalize_slice_uint8_to_model(x: np.ndarray, normalize_type: str) -> np.ndarray:
    """
    x: HxW float32 in native intensity space.
    For "medsam": min-max to [0,1] per slice.
    For "sam"   : ImageNet-like norm can be done in the model; here we do 0..1 as input then rely on SAM internals.
    """
    if normalize_type.lower() == "medsam":
        vmin, vmax = np.percentile(x, 0.5), np.percentile(x, 99.5)
        x = np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        # safe default: simple min-max per slice
        vmin, vmax = x.min(), x.max()
        x = (x - vmin) / (vmax - vmin + 1e-8) if vmax > vmin else x * 0
    return x


def resize_to_1024(x01: np.ndarray) -> np.ndarray:
    """
    Resize HxW float (0..1) to 1024x1024 with bilinear.
    """
    return cv2.resize(x01, (1024, 1024), interpolation=cv2.INTER_LINEAR)


def upsample_mask_to_hw(mask_01_1024: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Resize 0/1 mask from 1024x1024 back to original HxW with nearest.
    """
    return cv2.resize(
        mask_01_1024.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
    )


def keep_largest_component_3d(mask_zyx: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component (3D).
    """
    sitk_img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    cc = sitk.ConnectedComponent(sitk_img)
    lab = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.Equal(lab, 1)
    return sitk.GetArrayFromImage(largest).astype(np.uint8)


# -----------------------------
# SAM forward (prompt-free)
# -----------------------------
@torch.no_grad()
def predict_volume(
    sam,
    vol: tio.ScalarImage,
    device: torch.device,
    normalize_type: str,
    flip_R: bool = True,
    apply_lcc: bool = True,
) -> np.ndarray:
    """
    Slice-by-slice inference:
      - Normalize and resize each slice to 1024x1024
      - Replicate to 3 channels
      - Forward SAM in prompt-free mode
      - Argmax over mask logits (binary)
      - Upsample back to original HxW
      - Stack to Z,Y,X volume
    """
    # TorchIO data tensor: (C,Z,Y,X), here C=1
    vol_tensor = vol.data  # torch.Tensor
    assert (
        vol_tensor.ndim == 4 and vol_tensor.shape[0] == 1
    ), "Expect single-channel volumes."

    C, Z, H, W = vol_tensor.shape
    vol_np = vol_tensor[0].numpy().astype(np.float32)  # Z,H,W

    # optional LR flip for 'R' orientation for consistent handedness
    if flip_R:
        vol_np = maybe_flip_lr_for_R_orientation(vol, vol_np)

    masks_zyx = np.zeros_like(vol_np, dtype=np.uint8)

    # Some SAM wrappers split to (image_encoder, prompt_encoder, mask_decoder).
    # We'll try a generic forward signature similar to the notebook:
    #   - get image embedding for the 1024 image
    #   - feed prompt=None and get masks
    # Adjust if your model exposes a different API.

    for z in range(Z):
        slice_f = vol_np[z]  # HxW
        slice_n = normalize_slice_uint8_to_model(slice_f, normalize_type)
        inp_1024 = resize_to_1024(slice_n)  # 1024x1024 float

        # 3-channel replicate
        img_3ch = np.stack([inp_1024, inp_1024, inp_1024], axis=0)  # 3,1024,1024
        img_t = (
            torch.from_numpy(img_3ch)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

        # ----- SAM forward (prompt-free) -----
        # The exact API may vary. Two common patterns:

        # Pattern A: single-call predict function (pseudo)
        # masks, scores, logits = sam.predict(img_t)   # <- edit to your repo

        # Pattern B: explicit encoder/decoder calls
        try:
            # Try SAM-like API used in many forks:
            # 1) encode image
            img_embed = sam.image_encoder(img_t)
            # 2) no prompts
            sparse_embeds, dense_embeds = None, None
            # 3) decode masks (some apis expect h/w or orig size; adjust if needed)
            mask_logits, _ = sam.mask_decoder(
                image_embeddings=img_embed,
                image_pe=sam.prompt_encoder.get_dense_pe(),  # positional enc
                sparse_prompt_embeddings=sparse_embeds,
                dense_prompt_embeddings=dense_embeds,
                multimask_output=False,
            )
        except Exception:
            # Fallback: if your repo has a convenience method, replace here:
            raise RuntimeError(
                "Adjust the forward pass to your fine-tuned SAM API (image_encoder/prompt_encoder/mask_decoder or predict())."
            )

        # mask_logits: [B,1,256,256] or [B,1,1024,1024], depends on impl
        logit = mask_logits.squeeze(0).squeeze(0)  # HxW (downsampled or 1024)
        # Some decoders output 256x256; upsample to 1024 for consistent path
        if logit.shape[-1] != 1024:
            logit = F.interpolate(
                logit[None, None],
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        prob = torch.sigmoid(logit)
        mask_1024 = (prob > 0.5).float().cpu().numpy()
        mask_hw = upsample_mask_to_hw(mask_1024, H, W)  # back to native size
        masks_zyx[z] = mask_hw.astype(np.uint8)

    if apply_lcc:
        masks_zyx = keep_largest_component_3d(masks_zyx)

    return masks_zyx  # Z,Y,X uint8


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Batch 3D predictions (SAM fine-tuned) without masks."
    )
    ap.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Folder containing checkpoint_best.pth and args.json",
    )
    ap.add_argument(
        "--test_dir",
        type=Path,
        required=True,
        help="Folder with one subfolder per CASE_ID, each containing at least one .nii(.gz) image",
    )
    ap.add_argument(
        "--predict_dir",
        type=Path,
        required=True,
        help="Output folder for predictions (pred_mask.nii.gz per case)",
    )
    ap.add_argument(
        "--prefer",
        type=str,
        default="t1*ce*,T1*CE*",
        help="Comma-separated lowercase patterns to prefer for picking the image (e.g., 't1*ce*,post*')",
    )
    ap.add_argument(
        "--no_flip_R",
        action="store_true",
        help="Disable LR flip when orientation[0]=='R'",
    )
    ap.add_argument(
        "--no_lcc",
        action="store_true",
        help="Disable keep-largest-component postprocess",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir
    args_json = ckpt_dir / "args.json"
    ckpt_path = ckpt_dir / "checkpoint_best.pth"
    if not args_json.exists() or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing args.json or checkpoint_best.pth in {ckpt_dir}"
        )

    device = torch.device(args.device)
    sam, normalize_type, arch, finetune_type = build_sam_model(
        args_json, ckpt_path, device
    )
    print(
        f"[info] Loaded SAM arch={arch}, finetune_type={finetune_type}, normalize={normalize_type}, device={device}"
    )

    prefer_patterns = [p.strip().lower() for p in args.prefer.split(",") if p.strip()]
    cases = [d for d in sorted(args.test_dir.iterdir()) if d.is_dir()]
    if not cases:
        raise RuntimeError(f"No case folders found in {args.test_dir}")

    for case_dir in cases:
        case_id = case_dir.name
        out_dir = args.predict_dir / case_id
        out_mask = out_dir / "pred_mask.nii.gz"
        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = find_image_path(case_dir, prefer_patterns)
        if img_path is None:
            print(f"[warn] {case_id}: no .nii/.nii.gz found, skipping")
            continue

        print(f"[case] {case_id}  image={img_path.name}")
        vol = read_volume_tio(img_path)

        mask_zyx = predict_volume(
            sam,
            vol,
            device=device,
            normalize_type=normalize_type,
            flip_R=(not args.no_flip_R),
            apply_lcc=(not args.no_lcc),
        )

        # save with same affine/orientation as input
        save_mask_like(vol, mask_zyx, out_mask)
        print(f"[ok] wrote {out_mask}")

    print("[done] All cases processed.")


if __name__ == "__main__":
    main()
