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
    mask_xyz = np.transpose(mask_zyx.astype(np.uint8), (2, 1, 0))  # Z,Y,X -> X,Y,Z
    nii = nib.Nifti1Image(mask_xyz, affine, nib.Nifti1Header(ref_img.header.copy()))
    nii.set_data_dtype(np.uint8)
    nii.set_qform(affine, code=1)
    nii.set_sform(affine, code=1)
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
    vol: tio.ScalarImage,  # TorchIO image: (C,Z,Y,X), C=1
    device: torch.device,
    normalize_type: str = "robust",  # "robust" or "clahe" or "none"
    flip_R: bool = True,
    apply_lcc: bool = True,
    return_prob: bool = True,
    threshold: float = 0.35,
    use_tta: bool = False,  # simple H/V flip TTA
):
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from scipy import ndimage as ndi

    # ---- helpers ----
    def _sam_pixel_stats(sam):
        pm = getattr(sam, "pixel_mean", [123.675, 116.28, 103.53])
        ps = getattr(sam, "pixel_std", [58.395, 57.12, 57.375])
        pm = torch.tensor(pm, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        ps = torch.tensor(ps, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        return pm, ps

    def _normalize_slice_uint8(gray: np.ndarray, how: str) -> np.ndarray:
        """Return uint8 in [0,255]."""
        if how == "clahe":
            g = gray.astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            g = clahe.apply(g)
            return g
        elif how == "robust":
            g = gray.astype(np.float32)
            lo, hi = np.percentile(g, [2, 98])
            g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
            return g.astype(np.uint8)
        else:
            # assume already roughly 0..255
            g = gray
            if g.dtype != np.uint8:
                g = np.clip(g, 0, 255).astype(np.uint8)
            return g

    def _resize_to_1024(img_uint8_hw: np.ndarray) -> np.ndarray:
        return cv2.resize(img_uint8_hw, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    def _upsample_mask(mask_1024: np.ndarray, H: int, W: int) -> np.ndarray:
        return cv2.resize(
            mask_1024.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )

    def _sam_logits_best(sam, xt: torch.Tensor) -> torch.Tensor:
        """
        Run SAM with multimask_output=True and select the mask with highest IoU prediction.
        xt: [1,3,1024,1024] float32, already (x - mean)/std
        Returns logits [1,1,1024,1024].
        """
        img_emb = sam.image_encoder(xt)
        sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
        logits, iou_pred = sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        idx = int(torch.argmax(iou_pred))
        logit = logits[:, idx : idx + 1]
        if logit.shape[-1] != 1024:  # some impls output 256
            logit = F.interpolate(
                logit, size=(1024, 1024), mode="bilinear", align_corners=False
            )
        return logit

    def _sam_logits_best_tta(sam, xt: torch.Tensor) -> torch.Tensor:
        """Simple TTA over H/V flips; average logits back in original orientation."""
        lgts = []
        for tf in ("none", "flip_h", "flip_v", "flip_hv"):
            x = xt
            if tf == "flip_h":
                x = torch.flip(x, dims=[3])
            if tf == "flip_v":
                x = torch.flip(x, dims=[2])
            if tf == "flip_hv":
                x = torch.flip(x, dims=[2, 3])
            l = _sam_logits_best(sam, x)
            if tf == "flip_h":
                l = torch.flip(l, dims=[3])
            if tf == "flip_v":
                l = torch.flip(l, dims=[2])
            if tf == "flip_hv":
                l = torch.flip(l, dims=[2, 3])
            lgts.append(l)
        return torch.mean(torch.stack(lgts, dim=0), dim=0)

    def _largest_component_3d(mask_zyx: np.ndarray) -> np.ndarray:
        lbl, nlab = ndi.label(mask_zyx > 0)
        if nlab <= 1:
            return (mask_zyx > 0).astype(np.uint8)
        sizes = ndi.sum(mask_zyx, lbl, index=np.arange(1, nlab + 1))
        keep = 1 + int(np.argmax(sizes))
        return (lbl == keep).astype(np.uint8)

    def _maybe_flip_R(vol_tio: tio.ScalarImage, vol_np_zyx: np.ndarray) -> np.ndarray:
        return (
            maybe_flip_lr_for_R_orientation(vol_tio, vol_np_zyx)
            if flip_R
            else vol_np_zyx
        )

    # ---- load volume ----
    vol_tensor = vol.data
    assert vol_tensor.ndim == 4 and vol_tensor.shape[0] == 1, "Expect (C=1, Z, Y, X)."
    _, Z, H, W = vol_tensor.shape
    vol_np = vol_tensor[0].cpu().numpy().astype(np.float32)  # Z,H,W
    vol_np = _maybe_flip_R(vol, vol_np)

    masks_zyx = np.zeros((Z, H, W), dtype=np.uint8)
    prob_zyx = np.zeros((Z, H, W), dtype=np.float32)

    pixel_mean, pixel_std = _sam_pixel_stats(sam)

    # ---- per-slice inference ----
    for z in range(Z):
        # 1) grayscale slice -> robust/CLAHE -> 1024
        sl = vol_np[z]  # HxW float32
        sl_u8 = _normalize_slice_uint8(sl, normalize_type)
        sl_1024 = _resize_to_1024(sl_u8)  # 1024x1024 uint8

        # 2) replicate to 3ch and apply SAM pixel normalization
        img_3ch = np.stack([sl_1024, sl_1024, sl_1024], axis=0).astype(
            np.float32
        )  # 3,1024,1024
        xt = (
            torch.from_numpy(img_3ch)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )  # 1,3,1024,1024
        xt = (xt - pixel_mean) / pixel_std

        # 3) logits via multimask + IoU best (with optional TTA)
        if use_tta:
            logit = _sam_logits_best_tta(sam, xt)
        else:
            logit = _sam_logits_best(sam, xt)

        prob = torch.sigmoid(logit)[0, 0].detach().cpu().numpy()  # 1024x1024 float
        # simple diagnostics (optional):
        # print(f"[z={z:03d}] mean={prob.mean():.4f} std={prob.std():.4f}")

        # 4) threshold to binary at 1024, then resize back to native HxW
        mask_1024 = (prob >= threshold).astype(np.uint8)
        mask_hw = _upsample_mask(mask_1024, H, W)
        prob_hw = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

        masks_zyx[z] = mask_hw
        prob_zyx[z] = prob_hw

    # ---- light post-processing ----
    if apply_lcc:
        masks_zyx = _largest_component_3d(masks_zyx)
        # small 3D closing to fill tiny holes
        masks_zyx = ndi.binary_closing(masks_zyx, structure=np.ones((3, 3, 3))).astype(
            np.uint8
        )

    if return_prob:
        return masks_zyx, prob_zyx
    return masks_zyx


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

        masks, probs = predict_volume(
            sam=sam,
            vol=vol,  # TorchIO ScalarImage
            device=device,
            normalize_type="robust",
            flip_R=True,
            apply_lcc=True,
            return_prob=True,
            threshold=0.35,
            use_tta=False,
        )
        # save with same affine/orientation as input
        save_mask_like(vol, masks, out_mask)
        print(f"[ok] wrote {out_mask}")

    print("[done] All cases processed.")


if __name__ == "__main__":
    main()
