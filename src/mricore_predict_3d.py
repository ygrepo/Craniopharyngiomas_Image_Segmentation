#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import cv2, nibabel as nib

# project logging
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)

# mri_foundation repo on path
sys.path.append("../mri_foundation")
from models.sam import sam_model_registry


def _ensure_defaults(ns, image_size: int = 1024):
    def setdef(k, v):
        if not hasattr(ns, k):
            setattr(ns, k, v)

    # inference-only; adapters off
    setdef("if_update_encoder", False)
    setdef("if_update_decoder", False)
    setdef("if_encoder_adapter", False)
    setdef("encoder_adapter_depths", [])
    setdef("if_mask_decoder_adapter", False)
    setdef("decoder_adapt_depth", 0)
    setdef("if_prompt_adapter", False)
    setdef("if_low_rank_adapter", False)
    setdef("low_rank_rank", 8)
    # misc expected fields
    setdef("normalize_type", "slice_norm")
    setdef("image_size", image_size)
    setdef("num_cls", 1)
    setdef("mask_decoder_type", "default")
    setdef("prompt_type", "none")
    return ns


@torch.no_grad()
def predict_case(
    case_dir: Path, out_dir: Path, model, device: torch.device, image_size: int = 1024
):
    imgs_dir = case_dir / "images"
    meta_p = case_dir / "meta.json"
    if not imgs_dir.exists() or not meta_p.exists():
        logger.warning(f"{case_dir.name}: missing images/ or meta.json — skip")
        return

    meta = json.loads(meta_p.read_text())
    X, Y, Z = meta["shape_xyz"]
    affine = np.array(meta["affine"], dtype=np.float32)
    export_axis = int(meta.get("export_axis", 2))
    assert export_axis == 2, "This script assumes axial export along axis=2."

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_vol = np.zeros((X, Y, Z), dtype=np.uint8)

    # model normalization (SAM-style) expects RGB 0..255 then (x-mean)/std
    pixel_mean = torch.tensor(
        getattr(model, "pixel_mean", [123.675, 116.28, 103.53]),
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    pixel_std = torch.tensor(
        getattr(model, "pixel_std", [58.395, 57.12, 57.375]),
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)

    for png_p in sorted(imgs_dir.glob("*.png")):
        z = int(png_p.stem)  # 0000.png → 0
        img = cv2.imread(str(png_p), cv2.IMREAD_COLOR)  # BGR uint8 [0,255]
        if img is None:
            logger.warning(f"{case_dir.name}: cannot read {png_p}, skipping slice")
            continue
        h0, w0 = img.shape[:2]

        # BGR -> RGB, resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # to tensor, normalize
        xt = (
            torch.from_numpy(inp.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(device, dtype=torch.float32)
        )
        xt = (xt - pixel_mean) / pixel_std

        # SAM-like forward
        img_emb = model.image_encoder(xt)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        logits, _ = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        logit = logits[:, 0:1]
        if logit.shape[-1] != image_size:
            logit = F.interpolate(
                logit,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
        prob = torch.sigmoid(logit)[0, 0].cpu().numpy()

        mask_1024 = (prob > 0.5).astype(np.uint8)
        mask_hw = cv2.resize(mask_1024, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # place into (X,Y,Z) as axial slice z
        if (w0, h0) != (X, Y):
            mask_hw = cv2.resize(mask_hw, (X, Y), interpolation=cv2.INTER_NEAREST)
        mask_vol[..., z] = mask_hw

    # save with original affine (no transpose)
    out_nii = nib.Nifti1Image(mask_vol, affine)
    nib.save(out_nii, str(out_dir / "pred_mask.nii.gz"))
    logger.info(f"[ok] {case_dir.name} → {out_dir/'pred_mask.nii.gz'}")


def sanity(model):
    total = sum(p.numel() for p in model.parameters())
    nanp = sum(torch.isnan(p).sum().item() for p in model.parameters())
    logger.info(f"[model] params={total:,}  NaNs={nanp}")


def main():
    ap = argparse.ArgumentParser(description="MRI-CORE slice inference → 3D mask")
    ap.add_argument(
        "--slices_root",
        type=Path,
        required=True,
        help="Root from export_slices.py (cases with images/ + meta.json)",
    )
    ap.add_argument(
        "--predict_root",
        type=Path,
        required=True,
        help="Where to save pred_mask.nii.gz per case",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="MRI-CORE checkpoint, e.g. pretrained_weights/MRI_CORE_vit_b.pth",
    )
    ap.add_argument(
        "--arch", type=str, default="vit_b", choices=["vit_b", "vit_t", "vit_h"]
    )
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    ap.add_argument("--log_file", type=Path, default=None)
    args = ap.parse_args()

    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(
        f"Device: {args.device} | Checkpoint: {args.checkpoint} | Arch: {args.arch}"
    )

    # Build minimal args namespace for the model
    margs = argparse.Namespace()
    margs = _ensure_defaults(margs, image_size=args.image_size)

    model = (
        sam_model_registry[args.arch](
            margs,
            checkpoint=str(args.checkpoint),
            num_classes=1,
            image_size=args.image_size,
            pretrained_sam=False,  # use MRI-CORE weights, not SAM
        )
        .eval()
        .to(args.device)
    )
    sanity(model)

    cases = [d for d in sorted(args.slices_root.iterdir()) if d.is_dir()]
    for c in cases:
        predict_case(
            c,
            args.predict_root / c.name,
            model,
            torch.device(args.device),
            image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
