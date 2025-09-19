#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import cv2, nibabel as nib

# import from repo
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)

sys.path.append("../mri_foundation")
from models.sam import sam_model_registry


def _ensure_defaults(ns, image_size: int = 1024):
    def setdef(k, v):
        if not hasattr(ns, k):
            setattr(ns, k, v)

    # encoder/decoder update & adapters OFF for pretrained inference
    setdef("if_update_encoder", False)
    setdef("if_update_decoder", False)
    setdef("if_encoder_adapter", False)
    setdef("encoder_adapter_depths", [])  # not used when adapter off
    setdef("if_mask_decoder_adapter", False)
    setdef("if_prompt_adapter", False)
    setdef("if_low_rank_adapter", False)
    setdef("low_rank_rank", 8)  # harmless default
    setdef("decoder_adapt_depth", 0)
    # normalization used by MRI-CORE; 'slice_norm' is their common default
    setdef("normalize_type", "slice_norm")
    setdef("image_size", image_size)
    setdef("num_cls", 1)  # binary
    setdef("mask_decoder_type", "default")
    setdef("prompt_type", "none")
    return ns


#
@torch.no_grad()
def predict_case(
    case_dir: Path, out_dir: Path, model, device: torch.device, image_size: int = 1024
):
    imgs_dir = case_dir / "images"
    meta_p = case_dir / "meta.json"
    if not imgs_dir.exists() or not meta_p.exists():
        print(f"[warn] {case_dir.name}: missing images/ or meta.json, skip")
        return

    meta = json.loads(meta_p.read_text())
    X, Y, Z = meta["shape_xyz"]
    affine = np.array(meta["affine"], dtype=np.float32)
    export_axis = int(meta.get("export_axis", 2))
    assert export_axis == 2, "This script assumes axial export along axis=2."

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_vol = np.zeros((X, Y, Z), dtype=np.uint8)

    for png_p in sorted(imgs_dir.glob("*.png")):
        z = int(png_p.stem)  # 0000.png → 0
        img = cv2.imread(str(png_p), cv2.IMREAD_COLOR)  # BGR, uint8 in [0,255]
        h0, w0 = img.shape[:2]

        # 1) BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2) resize to the model input size (keep uint8 0..255)
        inp = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # 3) to tensor and normalize with the model's pixel mean/std
        xt = (
            torch.from_numpy(inp.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(device, dtype=torch.float32)
        )  # (1,3,H,W)

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

        # write into (X,Y,Z) volume as axial slice at z
        # png was HxW ~ equals YxX; nibabel shape is (X,Y,Z)
        if (w0, h0) != (X, Y):
            # safety: resize to exact NIfTI in-plane size if export changed it
            mask_hw = cv2.resize(mask_hw, (X, Y), interpolation=cv2.INTER_NEAREST)
        mask_vol[..., z] = mask_hw  # axial placement (axis=2)

    # save with original affine — no transpose
    out_nii = nib.Nifti1Image(mask_vol, affine)
    nib.save(out_nii, str(out_dir / "pred_mask.nii.gz"))
    print(f"[ok] {case_dir.name} → {out_dir/'pred_mask.nii.gz'}")


@torch.no_grad()
def predict_case(
    case_dir: Path, out_dir: Path, model, device: torch.device, image_size: int = 1024
):
    imgs_dir = case_dir / "images"
    meta_p = case_dir / "meta.json"
    if not imgs_dir.exists() or not meta_p.exists():
        logger.warning(f"[warn] {case_dir.name}: missing images/ or meta.json")
        return

    meta = json.loads(Path(meta_p).read_text())
    X, Y, Z = meta["shape_xyz"]
    affine = np.array(meta["affine"], dtype=np.float32)
    export_axis = int(meta.get("export_axis", 2))
    assert export_axis == 2, "This script assumes axial export along axis=2."

    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_vol = np.zeros((X, Y, Z), dtype=np.uint8)

    for png_p in sorted(imgs_dir.glob("*.png")):
        z = int(png_p.stem)  # 0000.png → 0
        img = cv2.imread(str(png_p), cv2.IMREAD_COLOR)  # HxW×3 uint8
        h0, w0 = img.shape[:2]
        img01 = img.astype(np.float32) / 255.0
        inp = cv2.resize(
            img01, (image_size, image_size), interpolation=cv2.INTER_LINEAR
        )
        xt = (
            torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        )  # 1,3,1024,1024

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

        # write into (X,Y,Z) volume as axial slice at z
        # png was HxW ~ equals YxX; nibabel shape is (X,Y,Z)
        if (w0, h0) != (X, Y):
            # safety: resize to exact NIfTI in-plane size if export changed it
            mask_hw = cv2.resize(mask_hw, (X, Y), interpolation=cv2.INTER_NEAREST)
        mask_vol[..., z] = mask_hw  # axial placement (axis=2)

    # save with original affine — no transpose
    out_nii = nib.Nifti1Image(mask_vol, affine)
    nib.save(out_nii, str(out_dir / "pred_mask.nii.gz"))
    logger.info(f"[ok] {case_dir.name} → {out_dir/'pred_mask.nii.gz'}")


def main():
    ap = argparse.ArgumentParser(description="MRI-CORE slice inference → 3D mask")
    ap.add_argument(
        "--slices_root",
        type=Path,
        required=True,
        help="output/slices from export_slices.py",
    )
    ap.add_argument(
        "--predict_root",
        type=Path,
        required=True,
        help="where to save pred_mask.nii.gz per case",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="pretrained_weights/MRI_CORE_*.pth",
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
        help="Logging level.",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Log file path (in addition to console).",
    )
    args = ap.parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)

    # build model like in README
    cfg = argparse.ArgumentParser()
    margs = cfg.parse_args([])  # empty to get default args object
    margs = _ensure_defaults(margs)

    model = (
        sam_model_registry[args.arch](
            margs,
            checkpoint=str(args.checkpoint),
            num_classes=1,
            image_size=args.image_size,
            pretrained_sam=False,
        )
        .eval()
        .to(args.device)
    )

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
