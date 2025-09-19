#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import cv2, nibabel as nib
import torch

# project logging
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)

# mri_foundation repo on path
sys.path.append("../mri_foundation")
from models.sam import sam_model_registry


def smart_load(
    model,
    ckpt_path: str,
    device: str = "cuda",
    strict: bool = False,
    image_size: int = 1024,
    vit_patch_size: int = 16,
):
    """
    Robust loader that:
      1) unwraps common containers (state_dict/model/net/module/teacher/student),
      2) tries direct load,
      3) if coverage is ~0, remaps MAE/DINO/MRI-CORE keys -> SAM names (image_encoder.*),
      4) handles pos_embed reshape to (1, H, W, C).
    """
    import torch, numpy as np

    sd = torch.load(
        ckpt_path, map_location=torch.device(device if device != "cuda" else "cuda:0")
    )

    # 1) unwrap
    for k in ("state_dict", "model", "net", "module", "teacher", "student"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    if not isinstance(sd, dict):
        raise ValueError(f"[smart_load] Unexpected checkpoint format: {type(sd)}")

    # strip single 'module.' if present
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # helper: coverage vs model
    def coverage_keys(sd_keys, model_keys):
        inter = set(sd_keys) & set(model_keys)
        return len(inter), inter

    # 2) try direct
    msg_direct = model.load_state_dict(sd, strict=False)
    inter_len, _ = coverage_keys(sd.keys(), model.state_dict().keys())
    if inter_len > 1000:  # good overlap -> done
        logger.info("[smart_load] direct load ok:", msg_direct)
        return msg_direct

    # 3) remap MRI-CORE/MAE/DINO -> SAM encoder
    new_sd = {}
    token_size = image_size // vit_patch_size

    # detect if pos_embed needs 2D reshape
    def fix_pos_embed(arr):
        # expect shape (1, L(+cls), C)
        if (
            arr.ndim == 3
            and arr.shape[0] == 1
            and (arr.shape[1] == token_size * token_size + 1)
        ):
            pe = arr[:, 1:, :]  # drop cls
            ts = int(np.sqrt(pe.shape[1]))
            pe = pe.reshape(1, ts, ts, arr.shape[2])  # 1,H,W,C
            # resize to target token grid
            pe_t = (
                torch.from_numpy(pe).permute(0, 3, 1, 2)
                if isinstance(pe, np.ndarray)
                else pe.permute(0, 3, 1, 2)
            )
            pe_t = torch.nn.functional.interpolate(
                pe_t,
                size=(token_size, token_size),
                mode="bilinear",
                align_corners=False,
            )
            pe = pe_t.permute(0, 2, 3, 1)  # back to 1,H,W,C
            return pe
        return arr

    for k, v in sd.items():
        if k.startswith("decoder") or k.startswith("mask_decoder"):
            continue  # ignore task heads
        nk = k

        # common renames from MRI-CORE/DINO to SAM
        nk = nk.replace("backbone.", "image_encoder.")
        nk = nk.replace("fc", "lin")  # mlp.fc -> mlp.lin
        nk = nk.replace("patch_embed.proj.", "patch_embed.proj.")
        nk = nk.replace("norm1", "norm1")
        nk = nk.replace("norm2", "norm2")

        # some DINOv2 dumps have chunked indices like image_encoder.0.0... collapse those
        parts = nk.split(".")
        if len(parts) > 3 and parts[2].isdigit() and parts[3].isdigit():
            nk = ".".join([p for i, p in enumerate(parts) if i != 2])

        # pos_embed special-case: SAM expects 1,H,W,C (later transposed inside)
        if "pos_embed" in k and "image_encoder" in nk:
            v = fix_pos_embed(
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            )

        # ensure all encoder weights live under image_encoder.*
        if not nk.startswith("image_encoder."):
            if "pos_embed" in k or "patch_embed" in k or "blocks" in k or "neck" in k:
                nk = "image_encoder." + nk

        new_sd[nk] = v

    # 4) load remapped
    msg = model.load_state_dict(new_sd, strict=False)

    # report
    model_keys = set(model.state_dict().keys())
    inter_len2, _ = coverage_keys(new_sd.keys(), model_keys)
    total = sum(p.numel() for p in model.parameters())
    nanp = sum(torch.isnan(p).sum().item() for p in model.parameters())
    logger.info(
        f"[smart_load] remap load: intersect={inter_len2}  params={total:,}  NaNs={nanp}"
    )
    logger.info(f"[smart_load] msg: {msg}")
    if inter_len2 < 1000:
        logger.warning(
            "[smart_load][warn] Very low overlap after remap — this checkpoint may not match the SAM ViT-B encoder. "
            "Try the other file (e.g., pretrained_weights/mri_foundation.pth) or confirm this is a SAM/MRI-CORE encoder dump."
        )
    return msg


def load_report(model, sd_path, device):
    sd = torch.load(
        sd_path, map_location=torch.device(device if device != "cuda" else "cuda:0")
    )
    for k in ("state_dict", "model", "net", "module", "teacher", "student"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    inter = model_keys & ckpt_keys

    # coverage
    loaded_params = sum(model.state_dict()[k].numel() for k in inter)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[coverage] loaded_params={loaded_params:,} / total={total_params:,} "
        f"({100.0*loaded_params/total_params:.2f}%)"
    )

    # show a few missing/unexpected prefixes
    miss = list(model_keys - ckpt_keys)
    unexp = list(ckpt_keys - model_keys)

    def head(xs):
        return sorted(xs)[:15]

    logger.info("[missing prefixes]", sorted({m.split(".")[0] for m in miss}))
    logger.info("[missing sample]", head(miss))
    logger.info("[unexpected prefixes]", sorted({u.split(".")[0] for u in unexp}))
    logger.info("[unexpected sample]", head(unexp))


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
        logger.info(
            f"logit mean/std: {float(logit.mean()):.4f} / {float(logit.std()):.4f}"
        )
        prob = torch.sigmoid(logit)[0, 0].cpu().numpy()

        mask_1024 = (prob > 0.35).astype(np.uint8)
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
    import torch

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
            checkpoint=None,
            num_classes=1,
            image_size=args.image_size,
            pretrained_sam=False,  # use MRI-CORE weights, not SAM
        )
        .eval()
        .to(args.device)
    )
    smart_load(
        model,
        str(args.checkpoint),
        device=args.device,
        strict=False,
        image_size=args.image_size,
        vit_patch_size=16,
    )
    model.eval()
    sanity(model)
    load_report(model, str(args.checkpoint), args.device)

    # Inspect checkpoint keys (after unwrapping)
    import torch

    ckpt_path = str(args.checkpoint)
    sd = torch.load(
        ckpt_path,
        map_location=torch.device(args.device if args.device != "cuda" else "cuda:0"),
    )
    for k in ("state_dict", "model", "net", "module", "teacher", "student"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break

    keys = list(sd.keys())
    sample = sorted(keys)[:50]
    logger.info("Checkpoint: %s | total keys: %d", ckpt_path, len(keys))
    logger.info("Sample keys (first %d): %s", len(sample), sample)

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
