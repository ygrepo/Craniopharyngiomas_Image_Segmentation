#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch, torch.nn.functional as F
import cv2, nibabel as nib
import torch

from typing import Optional
import re
from nibabel.processing import resample_from_to

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
    import torch, numpy as np

    def devobj(d):
        return torch.device(d if d != "cuda" else "cuda:0")

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float().cpu()
        return torch.from_numpy(np.asarray(x)).float().cpu()

    sd = torch.load(ckpt_path, map_location=devobj(device))
    # unwrap once
    for k in ("state_dict", "model", "net", "module", "teacher", "student"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    if not isinstance(sd, dict):
        raise ValueError(f"[smart_load] Unexpected checkpoint format: {type(sd)}")

    # try direct load first (in case it’s already SAM-style)
    msg_direct = model.load_state_dict(sd, strict=False)
    inter_direct = set(sd.keys()) & set(model.state_dict().keys())
    if len(inter_direct) > 1000:
        logger.info(f"[smart_load] direct load ok: {msg_direct}")
        return msg_direct, sd  # <-- RETURN THE SD WE USED

    # ---- remap path (MAE/DINO → SAM) ----
    target_tokens = image_size // vit_patch_size

    def fix_pos_embed(arr):
        """(1, L[+1], C) → (1, H, W, C), resized to (target_tokens, target_tokens)."""
        t = to_tensor(arr)
        if t.ndim != 3 or t.shape[0] != 1:
            return t
        L, C = t.shape[1], t.shape[2]
        if L <= 1:  # degenerate
            return t
        # prefer L-1 as square (drop CLS), else try L
        g = int((L - 1) ** 0.5)
        if g * g == (L - 1):
            t = t[:, 1:, :]  # drop CLS
            L = L - 1
        else:
            g = int(L**0.5)
            if g * g != L:
                return t  # not a flat grid
        t = t.view(1, g, g, C)  # 1,H,W,C
        tchw = t.permute(0, 3, 1, 2)  # 1,C,H,W
        tchw = torch.nn.functional.interpolate(
            tchw,
            size=(target_tokens, target_tokens),
            mode="bilinear",
            align_corners=False,
        )
        t = tchw.permute(0, 2, 3, 1)  # 1,H,W,C
        return t

    new_sd = {}
    drop_prefixes = ("decoder", "mask_decoder", "dino_head")  # heads not used

    for k, v in sd.items():
        if k.startswith(drop_prefixes):
            continue

        # backbone.* → image_encoder.*
        nk = k.replace("backbone.", "image_encoder.")
        # mlp.fc1/fc2 → mlp.lin1/lin2
        nk = nk.replace(".mlp.fc1.", ".mlp.lin1.")
        nk = nk.replace(".mlp.fc2.", ".mlp.lin2.")
        # collapse double numeric: blocks.0.0.* → blocks.0.*
        parts = nk.split(".")
        if len(parts) > 3 and parts[2].isdigit() and parts[3].isdigit():
            nk = ".".join([p for i, p in enumerate(parts) if i != 2])

        # pos_embed special case
        if "pos_embed" in k:
            v = fix_pos_embed(v)
        else:
            v = to_tensor(v)

        # ensure encoder prefix if obviously encoder-like
        if not nk.startswith("image_encoder.") and any(
            s in nk for s in ("pos_embed", "patch_embed", "blocks", "neck")
        ):
            nk = "image_encoder." + nk

        new_sd[nk] = v

    msg = model.load_state_dict(new_sd, strict=False)
    logger.info(f"[smart_load] remap load: {msg}")
    return msg, new_sd  # <-- RETURN THE SD WE USED


def load_report(model, used_sd: dict):
    """Report overlap/coverage against the *actual* SD used to load."""
    mkeys = set(model.state_dict().keys())
    ckeys = set(used_sd.keys())
    inter = mkeys & ckeys
    loaded_params = sum(model.state_dict()[k].numel() for k in inter)
    total_params = sum(p.numel() for p in model.parameters())
    miss = sorted(list(mkeys - ckeys))[:20]
    unexp = sorted(list(ckeys - mkeys))[:20]
    logger.info(
        f"[coverage] loaded_params={loaded_params:,} / total={total_params:,} ({100.0*loaded_params/total_params:.2f}%)"
    )
    logger.info(
        f"[missing prefixes]:{sorted({m.split('.')[0] for m in (mkeys - ckeys)})}"
    )
    logger.info(f"[missing sample]:{miss}")
    logger.info(
        f"[unexpected prefixes]:{sorted({u.split('.')[0] for u in (ckeys - mkeys)})}"
    )
    logger.info(f"[unexpected sample]:{unexp}")


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
    setdef("num_cls", 3)
    setdef("mask_decoder_type", "default")
    setdef("prompt_type", "none")
    return ns


REF_PATTERNS = [
    r".*_ALIGNED\.nii\.gz$",
    r".*T1.*(ALIGNED|RAS|coreg).*\.nii\.gz$",
    r".*T1_CE.*\.nii\.gz$",
    r".*T1.*\.nii\.gz$",
]


def find_ref_nii(case_dir: Path) -> Optional[Path]:
    # 1) look in case_dir
    cands = list(case_dir.glob("*.nii.gz"))
    # 2) also look in a sibling/parent 'output/preprocessed/<case>/' if it exists
    more = []
    parent = case_dir.parent
    case_id = case_dir.name
    for p in [
        parent / case_id,  # same level
        parent / "output" / "preprocessed" / case_id,
        parent / ".." / "output" / "preprocessed" / case_id,
    ]:
        p = p.resolve()
        if p.exists() and p.is_dir():
            more.extend(p.glob("*.nii.gz"))
    cands.extend(more)

    # rank by patterns
    scored = []
    for f in cands:
        name = f.name
        score = -1
        for i, pat in enumerate(REF_PATTERNS):
            if re.search(pat, name, flags=re.IGNORECASE):
                score = max(
                    score, len(REF_PATTERNS) - i
                )  # earlier pattern = higher score
        if score >= 0:
            scored.append((score, f))
    if not scored:
        return None
    scored.sort(reverse=True)  # highest score first
    return scored[0][1]


@torch.no_grad()
def predict_case(
    case_dir: Path,
    out_dir: Path,
    model,
    device: torch.device,
    image_size: int = 1024,
):
    from nibabel.processing import resample_from_to

    imgs_dir = case_dir / "images"
    meta_p = case_dir / "meta.json"
    if not imgs_dir.exists() or not meta_p.exists():
        logger.warning(f"{case_dir.name}: missing images/ or meta.json — skip")
        return

    meta = json.loads(meta_p.read_text())
    X, Y, Z = meta["shape_xyz"]
    src_affine = np.array(meta["affine"], dtype=np.float32)
    export_axis = int(meta.get("export_axis", 2))
    assert export_axis == 2, "This script assumes axial export along axis=2."

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_vol = np.zeros((X, Y, Z), dtype=np.uint8)

    # find per-case reference
    ref_p = find_ref_nii(case_dir)
    if ref_p is None:
        logger.error(
            f"{case_dir.name}: no reference NIfTI found (expected an *ALIGNED*.nii.gz); not saving."
        )
        return
    ref_img = nib.load(str(ref_p))

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

        # BGR -> RGB, resize to encoder size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # to tensor, normalize
        xt = (
            torch.from_numpy(inp.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(device, dtype=torch.float32)
        )
        xt = (xt - pixel_mean) / pixel_std

        # SAM-like forward (multimask + IoU selection)
        img_emb = model.image_encoder(xt)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        logits, iou_pred = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        idx = int(torch.argmax(iou_pred))
        logit = logits[:, idx : idx + 1]
        if logit.shape[-1] != image_size:
            logit = F.interpolate(
                logit,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )

        prob = torch.sigmoid(logit)[0, 0].detach().cpu().numpy()
        m, s = float(prob.mean()), float(prob.std())
        logger.debug(f"[slice {z:04d}] prob mean={m:.4f} std={s:.4f}")

        # threshold at native resolution, then back to original slice size
        mask_1024 = (prob > 0.35).astype(np.uint8)
        mask_hw = cv2.resize(mask_1024, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # place into (X,Y,Z) as axial slice z
        if (w0, h0) != (X, Y):
            mask_hw = cv2.resize(mask_hw, (X, Y), interpolation=cv2.INTER_NEAREST)
        mask_vol[..., z] = mask_hw

    # ---- post-loop: sanity & save ----
    frac_on = float(mask_vol.mean())
    logger.info(f"[{case_dir.name}] mask fraction ON = {frac_on:.2%}")
    if frac_on > 0.50:
        logger.error(
            f"[{case_dir.name}] mask is {frac_on:.1%} ON — likely invalid; not saving."
        )
        return

    # If shape differs from reference, resample to ref grid (NN)
    src_img = nib.Nifti1Image(
        mask_vol.astype(np.uint8), src_affine
    )  # use meta affine as source
    if src_img.shape != ref_img.shape or not np.allclose(
        src_img.affine, ref_img.affine
    ):
        logger.info(
            f"{case_dir.name}: resampling mask {src_img.shape} -> {ref_img.shape}"
        )
        target = (ref_img.shape, ref_img.affine)
        src_img = resample_from_to(src_img, target, order=0)  # NN preserves labels

    # Save with ref header/affine; labelmap + q/sforms
    out_img = nib.Nifti1Image(
        src_img.get_fdata().astype(np.uint8), ref_img.affine, ref_img.header.copy()
    )
    out_img.set_data_dtype(np.uint8)
    out_img.set_qform(ref_img.affine, code=1)
    out_img.set_sform(ref_img.affine, code=1)
    nib.save(out_img, str(out_dir / "pred_mask.nii.gz"))
    logger.info(
        f"[ok] {case_dir.name} → {out_dir/'pred_mask.nii.gz'} (ref={ref_p.name})"
    )


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
        "--sam_ckpt",
        type=Path,
        required=True,  # required because we need the heads
        help="Official SAM ViT-B checkpoint to seed prompt/decoder/neck (e.g., sam_vit_b.pth)",
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
            num_classes=3,
            image_size=args.image_size,
            pretrained_sam=False,  # use MRI-CORE weights, not SAM
        )
        .eval()
        .to(args.device)
    )
    mt = model.mask_decoder.mask_tokens.weight
    logger.info(f"mask_tokens shape: {tuple(mt.shape)}")
    assert mt.shape[0] == 4, f"Expected 4 mask tokens, got {mt.shape[0]}"

    # ---- 1) Seed prompt/decoder/neck from official SAM ViT-B ----
    sd_sam = torch.load(
        str(args.sam_ckpt),
        map_location=torch.device(args.device if args.device != "cuda" else "cuda:0"),
    )

    # unwrap common containers
    for k in ("state_dict", "model", "net", "module", "teacher", "student"):
        if isinstance(sd_sam, dict) and k in sd_sam and isinstance(sd_sam[k], dict):
            sd_sam = sd_sam[k]
            break

    # load (non-strict: SAM ckpts sometimes miss tiny buffers)
    msg_sam = model.load_state_dict(sd_sam, strict=False)
    logger.info("[sam seed] loaded SAM heads: %s", msg_sam)

    msg, used_sd = smart_load(
        model,
        str(args.checkpoint),
        device=args.device,
        strict=False,
        image_size=args.image_size,
        vit_patch_size=16,
    )
    model.eval()
    sanity(model)
    load_report(model, used_sd)

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
