from pathlib import Path
import nibabel as nib
import numpy as np
import json
import re
import random
from collections import defaultdict
import argparse

import SimpleITK as sitk
from tqdm import tqdm
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
    n4_bias_correct_np,
)

logger = get_logger(__name__)


# ---------- BraTS {0,1,2,4} -> {0,1,2,3} ----------
def remap_labels_to_0123(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.uint8)
    if (arr == 4).any():
        arr[arr == 4] = 3
    return arr


def _strip_ext(p: Path) -> str:
    s = p.name
    if s.endswith(".nii.gz"):
        return s[:-7]
    if s.endswith(".nii"):
        return s[:-4]
    return s


def _case_and_modality(path: Path) -> tuple[str, str | None]:
    """
    Parse BraTS-like base names:
      <case>_(flair|t1|t1ce|t2|seg)
    Returns (case_id, tag or None)
    """
    base = _strip_ext(path)
    m = re.match(r"^(.*)_(flair|t1|t1ce|t2|seg)$", base, flags=re.IGNORECASE)
    if not m:
        return (base, None)
    # normalize tag to lowercase
    return (m.group(1), m.group(2).lower())


def _save_image_to(
    path_in: Path, out_path: Path, run_n4: bool, n4_shrink: int, n4_iters: int
):
    nii = nib.load(str(path_in))
    data = nii.get_fdata().astype(np.float32)
    if run_n4:
        data = n4_bias_correct_np(data, shrink=n4_shrink, n_iters=n4_iters)
    out = nib.Nifti1Image(data, nii.affine, nii.header)
    out.set_data_dtype(np.float32)
    nib.save(out, str(out_path))


def convert_braTS_to_nnUNet(
    src_root: Path,
    dst_root: Path,
    *,
    dataset_id: int = 501,
    dataset_name: str = "BraTS2017_4ch",
    # Channel order for nnU-Net output
    modalities: tuple[str, ...] = ("flair", "t1", "t1ce", "t2"),
    split_ratio: tuple = (
        0.8,
        0.2,
    ),  # (train, test)  or provide (train, val, test) to merge val into train
    seed: int = 42,
    do_n4: bool = False,
    n4_shrink: int = 2,
    n4_iters: int = 50,
) -> Path:
    """
    Convert BraTS2017-style files to nnU-Net v2 raw structure.

    Input (any nested tree under src_root):
      <case>_flair.nii(.gz)
      <case>_t1.nii(.gz)
      <case>_t1ce.nii(.gz)
      <case>_t2.nii(.gz)
      <case>_seg.nii(.gz)

    Output:
      nnUNet_raw/Dataset<id>_<name>/{imagesTr,labelsTr,imagesTs}/
      dataset.json  (labels: {"0": "...", "1": "...", "2": "...", "3": "..."})
    """
    logger.info("Starting BraTS→nnU-Net conversion")

    # Validate split
    if len(split_ratio) not in (2, 3):
        raise ValueError("split_ratio must be (train, test) or (train, val, test).")
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")
    if len(split_ratio) == 3:
        train_frac = split_ratio[0] + split_ratio[1]  # merge val into train
        test_frac = split_ratio[2]
        split_ratio = (train_frac, test_frac)
        logger.info(
            f"Merging validation into training: train={train_frac:.3f}, test={test_frac:.3f}"
        )

    # ---- scan & group files by case_id ----
    all_files = list(src_root.rglob("*.nii")) + list(src_root.rglob("*.nii.gz"))
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    unknown = []
    for p in all_files:
        cid, tag = _case_and_modality(p)
        if tag is None:
            unknown.append(p)
            continue
        groups[cid][tag] = p

    case_ids = sorted(groups.keys())
    random.seed(seed)
    random.shuffle(case_ids)

    # ---- compute train/test split ----
    n_cases = len(case_ids)
    n_train = int(n_cases * split_ratio[0])
    train_ids = case_ids[:n_train]
    test_ids = case_ids[n_train:]

    # ---- prepare nnU-Net dirs ----
    out_dir = dst_root / f"Dataset{dataset_id}_{dataset_name}"
    imgTr = out_dir / "imagesTr"
    labTr = out_dir / "labelsTr"
    imgTs = out_dir / "imagesTs"
    for d in (imgTr, labTr, imgTs):
        d.mkdir(parents=True, exist_ok=True)

    # ---- write training cases (require all modalities + seg) ----
    kept_train, skipped_train = [], []
    for cid in tqdm(train_ids, desc="Writing training cases", unit="case"):
        have = {k.lower(): v for k, v in groups[cid].items()}
        if not all(m in have for m in modalities) or "seg" not in have:
            skipped_train.append((cid, sorted(have.keys())))
            continue
        # channels in the requested order
        for ch, m in enumerate(modalities):
            _save_image_to(
                have[m], imgTr / f"{cid}_{ch:04d}.nii.gz", do_n4, n4_shrink, n4_iters
            )
        # labels (remap 4->3)
        seg_nii = nib.load(str(have["seg"]))
        seg = remap_labels_to_0123(seg_nii.get_fdata()).astype(np.uint8)
        out_lbl = nib.Nifti1Image(seg, seg_nii.affine, seg_nii.header)
        out_lbl.set_data_dtype(np.uint8)
        nib.save(out_lbl, str(labTr / f"{cid}.nii.gz"))
        kept_train.append(cid)

    # ---- write test cases (require all modalities; no labels) ----
    kept_test, skipped_test = [], []
    for cid in tqdm(test_ids, desc="Writing test cases", unit="case"):
        have = {k.lower(): v for k, v in groups[cid].items()}
        if not all(m in have for m in modalities):
            skipped_test.append((cid, sorted(have.keys())))
            continue
        for ch, m in enumerate(modalities):
            _save_image_to(
                have[m], imgTs / f"{cid}_{ch:04d}.nii.gz", do_n4, n4_shrink, n4_iters
            )
        kept_test.append(cid)

    # ---- dataset.json (v2-friendly) ----
    kept_train_sorted = sorted(kept_train)
    kept_test_sorted = sorted(kept_test)

    channel_names = {
        "0": "FLAIR",
        "1": "T1",
        "2": "T1CE",
        "3": "T2",
    }
    modality_map = {str(i): "MRI" for i in range(len(modalities))}
    labels_int_to_name = {
        "0": "background",
        "1": "necrotic/non-enhancing",
        "2": "edema",
        "3": "enhancing",
    }

    ds = {
        "name": dataset_name,
        "description": f"BraTS2017-style; channels={list(modalities)}",
        "reference": "Local",
        "licence": "Research",
        "release": "1.0",
        "tensorImageSize": "3D",
        "file_ending": ".nii.gz",
        "channel_names": channel_names,
        "modality": modality_map,
        "labels": labels_int_to_name,
        "numTraining": len(kept_train_sorted),
        "numTest": len(kept_test_sorted),
        "training": [
            {
                "image": f"./imagesTr/{cid}_0000.nii.gz",
                "label": f"./labelsTr/{cid}.nii.gz",
            }
            for cid in kept_train_sorted
        ],
        "test": [f"./imagesTs/{cid}_0000.nii.gz" for cid in kept_test_sorted],
    }
    (out_dir / "dataset.json").write_text(json.dumps(ds, indent=2) + "\n")

    # ---- summary.txt ----
    summary_lines = [
        f"Dataset: Dataset{dataset_id}_{dataset_name}",
        f"Modalities (order): {list(modalities)}  -> channels 0000..{len(modalities)-1:04d}",
        f"N4: {do_n4} (shrink={n4_shrink}, iters={n4_iters})",
        f"Requested split: train={split_ratio[0]:.3f}, test={split_ratio[1]:.3f}",
        f"Found cases: {n_cases}",
        f"Kept: train={len(kept_train_sorted)}, test={len(kept_test_sorted)}",
    ]
    if skipped_train:
        summary_lines.append(
            f"Skipped train (incomplete): {len(skipped_train)} e.g. {skipped_train[0]}"
        )
    if skipped_test:
        summary_lines.append(
            f"Skipped test (incomplete): {len(skipped_test)} e.g. {skipped_test[0]}"
        )
    summary_lines.append("\n-- TRAIN CASE IDS --")
    summary_lines.extend(kept_train_sorted)
    summary_lines.append("\n-- TEST CASE IDS --")
    summary_lines.extend(kept_test_sorted)
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    logger.info(
        f"✅ Wrote {len(kept_train_sorted)} train and {len(kept_test_sorted)} test cases to {out_dir}"
    )
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description="Build nnU-Net v2 inputs (duplicate T1-CE into missing channels) and optionally run prediction."
    )
    ap.add_argument(
        "--src_root",
        type=Path,
        required=True,
        help="Folder with per-case subdirs that contain <CASEID>_0002.nii.gz (your current output/nifti).",
    )
    ap.add_argument(
        "--dst_root",
        type=Path,
        required=True,
        help="Where to write nnU-Net input cases (<CASEID>_0000..0003.nii.gz).",
    )
    ap.add_argument(
        "--dataset_id",
        "-d",
        type=str,
        default="002",
        help="Dataset ID of the installed BraTS-21 model (logger.infoed by installer; often 002).",
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

    setup_logging(Path(args.log_file), args.log_level)
    logger.info(f"Args: {args}")

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    convert_braTS_to_nnUNet(
        src_root=src_root,
        dst_root=dst_root,
        dataset_id=501,
        dataset_name="BraTS2017_4ch",
        modalities=("flair", "t1", "t1ce", "t2"),
        split_ratio=(0.8, 0.2),
        seed=42,
        do_n4=False,
    )
