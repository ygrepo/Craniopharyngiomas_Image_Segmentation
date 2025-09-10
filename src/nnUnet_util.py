from pathlib import Path
import nibabel as nib
import numpy as np
import json
import re
import random
from collections import defaultdict
import logging

import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------- Optional N4 bias field correction ----------
def n4_bias_correct_np(x: np.ndarray, shrink: int = 2, n_iters: int = 50) -> np.ndarray:
    img = sitk.GetImageFromArray(x.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)  # rough brain mask
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetShrinkFactor(shrink)
    n4.SetMaximumNumberOfIterations([n_iters])
    out = n4.Execute(img, mask)
    return sitk.GetArrayFromImage(out).astype(np.float32)


# ---------- Labels: {0,1,2,4} -> {0,1,2,3} ----------
def remap_labels(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.uint8)
    if 4 in np.unique(arr):
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
    base = _strip_ext(path)
    m = re.match(r"^(.*)-(t2f|t1c|t2w|t1n|seg)$", base)
    if not m:
        return (base, None)
    return (m.group(1), m.group(2))


def _save_image_to(path_in: Path, out_path: Path, run_n4: bool):
    nii = nib.load(path_in)
    data = nii.get_fdata().astype(np.float32)
    if run_n4:
        data = n4_bias_correct_np(data, shrink=n4_shrink, n_iters=n4_iters)
    nib.save(nib.Nifti1Image(data, nii.affine, nii.header), out_path)


def convert_braTS_to_nnUNet(
    src_root: Path,
    dst_root: Path,
    *,
    dataset_id: int = 501,
    dataset_name: str = "BraTS3M",
    modalities: tuple[str, ...] = ("t2f", "t1c", "t2w"),  # add "t1n" for 4 channels
    split_ratio: tuple = (
        0.8,
        0.2,
    ),  # (train, test). nnU-Net will do its own validation.
    seed: int = 42,
    do_n4: bool = False,
    n4_shrink: int = 2,
    n4_iters: int = 50,
    label_names: dict[int, str] | None = None,
    log_level: str = "INFO",
) -> Path:
    """
    Train/test-only converter for nnU-Net v2:
      - imagesTr/, labelsTr/ for training (nnU-Net will handle validation folds)
      - imagesTs/ for test (no labels)
      - dataset.json ("training", "test" only)
      - summary.txt with split details and case IDs
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if label_names is None:
        label_names = {
            0: "background",
            1: "necrotic/non-enhancing",
            2: "edema",
            3: "enhancing",
        }

    # allow 2- or 3-tuple; merge val into train if 3 given
    if len(split_ratio) not in (2, 3):
        raise ValueError("split_ratio must be (train, test) or (train, val, test).")
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")
    if len(split_ratio) == 3:
        train_frac = split_ratio[0] + split_ratio[1]
        test_frac = split_ratio[2]
        split_ratio = (train_frac, test_frac)
        logger.info(
            f"Merging validation into training: using (train={train_frac:.3f}, test={test_frac:.3f})"
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
        have = groups[cid]
        if not all(m in have for m in modalities) or "seg" not in have:
            skipped_train.append((cid, sorted(have.keys())))
            continue
        for ch, m in enumerate(modalities):
            _save_image_to(have[m], imgTr / f"{cid}_{ch:04d}.nii.gz", do_n4)
        seg_nii = nib.load(have["seg"])
        seg = remap_labels(seg_nii.get_fdata()).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(seg, seg_nii.affine, seg_nii.header),
            labTr / f"{cid}.nii.gz",
        )
        kept_train.append(cid)

    # ---- write test cases (require all modalities; no labels) ----
    kept_test, skipped_test = [], []
    for cid in tqdm(test_ids, desc="Writing test cases", unit="case"):
        have = groups[cid]
        if not all(m in have for m in modalities):
            skipped_test.append((cid, sorted(have.keys())))
            continue
        for ch, m in enumerate(modalities):
            _save_image_to(have[m], imgTs / f"{cid}_{ch:04d}.nii.gz", do_n4)
        kept_test.append(cid)

    # ---- dataset.json ----
    modality_map = {str(i): "MRI" for i, _ in enumerate(modalities)}
    ds = {
        "name": dataset_name,
        "description": f"BraTS-like with {len(modalities)} modalities {list(modalities)}",
        "reference": "Local",
        "licence": "Research",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": modality_map,
        "labels": {str(k): v for k, v in label_names.items()},
        "numTraining": len(kept_train),
        "numTest": len(kept_test),
        "training": [
            {"image": f"./imagesTr/{cid}.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
            for cid in kept_train
        ],
        "test": [f"./imagesTs/{cid}.nii.gz" for cid in kept_test],
    }
    (out_dir / "dataset.json").write_text(json.dumps(ds, indent=2) + "\n")

    # ---- summary.txt ----
    summary_lines = [
        f"Dataset: Dataset{dataset_id}_{dataset_name}",
        f"Modalities: {list(modalities)}",
        f"N4: {do_n4} (shrink={n4_shrink}, iters={n4_iters})",
        f"Requested split: train={split_ratio[0]:.3f}, test={split_ratio[1]:.3f}",
        f"Found cases: {n_cases}",
        f"Kept: train={len(kept_train)}, test={len(kept_test)}",
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
    summary_lines.extend(sorted(kept_train))
    summary_lines.append("\n-- TEST CASE IDS --")
    summary_lines.extend(sorted(kept_test))
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    logger.info(
        f"✅ Wrote {len(kept_train)} train and {len(kept_test)} test cases to {out_dir}"
    )
    if skipped_train or skipped_test:
        logger.warning(
            f"⚠️ Skipped {len(skipped_train)} train and {len(skipped_test)} test cases due to missing modalities/labels."
        )
    if unknown:
        logger.warning(
            f"⚠️ Ignored {len(unknown)} files with unrecognized names. Example: {unknown[0]}"
        )
    logger.info(f"summary.txt and dataset.json written in: {out_dir}")
    return out_dir


if __name__ == "__main__":
    convert_braTS_to_nnUNet(
        Path("data/BraTS2023-Glioma"),
        Path("data/nnUNet_raw"),
        dataset_id=501,
        dataset_name="BraTS3M",
        modalities=("t2f", "t1c", "t2w"),
        split_ratio=(0.8, 0.2),
        seed=42,
        do_n4=False,
    )
