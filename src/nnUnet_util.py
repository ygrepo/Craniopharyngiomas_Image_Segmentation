from pathlib import Path
import nibabel as nib
import numpy as np
import json
import shutil
import re
from collections import defaultdict


# --- optional N4 ---
def n4_bias_correct_np(x: np.ndarray, shrink: int = 2, n_iters: int = 50) -> np.ndarray:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(x.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetShrinkFactor(shrink)
    n4.SetMaximumNumberOfIterations([n_iters])
    out = n4.Execute(img, mask)
    return sitk.GetArrayFromImage(out).astype(np.float32)


def remap_labels(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.uint8)
    if 4 in np.unique(arr):
        arr[arr == 4] = 3  # BraTS {0,1,2,4} -> {0,1,2,3}
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
    Parse a BraTS-like filename and return (case_id, modality_tag or 'seg').
    Accepts both .nii and .nii.gz.
    Expected suffixes: -t2f, -t1c, -t2w, -t1n, -seg
    """
    base = _strip_ext(path)
    # match trailing -<tag> where tag in {t2f,t1c,t2w,t1n,seg}
    m = re.match(r"^(.*)-(t2f|t1c|t2w|t1n|seg)$", base)
    if not m:
        return (base, None)  # unrecognized
    return (m.group(1), m.group(2))


def convert_braTS_to_nnUNet(
    src_root: Path,
    dst_root: Path,
    *,
    dataset_id: int = 501,
    dataset_name: str = "BraTS3M",
    do_n4: bool = False,
    train_split: float = 1.0,
    modalities: tuple[str, ...] = ("t2f", "t1c", "t2w"),  # or ("t2f","t1c","t2w","t1n")
    label_names: dict[int, str] | None = None,
):
    if label_names is None:
        label_names = {
            0: "background",
            1: "necrotic/non-enhancing",
            2: "edema",
            3: "enhancing",
        }

    # 1) scan all files (nested ok)
    all_files = list(src_root.rglob("*.nii")) + list(src_root.rglob("*.nii.gz"))
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    unknown = []
    for p in all_files:
        case_id, tag = _case_and_modality(p)
        if tag is None:
            unknown.append(p)
            continue
        groups[case_id][tag] = p

    # 2) split train/test by case_id (sorted for reproducibility)
    case_ids = sorted(groups.keys())
    n_train = int(len(case_ids) * train_split)
    train_ids, test_ids = case_ids[:n_train], case_ids[n_train:]

    # 3) prepare nnUNet dirs
    out_dir = dst_root / f"Dataset{dataset_id}_{dataset_name}"
    imgTr = out_dir / "imagesTr"
    labTr = out_dir / "labelsTr"
    imgTs = out_dir / "imagesTs"
    imgTr.mkdir(parents=True, exist_ok=True)
    labTr.mkdir(parents=True, exist_ok=True)
    imgTs.mkdir(parents=True, exist_ok=True)

    # 4) helpers
    def load_mod(path: Path, use_n4: bool):
        nii = nib.load(path)
        arr = nii.get_fdata().astype(np.float32)
        if use_n4:
            arr = n4_bias_correct_np(arr)
        return arr, nii

    # 5) convert training (need all modalities + seg)
    training_entries = []
    incomplete_train = []
    for cid in train_ids:
        have = groups[cid]
        if not all(m in have for m in modalities) or ("seg" not in have):
            incomplete_train.append((cid, sorted(have.keys())))
            continue

        # write images (_000x)
        for ch, m in enumerate(modalities):
            data, nii = load_mod(have[m], do_n4)
            nib.save(
                nib.Nifti1Image(data, nii.affine, nii.header),
                imgTr / f"{cid}_{ch:04d}.nii.gz",
            )

        # write label (remap if needed)
        seg_nii = nib.load(have["seg"])
        seg = remap_labels(seg_nii.get_fdata())
        nib.save(
            nib.Nifti1Image(seg.astype(np.uint8), seg_nii.affine, seg_nii.header),
            labTr / f"{cid}.nii.gz",
        )

        training_entries.append(
            {"image": f"./imagesTr/{cid}.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
        )

    # 6) convert test (need all modalities; no labels)
    test_entries = []
    incomplete_test = []
    for cid in test_ids:
        have = groups[cid]
        if not all(m in have for m in modalities):
            incomplete_test.append((cid, sorted(have.keys())))
            continue
        for ch, m in enumerate(modalities):
            shutil.copy(have[m], imgTs / f"{cid}_{ch:04d}.nii.gz")
        test_entries.append(f"./imagesTs/{cid}.nii.gz")

    # 7) dataset.json
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
        "numTraining": len(training_entries),
        "numTest": len(test_entries),
        "training": training_entries,
        "test": test_entries,
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(ds, f, indent=2)

    # 8) reporting
    print(
        f"✅ Wrote {len(training_entries)} train and {len(test_entries)} test cases to {out_dir}"
    )
    if do_n4:
        print("✅ N4 bias correction applied")
    if unknown:
        print(
            f"⚠️ {len(unknown)} files with unrecognized names (ignored). Example: {unknown[0]}"
        )
    if incomplete_train:
        print(
            f"⚠️ Incomplete TRAIN cases: {len(incomplete_train)} (missing modalities or seg). Example: {incomplete_train[0]}"
        )
    if incomplete_test:
        print(
            f"⚠️ Incomplete TEST cases: {len(incomplete_test)}. Example: {incomplete_test[0]}"
        )


if __name__ == "__main__":
    convert_braTS_to_nnUNet(
        Path("data/BraTS2023-Glioma"),
        Path("data/nnUNet_raw"),
        dataset_id=502,
        dataset_name="BraTS4M",
        modalities=("t2f", "t1c", "t2w", "t1n"),  # or drop t1n for 3-ch
        do_n4=False,  # True if you want N4 (slower)
        train_split=1.0,  # 100% train; set <1.0 to put rest into imagesTs
    )
