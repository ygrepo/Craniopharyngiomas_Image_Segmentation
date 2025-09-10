from pathlib import Path
import nibabel as nib
import numpy as np
import json
import shutil
import SimpleITK as sitk


def n4_bias_correct_np(x: np.ndarray, shrink: int = 2, n_iters: int = 50) -> np.ndarray:

    img = sitk.GetImageFromArray(x.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetShrinkFactor(shrink)
    n4.SetMaximumNumberOfIterations([n_iters])
    return sitk.GetArrayFromImage(n4.Execute(img, mask)).astype(np.float32)


def remap_labels(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.uint8)
    if 4 in np.unique(arr):
        arr[arr == 4] = 3  # {0,1,2,4} -> {0,1,2,3}
    return arr


def convert_braTS_to_nnUNet(
    src_root: Path,
    dst_root: Path,
    *,
    dataset_id: int = 501,
    dataset_name: str = "BraTS3M",
    do_n4: bool = False,
    train_split: float = 1.0,
    modalities: tuple[str, ...] = ("t2f", "t1c", "t2w"),  # default 3 channels
    label_names: dict[int, str] | None = None,
):
    """
    modalities: any subset of ("t2f","t1c","t2w","t1n"), order defines _000x.
      e.g., ("t2f","t1c","t2w","t1n") -> 4 channels
    """
    # map modality tag -> human name for dataset.json
    modality_human = {"t2f": "MRI", "t1c": "MRI", "t2w": "MRI", "t1n": "MRI"}
    if label_names is None:
        label_names = {
            0: "background",
            1: "necrotic/non-enhancing",
            2: "edema",
            3: "enhancing",
        }

    out_dir = dst_root / f"Dataset{dataset_id}_{dataset_name}"
    imgTr = out_dir / "imagesTr"
    labTr = out_dir / "labelsTr"
    imgTs = out_dir / "imagesTs"
    imgTr.mkdir(parents=True, exist_ok=True)
    labTr.mkdir(parents=True, exist_ok=True)
    imgTs.mkdir(parents=True, exist_ok=True)

    cases = sorted(src_root.glob("*/*"))
    n_train = int(len(cases) * train_split)
    train_cases, test_cases = cases[:n_train], cases[n_train:]

    def load_mod(path, run_n4: bool):
        nii = nib.load(path)
        arr = nii.get_fdata().astype(np.float32)
        if run_n4:
            arr = n4_bias_correct_np(arr)
        return arr, nii

    training_entries, test_entries = [], []

    for case_dir in train_cases:
        stem = case_dir.name
        # build per-modality file paths
        need = {m: case_dir / f"{stem}-{m}.nii" for m in modalities}
        seg = case_dir / f"{stem}-seg.nii"
        if not (all(p.exists() for p in need.values()) and seg.exists()):
            print(f"⚠️ Skipping incomplete case {stem}")
            continue

        # load modalities (optionally N4) and write _000x.nii.gz
        saved_any = False
        for ch, m in enumerate(modalities):
            data, nii = load_mod(need[m], do_n4)
            nib.save(
                nib.Nifti1Image(data, nii.affine, nii.header),
                imgTr / f"{stem}_{ch:04d}.nii.gz",
            )
            saved_any = True

        if not saved_any:
            continue

        # labels (remap if needed)
        seg_nii = nib.load(seg)
        lab = remap_labels(seg_nii.get_fdata())
        nib.save(
            nib.Nifti1Image(lab.astype(np.uint8), seg_nii.affine, seg_nii.header),
            labTr / f"{stem}.nii.gz",
        )

        training_entries.append(
            {"image": f"./imagesTr/{stem}.nii.gz", "label": f"./labelsTr/{stem}.nii.gz"}
        )

    for case_dir in test_cases:
        stem = case_dir.name
        need = {m: case_dir / f"{stem}-{m}.nii" for m in modalities}
        if not all(p.exists() for p in need.values()):
            print(f"⚠️ Skipping incomplete test case {stem}")
            continue
        for ch, m in enumerate(modalities):
            shutil.copy(need[m], imgTs / f"{stem}_{ch:04d}.nii.gz")
        test_entries.append(f"./imagesTs/{stem}.nii.gz")

    # dataset.json
    modality_map = {str(i): modality_human[m] for i, m in enumerate(modalities)}
    ds = {
        "name": dataset_name,
        "description": f"BraTS-like glioma with {len(modalities)} MRI modalities {list(modalities)}",
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

    print(
        f"✅ Wrote {len(training_entries)} train and {len(test_entries)} test cases to {out_dir}"
    )
    if do_n4:
        print("✅ N4 bias correction applied")


if __name__ == "__main__":
    convert_braTS_to_nnUNet(
        Path("data/BraTS2023-Glioma"),
        Path("data/nnUNet_raw"),
        do_n4=False,
        modalities=("t2f", "t1c", "t2w", "t1n"),  # 4-ch
        dataset_name="BraTS4M",
        dataset_id=502,
    )
    # convert_braTS_to_nnUNet(
    #     Path("data/BraTS2023-Glioma"),
    #     Path("data/nnUNet_raw"),
    #     do_n4=True,
    #     modalities=("t2f", "t1c", "t2w"),
    # )

# Example usage:
# convert_braTS_to_nnUNet(Path("data/BraTS2023-Glioma"), Path("data/nnUNet_raw"),
#                         do_n4=True, modalities=("t2f","t1c","t2w"))        # 3-ch
