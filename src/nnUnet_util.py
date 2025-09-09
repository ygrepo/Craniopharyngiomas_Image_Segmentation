import json
from pathlib import Path
import nibabel as nib
import numpy as np
import shutil
import SimpleITK as sitk


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Optional N4 bias correction
def n4_bias_correct_np(x: np.ndarray, shrink: int = 2, n_iters: int = 50) -> np.ndarray:

    img = sitk.GetImageFromArray(x.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)  # rough brain mask
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetShrinkFactor(shrink)
    corrector.SetMaximumNumberOfIterations([n_iters])
    corrected = corrector.Execute(img, mask)
    return sitk.GetArrayFromImage(corrected).astype(np.float32)


def remap_labels(arr: np.ndarray) -> np.ndarray:
    """Remap BraTS labels {0,1,2,4} → {0,1,2,3}."""
    arr = arr.astype(np.uint8)
    if 4 in np.unique(arr):
        arr[arr == 4] = 3
    return arr


def convert_braTS_to_nnUNet(
    src_root: Path,
    dst_root: Path,
    dataset_id: int = 501,
    dataset_name: str = "BraTS3M",
    do_n4: bool = False,
    train_split: float = 1.0,  # fraction of cases to put into training
    log_level: str = "INFO",
):
    """
    Convert BraTS-style folders to nnU-Net format with dataset.json.

    Args:
        src_root: Path with subfolders like BraTS-GLI-xxxx.
        dst_root: nnUNet_raw folder (e.g. Path(os.environ["nnUNet_raw"])).
        dataset_id: integer ID for the dataset.
        dataset_name: string name for dataset.json.
        do_n4: run N4 bias correction if True.
        train_split: fraction of cases to put in training (rest go to test).
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    out_dir = dst_root / f"Dataset{dataset_id}_{dataset_name}"
    imgTr = out_dir / "imagesTr"
    labTr = out_dir / "labelsTr"
    imgTs = out_dir / "imagesTs"
    for d in [imgTr, labTr, imgTs]:
        d.mkdir(parents=True, exist_ok=True)

    cases = sorted(src_root.glob("*/*"))  # each case dir
    train_cases = cases[: int(len(cases) * train_split)]
    test_cases = cases[int(len(cases) * train_split) :]

    training_entries = []
    test_entries = []

    for case_dir in train_cases:
        stem = case_dir.name
        flair = case_dir / f"{stem}-t2f.nii"
        t1ce = case_dir / f"{stem}-t1c.nii"
        t2w = case_dir / f"{stem}-t2w.nii"
        seg = case_dir / f"{stem}-seg.nii"

        if not (flair.exists() and t1ce.exists() and t2w.exists() and seg.exists()):
            logger.warning(f"⚠️ Skipping incomplete case {stem}")
            continue

        # load and (optionally) N4
        def load_and_maybe_n4(path):
            arr = nib.load(path).get_fdata().astype(np.float32)
            return n4_bias_correct_np(arr) if do_n4 else arr, nib.load(path)

        flair_data, flair_nii = load_and_maybe_n4(flair)
        t1ce_data, t1ce_nii = load_and_maybe_n4(t1ce)
        t2w_data, t2w_nii = load_and_maybe_n4(t2w)

        seg_nii = nib.load(seg)
        seg_data = remap_labels(seg_nii.get_fdata())

        # save to nnUNet format (_0000, _0001, _0002)
        nib.save(
            nib.Nifti1Image(flair_data, flair_nii.affine, flair_nii.header),
            imgTr / f"{stem}_0000.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(t1ce_data, t1ce_nii.affine, t1ce_nii.header),
            imgTr / f"{stem}_0001.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(t2w_data, t2w_nii.affine, t2w_nii.header),
            imgTr / f"{stem}_0002.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(seg_data.astype(np.uint8), seg_nii.affine, seg_nii.header),
            labTr / f"{stem}.nii.gz",
        )

        training_entries.append(
            {"image": f"./imagesTr/{stem}.nii.gz", "label": f"./labelsTr/{stem}.nii.gz"}
        )

    for case_dir in test_cases:
        stem = case_dir.name
        flair = case_dir / f"{stem}-t2f.nii"
        t1ce = case_dir / f"{stem}-t1c.nii"
        t2w = case_dir / f"{stem}-t2w.nii"
        if not (flair.exists() and t1ce.exists() and t2w.exists()):
            print(f"⚠️ Skipping incomplete test case {stem}")
            continue
        shutil.copy(flair, imgTs / f"{stem}_0000.nii.gz")
        shutil.copy(t1ce, imgTs / f"{stem}_0001.nii.gz")
        shutil.copy(t2w, imgTs / f"{stem}_0002.nii.gz")
        test_entries.append(f"./imagesTs/{stem}.nii.gz")

    # write dataset.json
    ds = {
        "name": dataset_name,
        "description": "BraTS-like glioma with 3 MRI modalities (FLAIR, T1CE, T2)",
        "reference": "Local",
        "licence": "Research",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {"0": "MRI", "1": "MRI", "2": "MRI"},
        "labels": {
            "0": "background",
            "1": "necrotic/non-enhancing",
            "2": "edema",
            "3": "enhancing",
        },
        "numTraining": len(training_entries),
        "numTest": len(test_entries),
        "training": training_entries,
        "test": test_entries,
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(ds, f, indent=2)

    logger.info(
        f"✅ Converted {len(training_entries)} training and {len(test_entries)} test cases to {out_dir}"
    )
    if do_n4:
        logger.info("✅ N4 bias correction applied")


if __name__ == "__main__":
    convert_braTS_to_nnUNet(
        Path("data/BraTS2023-Glioma"),
        Path("data/nnUNet_raw"),
        do_n4=True,
    )
