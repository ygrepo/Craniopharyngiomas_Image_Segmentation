from pathlib import Path
import nibabel as nib
import numpy as np
import json
import re
import random
from collections import OrderedDict, defaultdict
import argparse
from tqdm import tqdm
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
    safe_load_seg_any,
    load_sitk,
    save_img_like_reference,
    save_nifti_image,
    save_nifti,
    assert_matches_ref,
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


def convert_braTS_to_nnUNet(
    src_root: Path,
    dst_root: Path,
    *,
    dataset_id: int = 501,
    dataset_name: str = "BraTS2017_4ch",
    modalities: tuple[str, ...] = ("flair", "t1", "t1ce", "t2"),
    split_ratio: tuple = (
        0.8,
        0.2,
    ),  # (train, test) or (train, val, test -> val merged into train)
    seed: int = 42,
    do_n4: bool = False,
    n4_shrink: int = 2,
    n4_iters: int = 50,
    on_error: str = "skip_case",  # "skip_case" | "skip_modality" | "raise"
) -> Path:
    """
    Convert BraTS2017-style files to nnU-Net v2 raw structure.

    Input under src_root (any nesting):
      <case>_flair.nii(.gz)
      <case>_t1.nii(.gz)
      <case>_t1ce.nii(.gz)
      <case>_t2.nii(.gz)
      <case>_seg.nii(.gz)

    Output under nnUNet_raw/Dataset{dataset_id}_{dataset_name}/:
      imagesTr/, labelsTr/, imagesTs/ (optional), dataset.json, summary.txt, skipped_cases.json
    """
    logger.info("Starting BraTS→nnU-Net conversion")

    assert on_error in ("skip_case", "skip_modality", "raise")
    logger.info(f"dataset_id={dataset_id}, dataset_name={dataset_name}")
    logger.info(f"modalities={modalities}")
    logger.info(f"split_ratio={split_ratio}")
    logger.info(f"do_n4={do_n4}, n4_shrink={n4_shrink}, n4_iters={n4_iters}")

    # ---- validate split ----
    if len(split_ratio) not in (2, 3):
        raise ValueError("split_ratio must be (train, test) or (train, val, test).")
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")
    if len(split_ratio) == 3:
        train_frac = split_ratio[0] + split_ratio[1]
        test_frac = split_ratio[2]
        split_ratio = (train_frac, test_frac)
        logger.info(
            f"Merging validation into training: train={train_frac:.3f}, test={test_frac:.3f}"
        )

    # ---- scan & group files by case_id ----
    all_files = list(src_root.rglob("*.nii")) + list(src_root.rglob("*.nii.gz"))
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    unknown: list[Path] = []
    for p in all_files:
        cid, tag = _case_and_modality(p)  # expects <case>_(flair|t1|t1ce|t2|seg)
        if tag is None:
            unknown.append(p)
            continue
        groups[cid][tag.lower()] = p

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

    # ---- containers for bookkeeping ----
    kept_train: list[str] = []
    kept_test: list[str] = []
    skipped_train: list[tuple[str, str, list[str]]] = []
    skipped_test: list[tuple[str, str, list[str]]] = []
    broken_train: list[tuple[str, str]] = []
    broken_test: list[tuple[str, str]] = []

    # ---- write training cases (require all modalities + seg) ----
    for cid in tqdm(train_ids, desc="Writing training cases", unit="case"):
        have = groups[cid]
        missing = [m for m in modalities if m not in have] + (
            ["seg"] if "seg" not in have else []
        )
        if missing:
            logger.warning(f"Skipping {cid} (train): missing {missing}")
            skipped_train.append((cid, f"missing={missing}", sorted(have.keys())))
            continue

        case_failed = False
        # have: dict modality->Path for this cid; choose reference modality
        ref_mod = "flair" if "flair" in have else next(iter(have.keys() - {"seg"}))
        ref_sitk = load_sitk(have[ref_mod])
        # images in fixed channel order
        for ch, m in enumerate(modalities):
            try:
                save_img_like_reference(
                    ref_sitk,
                    have[m],
                    imgTr / f"{cid}_{ch:04d}.nii.gz",
                    is_label=False,
                    do_n4=do_n4,
                    n4_shrink=n4_shrink,
                    n4_iters=n4_iters,
                )
                assert_matches_ref(cid, m, ref_sitk, imgTr / f"{cid}_{ch:04d}.nii.gz")
            except Exception as e:
                msg = f"{cid}:{m} -> {type(e).__name__}: {e}"
                if on_error == "skip_modality":
                    logger.warning(f"Skipping modality {msg}")
                    case_failed = (
                        True  # nnU-Net needs all channels, so mark case failed
                    )
                    break
                elif on_error == "skip_case":
                    logger.error(f"Skipping case due to error {msg}")
                    case_failed = True
                    break
                else:
                    raise

        if case_failed:
            broken_train.append((cid, "image_write_error"))
            # clean partial outputs
            for ch in range(len(modalities)):
                p = imgTr / f"{cid}_{ch:04d}.nii.gz"
                if p.exists():
                    logger.info(f"Deleting partial output: {p}")
                    p.unlink(missing_ok=True)
            (labTr / f"{cid}.nii.gz").unlink(missing_ok=True)
            continue

        # labels
        try:
            save_img_like_reference(
                ref_sitk,
                have["seg"],
                labTr / f"{cid}.nii.gz",
                is_label=True,
                do_n4=False,
            )
            assert_matches_ref(cid, "seg", ref_sitk, labTr / f"{cid}.nii.gz")
            seg_nii = safe_load_seg_any(have["seg"], out_dtype=np.uint8)
            seg = seg_nii.get_fdata().astype(np.uint8, copy=False)
            seg = remap_labels_to_0123(seg)
            out_lbl = nib.Nifti1Image(seg, seg_nii.affine, seg_nii.header)
            out_lbl.set_data_dtype(np.uint8)
            save_nifti(out_lbl, labTr / f"{cid}.nii.gz")
        except Exception as e:
            msg = f"{cid}:seg -> {type(e).__name__}: {e}"
            if on_error == "skip_case":
                logger.error(f"Skipping case due to label error {msg}")
                for ch in range(len(modalities)):
                    (imgTr / f"{cid}_{ch:04d}.nii.gz").unlink(missing_ok=True)
                broken_train.append((cid, "label_write_error"))
                continue
            else:
                raise

        kept_train.append(cid)

    # ---- write test cases (require all modalities; no labels) ----
    for cid in tqdm(test_ids, desc="Writing test cases", unit="case"):
        have = groups[cid]
        missing = [m for m in modalities if m not in have]
        if missing:
            logger.warning(f"Skipping {cid} (test): missing {missing}")
            skipped_test.append((cid, f"missing={missing}", sorted(have.keys())))
            continue
        try:
            for ch, m in enumerate(modalities):
                save_nifti_image(
                    have[m],
                    imgTs / f"{cid}_{ch:04d}.nii.gz",
                    do_n4,
                    n4_shrink,
                    n4_iters,
                )
            kept_test.append(cid)
        except Exception as e:
            logger.error(f"Skipping test case {cid}: {e}")
            for ch in range(len(modalities)):
                (imgTs / f"{cid}_{ch:04d}.nii.gz").unlink(missing_ok=True)
            broken_test.append((cid, "image_write_error"))
            continue
    logger.info(f"Kept {len(kept_test)} test cases")

    # ---- dataset.json (nnU-Net v2 friendly) ----
    kept_train_sorted = sorted(kept_train)
    kept_test_sorted = sorted(kept_test)
    # Generate channel_names from the actual modalities argument (order-locked)
    channel_names = OrderedDict((str(i), m.upper()) for i, m in enumerate(modalities))
    logger.info(f"Channel names: {channel_names}")
    modality_map = OrderedDict((str(i), "MRI") for i in range(len(modalities)))
    logger.info(f"Modality map: {modality_map}")
    # channel_names = {
    #     "0": "FLAIR",
    #     "1": "T1",
    #     "2": "T1CE",
    #     "3": "T2",
    # }
    labels = OrderedDict(
        [
            ("background", 0),
            ("tumor", [1, 3]),
            ("edema", 2),
        ]
    )
    logger.info(f"Labels: {labels}")
    regions_class_order = [1, 2]
    ds = OrderedDict(
        [
            ("name", dataset_name),
            ("description", f"BraTS2017; channels={list(modalities)}"),
            ("reference", "Local"),
            ("licence", "Research"),
            ("release", "1.0"),
            ("tensorImageSize", "3D"),
            ("file_ending", ".nii.gz"),
            ("channel_names", channel_names),
            ("modality", modality_map),
            ("labels", labels),
            ("regions_class_order", regions_class_order),
            ("numTraining", len(kept_train_sorted)),
            ("numTest", len(kept_test_sorted)),
            (
                "training",
                [
                    {
                        "image": f"./imagesTr/{cid}_0000.nii.gz",
                        "label": f"./labelsTr/{cid}.nii.gz",
                    }
                    for cid in kept_train_sorted
                ],
            ),
            ("test", [f"./imagesTs/{cid}_0000.nii.gz" for cid in kept_test_sorted]),
        ]
    )
    (out_dir / "dataset.json").write_text(
        json.dumps(ds, indent=2, sort_keys=False) + "\n"
    )

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
            f"Skipped train (incomplete): {len(skipped_train)}  e.g. {skipped_train[0]}"
        )
    if skipped_test:
        summary_lines.append(
            f"Skipped test  (incomplete): {len(skipped_test)}  e.g. {skipped_test[0]}"
        )
    if broken_train:
        summary_lines.append(f"Broken train (IO/errors): {len(broken_train)}")
    if broken_test:
        summary_lines.append(f"Broken test  (IO/errors): {len(broken_test)}")
    if unknown:
        summary_lines.append(
            f"Unknown-named files ignored: {len(unknown)} (see skipped_cases.json)"
        )

    summary_lines.append("\n-- TRAIN CASE IDS --")
    summary_lines.extend(kept_train_sorted)
    summary_lines.append("\n-- TEST CASE IDS --")
    summary_lines.extend(kept_test_sorted)
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    # ---- persist skipped/broken/unknown details ----
    skipped_info = {
        "skipped_train": [
            {"case": cid, "reason": reason, "have": have}
            for (cid, reason, have) in skipped_train
        ],
        "broken_train": [
            {"case": cid, "reason": reason} for (cid, reason) in broken_train
        ],
        "skipped_test": [
            {"case": cid, "reason": reason, "have": have}
            for (cid, reason, have) in skipped_test
        ],
        "broken_test": [
            {"case": cid, "reason": reason} for (cid, reason) in broken_test
        ],
        "unknown_files": [str(p) for p in unknown],
    }
    (out_dir / "skipped_cases.json").write_text(
        json.dumps(skipped_info, indent=2) + "\n"
    )

    logger.info(
        f"Wrote {len(kept_train_sorted)} train and {len(kept_test_sorted)} test cases to {out_dir}"
    )
    logger.info("summary.txt and skipped_cases.json written.")
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
        type=int,
        default=502,
        help="Dataset ID of the installed BraTS-2017 model (logger.infoed by installer; often 002).",
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
        dataset_id=args.dataset_id,
        dataset_name="BraTS2017_4ch",
        modalities=("flair", "t1", "t1ce", "t2"),
        split_ratio=(0.8, 0.2),
        seed=42,
        do_n4=False,
    )


if __name__ == "__main__":
    main()
