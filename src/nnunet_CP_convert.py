from __future__ import annotations
from pathlib import Path
import nibabel as nib
import numpy as np
import json
import re
import random
from collections import OrderedDict, defaultdict
import argparse
from tqdm import tqdm
from typing import Optional, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
    safe_load_nifti,
    save_nifti_image,
)

logger = get_logger(__name__)


_EXT_RE = re.compile(r"(\.nii(\.gz)?|\.nrrd|\.mha|\.mhd)$", re.IGNORECASE)


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


def _case_and_modality(path: Path) -> Tuple[str, Optional[str]]:
    """
    Parse BraTS-like names with extra tokens, e.g.:
      89425108_T1_CE_3D_AX_ALIGNED.nrrd     -> ('89425108', 't1ce')
      89425108_T2_AX_ALIGNED.nrrd           -> ('89425108', 't2')
      89425108_T2_FLAIR_AX_ALIGNED.nrrd     -> ('89425108', 'flair')
      89425108_Tumor.seg.nrrd               -> ('89425108', 'seg')
    Returns (case_id, tag or None) where tag ∈ {'t1','t1ce','t2','flair','seg'}.
    """
    base = _EXT_RE.sub("", path.name)  # drop extension
    base = re.sub(
        r"\.seg$", "", base, flags=re.I
    )  # drop trailing ".seg" token if present

    if "_" not in base:
        return (base, None)

    case_id, rest = base.split("_", 1)
    s = rest.lower()

    # segmentation first (e.g., Tumor / seg)
    if re.search(r"(?:^|_)(tumou?r|seg)(?:_|$)", s):
        return (case_id, "seg")

    # flair (avoid misclassifying 't2_flair' as t2)
    if "flair" in s:
        return (case_id, "flair")

    # t1ce: t1ce / t1_ce / t1c / t1gd / t1_post / t1-contrast
    if re.search(r"\bt1[_\- ]?(ce|c|gd|post|contrast)\b", s):
        return (case_id, "t1ce")

    # plain t2 (but not t2_flair which was handled above)
    if re.search(r"(?:^|_)t2(?:_|$)", s):
        return (case_id, "t2")

    # plain t1
    if re.search(r"(?:^|_)t1(?:_|$)", s):
        return (case_id, "t1")

    return (case_id, None)


def convert_to_nnUNet(
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
    Convert CP-style files to nnU-Net v2 raw structure.

    Input under src_root (any nesting):
      <case>_flair.nii(.gz)
      <case>_t1.nii(.gz)
      <case>_t1ce.nii(.gz)
      <case>_t2.nii(.gz)
      <case>_seg.nii(.gz)

    Output under nnUNet_raw/Dataset{dataset_id}_{dataset_name}/:
      imagesTr/, labelsTr/, imagesTs/ (optional), dataset.json, summary.txt, skipped_cases.json
    """
    logger.info("Starting CP→nnU-Net conversion")

    assert on_error in ("skip_case", "skip_modality", "raise")

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
    all_files = list(src_root.rglob("*.nrrd"))
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    unknown: list[Path] = []
    for p in all_files:
        logger.info(f"Processing {p}")
        cid, tag = _case_and_modality(p)  # expects <case>_(flair|t1|t1ce|t2|seg)
        if tag is None:
            unknown.append(p)
            continue
        logger.info(f"Found {cid}:{tag} -> {p}")
        groups[cid][tag.lower()] = p

    case_ids = sorted(groups.keys())
    logger.info(f"Found {len(case_ids)} cases.")
    random.seed(seed)
    random.shuffle(case_ids)

    # # ---- compute train/test split ----
    # n_cases = len(case_ids)
    # n_train = int(n_cases * split_ratio[0])
    # train_ids = case_ids[:n_train]
    # test_ids = case_ids[n_train:]

    # # ---- prepare nnU-Net dirs ----
    # out_dir = dst_root / f"Dataset{dataset_id}_{dataset_name}"
    # imgTr = out_dir / "imagesTr"
    # labTr = out_dir / "labelsTr"
    # imgTs = out_dir / "imagesTs"
    # for d in (imgTr, labTr, imgTs):
    #     d.mkdir(parents=True, exist_ok=True)

    # # ---- containers for bookkeeping ----
    # kept_train: list[str] = []
    # kept_test: list[str] = []
    # skipped_train: list[tuple[str, str, list[str]]] = []
    # skipped_test: list[tuple[str, str, list[str]]] = []
    # broken_train: list[tuple[str, str]] = []
    # broken_test: list[tuple[str, str]] = []

    # # ---- write training cases (require all modalities + seg) ----
    # for cid in tqdm(train_ids, desc="Writing training cases", unit="case"):
    #     have = groups[cid]
    #     missing = [m for m in modalities if m not in have] + (
    #         ["seg"] if "seg" not in have else []
    #     )
    #     if missing:
    #         logger.warning(f"Skipping {cid} (train): missing {missing}")
    #         skipped_train.append((cid, f"missing={missing}", sorted(have.keys())))
    #         continue

    #     case_failed = False
    #     # images in fixed channel order
    #     for ch, m in enumerate(modalities):
    #         try:
    #             save_nifti_image(
    #                 have[m],
    #                 imgTr / f"{cid}_{ch:04d}.nii.gz",
    #                 do_n4,
    #                 n4_shrink,
    #                 n4_iters,
    #             )
    #         except Exception as e:
    #             msg = f"{cid}:{m} -> {type(e).__name__}: {e}"
    #             if on_error == "skip_modality":
    #                 logger.warning(f"Skipping modality {msg}")
    #                 case_failed = (
    #                     True  # nnU-Net needs all channels, so mark case failed
    #                 )
    #                 break
    #             elif on_error == "skip_case":
    #                 logger.error(f"Skipping case due to error {msg}")
    #                 case_failed = True
    #                 break
    #             else:
    #                 raise

    #     if case_failed:
    #         broken_train.append((cid, "image_write_error"))
    #         # clean partial outputs
    #         for ch in range(len(modalities)):
    #             p = imgTr / f"{cid}_{ch:04d}.nii.gz"
    #             if p.exists():
    #                 logger.info(f"Deleting partial output: {p}")
    #                 p.unlink(missing_ok=True)
    #         (labTr / f"{cid}.nii.gz").unlink(missing_ok=True)
    #         continue

    #     # labels
    #     try:
    #         seg_nii = safe_load_nifti(have["seg"], dtype=np.float32)
    #         seg = remap_labels_to_0123(seg_nii.get_fdata()).astype(np.uint8, copy=False)
    #         save_nifti_image(
    #             have["seg"],
    #             labTr / f"{cid}.nii.gz",
    #             do_n4,
    #             n4_shrink,
    #             n4_iters,
    #         )
    #         out_lbl = nib.Nifti1Image(seg, seg_nii.affine, seg_nii.header)
    #         out_lbl.set_data_dtype(np.uint8)
    #         nib.save(out_lbl, str(labTr / f"{cid}.nii.gz"))
    #     except Exception as e:
    #         msg = f"{cid}:seg -> {type(e).__name__}: {e}"
    #         if on_error == "skip_case":
    #             logger.error(f"Skipping case due to label error {msg}")
    #             for ch in range(len(modalities)):
    #                 (imgTr / f"{cid}_{ch:04d}.nii.gz").unlink(missing_ok=True)
    #             broken_train.append((cid, "label_write_error"))
    #             continue
    #         else:
    #             raise

    #     kept_train.append(cid)

    # # ---- write test cases (require all modalities; no labels) ----
    # for cid in tqdm(test_ids, desc="Writing test cases", unit="case"):
    #     have = groups[cid]
    #     missing = [m for m in modalities if m not in have]
    #     if missing:
    #         logger.warning(f"Skipping {cid} (test): missing {missing}")
    #         skipped_test.append((cid, f"missing={missing}", sorted(have.keys())))
    #         continue
    #     try:
    #         for ch, m in enumerate(modalities):
    #             save_nifti_image(
    #                 have[m],
    #                 imgTs / f"{cid}_{ch:04d}.nii.gz",
    #                 do_n4,
    #                 n4_shrink,
    #                 n4_iters,
    #             )
    #         kept_test.append(cid)
    #     except Exception as e:
    #         logger.error(f"Skipping test case {cid}: {e}")
    #         for ch in range(len(modalities)):
    #             (imgTs / f"{cid}_{ch:04d}.nii.gz").unlink(missing_ok=True)
    #         broken_test.append((cid, "image_write_error"))
    #         continue
    # logger.info(f"Kept {len(kept_test)} test cases")

    # # ---- dataset.json (nnU-Net v2 friendly) ----
    # kept_train_sorted = sorted(kept_train)
    # kept_test_sorted = sorted(kept_test)
    # # Generate channel_names from the actual modalities argument (order-locked)
    # channel_names = OrderedDict((str(i), m.upper()) for i, m in enumerate(modalities))
    # logger.info(f"Channel names: {channel_names}")
    # modality_map = OrderedDict((str(i), "MRI") for i in range(len(modalities)))
    # logger.info(f"Modality map: {modality_map}")
    # # channel_names = {
    # #     "0": "FLAIR",
    # #     "1": "T1",
    # #     "2": "T1CE",
    # #     "3": "T2",
    # # }
    # labels = OrderedDict(
    #     [
    #         ("background", 0),
    #         ("whole_tumor", [1, 2, 3]),
    #         ("tumor_core", [2, 3]),
    #         ("enhancing_tumor", 3),
    #     ]
    # )
    # logger.info(f"Labels: {labels}")
    # regions_class_order = [1, 2, 3]
    # ds = OrderedDict(
    #     [
    #         ("name", dataset_name),
    #         ("description", f"BraTS2017; channels={list(modalities)}"),
    #         ("reference", "Local"),
    #         ("licence", "Research"),
    #         ("release", "1.0"),
    #         ("tensorImageSize", "3D"),
    #         ("file_ending", ".nii.gz"),
    #         ("channel_names", channel_names),
    #         ("modality", modality_map),
    #         ("labels", labels),
    #         ("regions_class_order", regions_class_order),
    #         ("numTraining", len(kept_train_sorted)),
    #         ("numTest", len(kept_test_sorted)),
    #         (
    #             "training",
    #             [
    #                 {
    #                     "image": f"./imagesTr/{cid}_0000.nii.gz",
    #                     "label": f"./labelsTr/{cid}.nii.gz",
    #                 }
    #                 for cid in kept_train_sorted
    #             ],
    #         ),
    #         ("test", [f"./imagesTs/{cid}_0000.nii.gz" for cid in kept_test_sorted]),
    #     ]
    # )
    # (out_dir / "dataset.json").write_text(
    #     json.dumps(ds, indent=2, sort_keys=False) + "\n"
    # )

    # # ---- summary.txt ----
    # summary_lines = [
    #     f"Dataset: Dataset{dataset_id}_{dataset_name}",
    #     f"Modalities (order): {list(modalities)}  -> channels 0000..{len(modalities)-1:04d}",
    #     f"N4: {do_n4} (shrink={n4_shrink}, iters={n4_iters})",
    #     f"Requested split: train={split_ratio[0]:.3f}, test={split_ratio[1]:.3f}",
    #     f"Found cases: {n_cases}",
    #     f"Kept: train={len(kept_train_sorted)}, test={len(kept_test_sorted)}",
    # ]
    # if skipped_train:
    #     summary_lines.append(
    #         f"Skipped train (incomplete): {len(skipped_train)}  e.g. {skipped_train[0]}"
    #     )
    # if skipped_test:
    #     summary_lines.append(
    #         f"Skipped test  (incomplete): {len(skipped_test)}  e.g. {skipped_test[0]}"
    #     )
    # if broken_train:
    #     summary_lines.append(f"Broken train (IO/errors): {len(broken_train)}")
    # if broken_test:
    #     summary_lines.append(f"Broken test  (IO/errors): {len(broken_test)}")
    # if unknown:
    #     summary_lines.append(
    #         f"Unknown-named files ignored: {len(unknown)} (see skipped_cases.json)"
    #     )

    # summary_lines.append("\n-- TRAIN CASE IDS --")
    # summary_lines.extend(kept_train_sorted)
    # summary_lines.append("\n-- TEST CASE IDS --")
    # summary_lines.extend(kept_test_sorted)
    # (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    # # ---- persist skipped/broken/unknown details ----
    # skipped_info = {
    #     "skipped_train": [
    #         {"case": cid, "reason": reason, "have": have}
    #         for (cid, reason, have) in skipped_train
    #     ],
    #     "broken_train": [
    #         {"case": cid, "reason": reason} for (cid, reason) in broken_train
    #     ],
    #     "skipped_test": [
    #         {"case": cid, "reason": reason, "have": have}
    #         for (cid, reason, have) in skipped_test
    #     ],
    #     "broken_test": [
    #         {"case": cid, "reason": reason} for (cid, reason) in broken_test
    #     ],
    #     "unknown_files": [str(p) for p in unknown],
    # }
    # (out_dir / "skipped_cases.json").write_text(
    #     json.dumps(skipped_info, indent=2) + "\n"
    # )

    # logger.info(
    #     f"Wrote {len(kept_train_sorted)} train and {len(kept_test_sorted)} test cases to {out_dir}"
    # )
    # logger.info("summary.txt and skipped_cases.json written.")
    # return out_dir


def main():
    ap = argparse.ArgumentParser(
        description="Build nnU-Net v2 inputs and optionally run prediction."
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
        default=503,
        help="Dataset ID.",
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

    convert_to_nnUNet(
        src_root=src_root,
        dst_root=dst_root,
        dataset_id=args.dataset_id,
        dataset_name="CP",
        modalities=("flair", "t1", "t2"),
        split_ratio=(0.8, 0.2),
        seed=42,
        do_n4=False,
    )


if __name__ == "__main__":
    main()
