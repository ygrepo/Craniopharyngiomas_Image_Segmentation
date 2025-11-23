import argparse
import ants
from pathlib import Path
from typing import List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(description="Register chiasm mask to patient space.")
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
    return args


def process_split(
    split_name: str,
    split_dir: Path,
    mni_t1: ants.ANTsImage,
    chiasm_mni: ants.ANTsImage,
    outdir: Path,
) -> List[Tuple[str, str, str, str]]:
    """
    Process a given split (imagesTr or imagesTs).

    Parameters
    ----------
    split_name : str
        Name of the split, e.g. "imagesTr" or "imagesTs".
    split_dir : Path
        Directory containing the *_0001.nii.gz files.
    mni_t1 : ants.ANTsImage
        Atlas T1 in MNI space.
    chiasm_mni : ants.ANTsImage
        Chiasm mask in MNI space.
    outdir : Path
        Base output directory for chiasm masks.

    Returns
    -------
    records : list of (split_name, case_id, status, message)
        Status per case.
    """
    records: List[Tuple[str, str, str, str]] = []

    split_outdir = outdir / split_name
    split_outdir.mkdir(parents=True, exist_ok=True)

    # all T1CE files: channel 1
    t1_files = sorted(split_dir.glob("*_0001.nii.gz"))
    logger.info(f"[{split_name}] Found {len(t1_files)} T1CE files.")

    for t1_path in t1_files:
        case_id = t1_path.name.replace("_0001.nii.gz", "")
        out_file = split_outdir / f"{case_id}_chiasm_mask.nii.gz"

        if out_file.exists():
            msg = f"[SKIP]: {case_id}, file exists: {out_file}"
            logger.warning(msg)
            records.append((split_name, case_id, "skipped_exists", msg))
            continue

        logger.info(f"[RUN {split_name}]: {case_id}")
        try:
            patient_t1 = ants.image_read(str(t1_path)).reorient_image2("RAI")
            logger.info(f"Patient T1 shape: {patient_t1.shape}")

            # --- Registration: atlas -> patient ---
            reg = ants.registration(
                fixed=patient_t1,
                moving=mni_t1,
                type_of_transform="SyN",
                verbose=False,
            )

            # --- Warp chiasm mask into patient space ---
            chiasm_patient = ants.apply_transforms(
                fixed=patient_t1,
                moving=chiasm_mni,
                transformlist=reg["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            logger.info(f"Chiasm mask patient shape: {chiasm_patient.shape}")

            ants.image_write(chiasm_patient, str(out_file))
            logger.info(f"[OK {split_name}] saved {out_file}")
            records.append((split_name, case_id, "ok", ""))

        except Exception as e:
            msg = f"[FAIL {split_name}]: {case_id}: {e}"
            logger.exception(msg)
            records.append((split_name, case_id, "failed", str(e)))
            # continue to next case

    return records


def main():
    # --- Paths ---
    args = get_args()
    setup_logging(args.log_file.resolve() if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")
    logger.info(f"  REPO_ROOT: {REPO_ROOT}")
    try:
        base = Path("nnUNet_raw/Dataset503_CP")
        imagesTr = base / "imagesTr"
        imagesTs = base / "imagesTs"

        # IMPORTANT: use the 09a atlas and your manual chiasm mask
        mni_t1_path = Path("./data/CP/atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz")
        mni_chiasm_path = Path("./data/CP/atlas/chiasm.nii")

        outdir = base / "chiasm_masks"
        outdir.mkdir(exist_ok=True)

        # Load atlas once
        mni_t1 = ants.image_read(str(mni_t1_path)).reorient_image2("RAI")
        chiasm_mni = ants.image_read(str(mni_chiasm_path)).reorient_image2("RAI")
        logger.info(f"Atlas T1 shape: {mni_t1.shape}")
        logger.info(f"Chiasm mask shape: {chiasm_mni.shape}")

        all_records: List[Tuple[str, str, str, str]] = []
        all_records.extend(
            process_split("imagesTr", imagesTr, mni_t1, chiasm_mni, outdir)
        )
        all_records.extend(
            process_split("imagesTs", imagesTs, mni_t1, chiasm_mni, outdir)
        )

        # --- Write summary TSV with registration status ---
        summary_path = outdir / "registration_summary.tsv"
        with summary_path.open("w") as f:
            f.write("split\tcase_id\tstatus\tmessage\n")
            for split_name, case_id, status, msg in all_records:
                # sanitize message for single-line TSV
                msg_clean = msg.replace("\n", " | ").replace("\t", " ")
                f.write(f"{split_name}\t{case_id}\t{status}\t{msg_clean}\n")

        logger.info(f"[OK] Done. Summary written to {summary_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


if __name__ == "__main__":
    main()
