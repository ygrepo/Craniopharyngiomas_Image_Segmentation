import argparse
import ants
from pathlib import Path

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
    split_dir: Path, mni_t1: ants.ANTsImage, chiasm_mni: ants.ANTsImage, outdir: Path
):
    # all T1CE files: channel 1
    t1_files = sorted(split_dir.glob("*_0001.nii.gz"))

    for t1_path in t1_files:
        case_id = t1_path.name.replace("_0001.nii.gz", "")
        out_file = outdir / f"{case_id}_chiasm_mask.nii.gz"

        if out_file.exists():
            logger.warning(f"[SKIP]:{case_id}")
            continue

        logger.info(f"[RUN]:{case_id}")
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
        logger.info(f"[ok] saved {out_file}")


def main():
    # --- Paths ---
    args = get_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
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

        process_split(imagesTr, mni_t1, chiasm_mni, outdir)
        process_split(imagesTs, mni_t1, chiasm_mni, outdir)
        logger.info("[OK] Done.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


if __name__ == "__main__":
    main()
