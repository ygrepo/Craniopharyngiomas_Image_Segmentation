import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
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
    ap.add_argument(
        "--base_dir",
        type=Path,
        default=Path("nnUNet_raw/Dataset503_CP"),
        help="Base directory containing the data.",
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV file path. If not provided, will be saved in base_dir.",
    )
    return ap.parse_args()


def compute_chiasm_metrics(tumor_path, chiasm_path):
    """
    Compute tumor–chiasm spatial metrics:
        - Minimum Euclidean distance (mm)
        - Contact (overlapping voxels)
        - Overlap volume (mm³)
    """
    try:
        tumor_img = nib.load(str(tumor_path))
        chiasm_img = nib.load(str(chiasm_path))

        tumor = tumor_img.get_fdata() > 0
        chiasm = chiasm_img.get_fdata() > 0

        # Sanity checks
        if tumor.shape != chiasm.shape:
            raise ValueError(
                f"Shape mismatch: tumor {tumor.shape}, chiasm {chiasm.shape}"
            )
        if not np.allclose(tumor_img.affine, chiasm_img.affine):
            raise ValueError("Affine mismatch between tumor and chiasm images.")

        # Voxel volume in mm³
        zooms = tumor_img.header.get_zooms()
        voxel_volume_mm3 = zooms[0] * zooms[1] * zooms[2]

        # --- Overlap detection ---
        overlap_mask = np.logical_and(tumor, chiasm)
        overlap_voxels = int(overlap_mask.sum())
        overlap_volume_mm3 = overlap_voxels * voxel_volume_mm3

        # Contact = overlap present
        contact = overlap_voxels > 0

        # If contact → min distance is 0
        if contact:
            return 0.0, True, overlap_volume_mm3

        # --- Distance computation ---
        tumor_vox = np.argwhere(tumor)
        chiasm_vox = np.argwhere(chiasm)

        if tumor_vox.size == 0:
            raise ValueError(f"Tumor mask empty: {tumor_path}")
        if chiasm_vox.size == 0:
            raise ValueError(f"Chiasm mask empty: {chiasm_path}")

        affine = tumor_img.affine

        # Convert voxel coordinates → mm
        tumor_xyz = nib.affines.apply_affine(affine, tumor_vox)
        chiasm_xyz = nib.affines.apply_affine(affine, chiasm_vox)

        # KD-tree nearest neighbor search
        tree = cKDTree(tumor_xyz)
        dist_vals, _ = tree.query(chiasm_xyz, k=1)
        min_dist_mm = float(dist_vals.min())

        return min_dist_mm, False, overlap_volume_mm3

    except Exception as e:
        logger.error(f"Error computing metrics for {tumor_path} and {chiasm_path}: {e}")
        raise


def process_case(case_id, tumor_path, chiasm_path):
    """Process a single case and return results as a dictionary."""
    logger.info(f"Processing case: {case_id}")

    try:
        min_dist_mm, contact, overlap_volume_mm3 = compute_chiasm_metrics(
            tumor_path, chiasm_path
        )

        logger.info(
            f"Case {case_id}: dist = {min_dist_mm:.2f} mm | "
            f"contact = {contact} | overlap = {overlap_volume_mm3:.2f} mm³"
        )

        return {
            "case_id": case_id,
            "min_distance_mm": min_dist_mm,
            "contact": contact,
            "overlap_volume_mm3": overlap_volume_mm3,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error processing case {case_id}: {e}")
        return {
            "case_id": case_id,
            "min_distance_mm": np.nan,
            "contact": False,
            "overlap_volume_mm3": np.nan,
            "status": f"error: {str(e)}",
        }


def main():
    args = get_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Starting analysis with args: {args}")

    try:
        base_dir = Path(args.base_dir)
        chiasm_dir = base_dir / "chiasm_masks"
        tumor_dir = base_dir / "labelsTr"  # or replace with nnUNet predictions

        # Set output CSV path
        if args.output_csv:
            out_csv = Path(args.output_csv)
        else:
            out_csv = base_dir / "tumor_chiasm_metrics.csv"

        # Ensure output directory exists
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Get all chiasm files
        chiasm_files = sorted(chiasm_dir.glob("*_chiasm_mask.nii.gz"))
        logger.info(f"Found {len(chiasm_files)} chiasm mask files")

        if not chiasm_files:
            logger.warning(f"No chiasm mask files found in {chiasm_dir}")
            return

        # Process all cases
        results = []
        for chiasm_path in chiasm_files:
            case_id = chiasm_path.name.replace("_chiasm_mask.nii.gz", "")
            tumor_path = tumor_dir / f"{case_id}.nii.gz"

            if not tumor_path.exists():
                logger.warning(f"No tumor mask found for {case_id}, skipping")
                results.append(
                    {
                        "case_id": case_id,
                        "min_distance_mm": np.nan,
                        "contact": False,
                        "overlap_volume_mm3": np.nan,
                        "status": "missing_tumor_mask",
                    }
                )
                continue

            result = process_case(case_id, tumor_path, chiasm_path)
            results.append(result)

        # Create DataFrame and save
        if results:
            df = pd.DataFrame(results)

            # Save with pandas to_csv
            df.to_csv(out_csv, index=False, float_format="%.3f")
            logger.info(f"Results saved to: {out_csv}")

            # Print summary statistics
            successful_cases = df[df["status"] == "success"]
            if len(successful_cases) > 0:
                logger.info(
                    f"\nSummary Statistics ({len(successful_cases)} successful cases):"
                )
                logger.info(
                    f"Min distance - Mean: {successful_cases['min_distance_mm'].mean():.2f} mm, "
                    f"Std: {successful_cases['min_distance_mm'].std():.2f} mm"
                )
                logger.info(
                    f"Contact cases: {successful_cases['contact'].sum()} "
                    f"({successful_cases['contact'].mean()*100:.1f}%)"
                )
                logger.info(
                    f"Mean overlap volume: {successful_cases['overlap_volume_mm3'].mean():.2f} mm³"
                )

            # Print error summary
            error_cases = df[df["status"] != "success"]
            if len(error_cases) > 0:
                logger.warning(f"\nFailed cases: {len(error_cases)}")
                for status in error_cases["status"].unique():
                    count = (error_cases["status"] == status).sum()
                    logger.warning(f"  {status}: {count} cases")
        else:
            logger.warning("No results to save.")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    main()
