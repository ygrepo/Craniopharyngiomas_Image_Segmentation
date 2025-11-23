import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def get_args():
    ap = argparse.ArgumentParser(
        description="Compute spatial radiomics (Dist/Overlap) between Tumor and Chiasm."
    )
    ap.add_argument(
        "--base_dir",
        type=Path,
        default=Path("nnUNet_raw/Dataset503_CP"),
        help="Base directory containing labelsTr/ and chiasm_masks/imagesTr, imagesTs.",
    )
    ap.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory containing test tumor labels (predictions for imagesTs).",
    )
    ap.add_argument(
        "--output_csv", type=Path, default=None, help="Path to save results."
    )
    ap.add_argument(
        "--log_file", type=Path, default="radiomics.log", help="Log file path."
    )
    ap.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return ap.parse_args()


def get_surface_voxels(mask: np.ndarray) -> np.ndarray:
    """
    Extracts only the boundary/surface voxels of a binary mask.
    Optimization: Drastically reduces points for KDTree.
    """
    # Erode the mask by 1 pixel
    eroded = binary_erosion(mask)
    # The surface is where the mask exists but the eroded version does not
    surface = mask ^ eroded
    return np.argwhere(surface)


def compute_metrics(tumor_path: Path, chiasm_path: Path):
    """
    Computes spatial metrics.
    Returns: Min_Dist(mm), Hausdorff95(mm), Overlap(mm3), Contact(bool)
    """
    try:
        # Load Images
        t_img = nib.load(str(tumor_path))
        c_img = nib.load(str(chiasm_path))

        affine = t_img.affine
        zooms = t_img.header.get_zooms()
        voxel_vol = np.prod(zooms)

        # Get Data (Ensure Binary)
        tumor_arr = t_img.get_fdata() > 0
        chiasm_arr = c_img.get_fdata() > 0

        # 1. SANITY CHECK: Geometry
        if (
            not np.allclose(affine, c_img.affine, atol=1e-3)
            or tumor_arr.shape != chiasm_arr.shape
        ):
            logger.error(f"Geometry mismatch for {tumor_path.name}")
            return None

        # 2. OVERLAP / CONTACT
        overlap_mask = tumor_arr & chiasm_arr
        overlap_vox = np.sum(overlap_mask)
        overlap_mm3 = overlap_vox * voxel_vol
        contact = overlap_vox > 0

        if contact:
            # If touching, distances are 0
            return 0.0, 0.0, overlap_mm3, True

        # 3. DISTANCE (Optimized)
        # Only get surface voxels to speed up KDTree
        t_surf_idx = get_surface_voxels(tumor_arr)
        c_surf_idx = get_surface_voxels(chiasm_arr)

        if len(t_surf_idx) == 0 or len(c_surf_idx) == 0:
            logger.warning(f"Empty mask found: {tumor_path.name}")
            return np.nan, np.nan, 0.0, False

        # Convert voxel indices to Millimeters
        t_mm = nib.affines.apply_affine(affine, t_surf_idx)
        c_mm = nib.affines.apply_affine(affine, c_surf_idx)

        # KDTree: Find nearest neighbor in Tumor for every Chiasm point
        tree = cKDTree(t_mm)
        dists, _ = tree.query(c_mm, k=1)

        min_dist = np.min(dists)
        hd95 = np.percentile(dists, 95)  # Hausdorff 95% (Robust to outliers)

        return min_dist, hd95, overlap_mm3, False

    except Exception as e:
        logger.error(f"Error processing {tumor_path.name}: {e}")
        return None


def process_split(
    split_name: str,
    tumor_dir: Path,
    chiasm_dir: Path,
) -> list[dict]:
    """
    Process a given split (train/test) and return a list of result dicts.
    """
    results: list[dict] = []

    if not tumor_dir.exists() or not chiasm_dir.exists():
        logger.warning(
            f"[{split_name}] Directories not found. Checked:\n"
            f"  Tumor:  {tumor_dir}\n"
            f"  Chiasm: {chiasm_dir}\n"
            f"Skipping this split."
        )
        return results

    tumor_files = sorted(list(tumor_dir.glob("*.nii.gz")))
    logger.info(f"[{split_name}] Found {len(tumor_files)} tumor files. Starting...")

    for t_path in tumor_files:
        case_id = t_path.name.replace(".nii.gz", "")

        c_path = chiasm_dir / f"{case_id}_chiasm_mask.nii.gz"
        if not c_path.exists():
            logger.warning(f"[{split_name}] Missing chiasm for {case_id}: {c_path}")
            continue

        metrics = compute_metrics(t_path, c_path)
        if metrics:
            min_d, hd95, vol, is_contact = metrics
            results.append(
                {
                    "Split": split_name,
                    "Case_ID": case_id,
                    "Min_Distance_mm": min_d,
                    "Hausdorff95_mm": hd95,
                    "Overlap_Volume_mm3": vol,
                    "Contact": int(is_contact),
                }
            )

    logger.info(f"[{split_name}] Completed. Valid cases: {len(results)}")
    return results


def main():
    args = get_args()
    setup_logging(args.log_file.resolve() if args.log_file else None, args.log_level)
    logger.info("Starting analysis")

    try:
        logger.info(f"Base dir: {args.base_dir}")
        logger.info(f"Pred dir: {args.pred_dir}")
        logger.info(f"Output CSV: {args.output_csv}")
        logger.info(f"Log file: {args.log_file}")

        # Training split directories
        tumor_dir_tr = args.base_dir / "labelsTr"
        chiasm_dir_tr = args.base_dir / "chiasm_masks" / "imagesTr"

        # Test split directories
        tumor_dir_ts = args.pred_dir
        chiasm_dir_ts = args.base_dir / "chiasm_masks" / "imagesTs"

        all_results: list[dict] = []

        # Process training
        all_results.extend(process_split("train", tumor_dir_tr, chiasm_dir_tr))

        # Process test
        all_results.extend(process_split("test", tumor_dir_ts, chiasm_dir_ts))

        # Save
        if all_results:
            df = pd.DataFrame(all_results)
            save_path = (
                args.output_csv
                if args.output_csv
                else args.base_dir / "radiomics_results.csv"
            )
            df.to_csv(save_path, index=False)
            logger.info(
                f"Success! Saved metrics for {len(df)} cases (both splits) to {save_path}"
            )

            # Quick Stats (overall)
            logger.info(
                f"Overall Mean Min Distance: {df['Min_Distance_mm'].mean():.2f} mm"
            )
            logger.info(f"Overall Contact Cases: {df['Contact'].sum()} / {len(df)}")

            # Optional: per-split stats
            for split_name, g in df.groupby("Split"):
                logger.info(
                    f"[{split_name}] n={len(g)}, "
                    f"Mean Min Distance={g['Min_Distance_mm'].mean():.2f} mm, "
                    f"Contact Cases={g['Contact'].sum()} / {len(g)}"
                )
        else:
            logger.warning("No results generated for any split.")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    main()
