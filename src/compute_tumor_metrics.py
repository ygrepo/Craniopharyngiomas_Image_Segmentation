import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import fnmatch
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)


# ---------- I/O ----------
def read_vol(path: Path) -> sitk.Image:
    if not path or not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return sitk.ReadImage(str(path))


def first_match(case_dir: Path, suffix_glob: str) -> Optional[Path]:
    """
    Case-insensitive match for files like <CASE>_<suffix>.
    If the suffix has no '.', we treat it as a stem and allow any extension (adds a trailing '*').
    Examples:
      suffix_glob='t1*ce*aligned*.nii.gz'  -> matches exactly
      suffix_glob='T1_CE_3D_AX_ALIGNED'    -> matches any ext (e.g., .nii.gz)
    """
    files = [p for p in case_dir.iterdir() if p.is_file()]
    patt = suffix_glob.lower()
    # If no dot in pattern, allow any extension
    if "." not in patt:
        patt = patt + "*"
    # Expect filenames like '<CASE>_<suffix>'
    pattern = f"*_{patt}"
    hits = [p for p in files if fnmatch.fnmatch(p.name.lower(), pattern)]
    return sorted(hits)[0] if hits else None


def compute_feret_diameter(mask_array: np.ndarray, spacing: np.ndarray) -> float:
    coords = np.argwhere(mask_array > 0)  # voxel indices
    coords_mm = coords * spacing[::-1]  # convert to mm (careful with axis order)
    dmax = np.max(
        np.linalg.norm(coords_mm[:, None, :] - coords_mm[None, :, :], axis=-1)
    )
    return dmax


def label_stats(mask_u8: sitk.Image, label: int = 1) -> Dict[str, Any]:
    """Compute shape stats with SimpleITK LabelShapeStatistics."""
    lss = sitk.LabelShapeStatisticsImageFilter()
    lss.Execute(mask_u8)
    labels = list(lss.GetLabels())
    if not labels:
        return {"has_label": False}

    # pick 'label' if present else largest by voxel count
    if label in labels:
        lbl = label
    else:
        sizes = {lbl_i: lss.GetNumberOfPixels(lbl_i) for lbl_i in labels}
        lbl = max(sizes, key=sizes.get)

    out: Dict[str, Any] = {"has_label": True, "label_used": int(lbl)}
    out["num_voxels"] = int(lss.GetNumberOfPixels(lbl))
    out["physical_size_mm3"] = float(lss.GetPhysicalSize(lbl))  # volume in mm^3
    out["centroid_world_xyz_mm"] = tuple(
        map(float, lss.GetCentroid(lbl))
    )  # (x,y,z) in mm

    # Bounding box in index space (start index, size) -> convert to mm
    bb_index = lss.GetBoundingBox(lbl)  # (startX, startY, startZ, sizeX, sizeY, sizeZ)
    out["bbox_index_start_xyz"] = tuple(int(v) for v in bb_index[:3])
    out["bbox_index_size_xyz"] = tuple(int(v) for v in bb_index[3:6])

    # Extra (not always available in older SITK versions) – guard in try/except
    try:
        out["elongation"] = float(lss.GetElongation(lbl))
    except Exception:
        out["elongation"] = np.nan
    # try:
    #     out["feret_diameter_mm"] = compute_feret_diameter(
    #         sitk.GetArrayFromImage(mask_u8), np.array(mask_u8.GetSpacing(), float)
    #     )
    # except Exception:
    #     out["feret_diameter_mm"] = np.nan
    try:
        out["feret_diameter_mm"] = float(lss.GetFeretDiameter(lbl))
    except Exception:
        out["feret_diameter_mm"] = np.nan
    try:
        out["principal_moments"] = tuple(map(float, lss.GetPrincipalMoments(lbl)))
    except Exception:
        out["principal_moments"] = (np.nan, np.nan, np.nan)

    return out


def basic_geometry(mask_u8: sitk.Image) -> Dict[str, Any]:
    """Axis-aligned extent & centroid also via array, plus spacing-aware metrics."""
    logger.info("Computing basic geometry")
    arr = sitk.GetArrayFromImage(mask_u8)  # (z,y,x)
    info: Dict[str, Any] = {}
    if arr.max() == 0:
        info.update(
            {
                "vox_count_total": 0,
                "extent_index_size_xyz": (0, 0, 0),
                "extent_mm_size_xyz": (0.0, 0.0, 0.0),
            }
        )
        return info

    spacing = np.array(mask_u8.GetSpacing(), float)  # (x,y,z)
    coords = np.argwhere(arr > 0)  # (n, 3) in z,y,x
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)  # inclusive index
    size_z, size_y, size_x = (zmax - zmin + 1), (ymax - ymin + 1), (xmax - xmin + 1)
    info["vox_count_total"] = int(coords.shape[0])
    info["extent_index_size_xyz"] = (int(size_x), int(size_y), int(size_z))
    info["extent_mm_size_xyz"] = tuple(
        map(float, np.array([size_x, size_y, size_z]) * spacing)
    )
    return info


def index_center_world(
    img: sitk.Image, idx_xyz: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return tuple(
        map(float, img.TransformIndexToPhysicalPoint([int(round(i)) for i in idx_xyz]))
    )


def compute_location_heuristics(
    vol: sitk.Image,
    centroid_world_mm: Tuple[float, float, float],
    midline_tol_mm: float = 6.0,
    si_thresh_mm: float = 6.0,
) -> Dict[str, Any]:
    """
    Heuristic location flags WITHOUT an atlas.
    - 'sellar' vs 'suprasellar': by centroid superior/inferior shift relative to volume center along SI axis.
    - 'stalk_like' or 'duct_like': by midline proximity (small |x|) and AP near mid-anterior corridor (very rough).
    Notes:
      * This is only a coarse guess; for clinical-grade zones, use an atlas or landmarks.
    """
    logger.info("Computing location heuristics")
    D = np.array(vol.GetDirection(), float).reshape(
        3, 3
    )  # columns: image axes in physical (LPS)
    size = np.array(vol.GetSize(), float)

    # Volume center in world
    center_idx = (size - 1) / 2.0
    center_world = np.array(index_center_world(vol, tuple(center_idx)))
    centroid = np.array(centroid_world_mm)

    # Determine the image axis most aligned with physical Z (superior-inferior)
    si_axis = np.argmax(
        np.abs(D[2, :])
    )  # which image axis contributes most to physical Z
    si_sign = np.sign(D[2, si_axis])  # +1 if increasing index goes superior
    axis_dir = D[:, si_axis]  # unit vector in physical space for that axis

    # Signed SI displacement (mm): positive if centroid is superior to center
    disp_vec = centroid - center_world
    disp_si = float(np.dot(disp_vec, axis_dir) * si_sign)

    # Midline proximity (mm): distance to the mid-sagittal plane estimated by x = center_world_x
    midline_offset_mm = float(abs(centroid[0] - center_world[0]))

    flags = {
        "si_disp_mm": disp_si,
        "midline_offset_mm": midline_offset_mm,
        "is_suprasellar_like": disp_si > si_thresh_mm,
        "is_sellar_like": disp_si < -si_thresh_mm,
        "is_midline_like": midline_offset_mm <= midline_tol_mm,
    }

    # Draft label (non-exclusive)
    label = []
    if flags["is_sellar_like"]:
        label.append("Sellar")
    if flags["is_suprasellar_like"]:
        label.append("Suprasellar")
    if flags["is_midline_like"]:
        # midline proximity could suggest stalk/duct region
        label.append("Stalk/Duct-like")
    if not label:
        label.append("Indeterminate")

    flags["heuristic_region_label"] = "+".join(label)
    return flags


def case_stats(
    case_dir: Path,
    modality_glob: str,
    mask_glob: str,
    si_thresh_mm: float,
    midline_tol_mm: float,
) -> Optional[Dict[str, Any]]:
    case_id = case_dir.name
    vol_p = first_match(case_dir, modality_glob)
    mask_p = first_match(case_dir, mask_glob)

    if not vol_p or not mask_p:
        present = ", ".join(sorted(p.name for p in case_dir.iterdir() if p.is_file()))
        logger.warning(
            "%s: missing volume or mask (vol:%s, mask:%s). Files here: %s",
            case_id,
            vol_p,
            mask_p,
            present,
        )
        return None

    vol = read_vol(vol_p)
    mask = read_vol(mask_p)
    # enforce label dtype
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

    # spacing & voxel volume
    spacing = np.array(vol.GetSpacing(), float)  # (x,y,z) mm
    voxel_mm3 = float(np.prod(spacing))

    # primary stats via SITK
    logger.info(f"Computing label stats for {case_id}")
    s = label_stats(mask, label=1)
    if not s.get("has_label", False) or s.get("num_voxels", 0) == 0:
        return {
            "case_id": case_id,
            "has_label": False,
        }

    # supplement
    geo = basic_geometry(mask)

    # centroid & location heuristics
    centroid_world = s["centroid_world_xyz_mm"]
    loc = compute_location_heuristics(
        vol, centroid_world, midline_tol_mm=midline_tol_mm, si_thresh_mm=si_thresh_mm
    )

    # aggregate
    row: Dict[str, Any] = {
        "case_id": case_id,
        "image_file": vol_p.name,
        "mask_file": mask_p.name,
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "voxel_volume_mm3": voxel_mm3,
        "has_label": True,
        "label_used": s.get("label_used", 1),
        "voxels_in_label": s["num_voxels"],
        "volume_mm3": s["physical_size_mm3"],
        "volume_mL": s["physical_size_mm3"] / 1000.0,
        "centroid_world_x_mm": centroid_world[0],
        "centroid_world_y_mm": centroid_world[1],
        "centroid_world_z_mm": centroid_world[2],
        "bbox_start_x": s["bbox_index_start_xyz"][0],
        "bbox_start_y": s["bbox_index_start_xyz"][1],
        "bbox_start_z": s["bbox_index_start_xyz"][2],
        "bbox_size_x": s["bbox_index_size_xyz"][0],
        "bbox_size_y": s["bbox_index_size_xyz"][1],
        "bbox_size_z": s["bbox_index_size_xyz"][2],
        "extent_size_x_vox": geo["extent_index_size_xyz"][0],
        "extent_size_y_vox": geo["extent_index_size_xyz"][1],
        "extent_size_z_vox": geo["extent_index_size_xyz"][2],
        "extent_size_x_mm": geo["extent_mm_size_xyz"][0],
        "extent_size_y_mm": geo["extent_mm_size_xyz"][1],
        "extent_size_z_mm": geo["extent_mm_size_xyz"][2],
        "feret_diameter_mm": s.get("feret_diameter_mm", np.nan),
        "elongation": s.get("elongation", np.nan),
        "principal_moment_1": (s.get("principal_moments") or (np.nan, np.nan, np.nan))[
            0
        ],
        "principal_moment_2": (s.get("principal_moments") or (np.nan, np.nan, np.nan))[
            1
        ],
        "principal_moment_3": (s.get("principal_moments") or (np.nan, np.nan, np.nan))[
            2
        ],
        "si_disp_mm": loc["si_disp_mm"],
        "midline_offset_mm": loc["midline_offset_mm"],
        "is_sellar_like": loc["is_sellar_like"],
        "is_suprasellar_like": loc["is_suprasellar_like"],
        "is_midline_like": loc["is_midline_like"],
        "heuristic_region_label": loc["heuristic_region_label"],
    }
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Compute tumor mask statistics and heuristic location; save CSV."
    )
    ap.add_argument(
        "--in_dir",
        type=Path,
        required=True,
        help="Directory with case subfolders (e.g., preproc/<CASE>/...).",
    )
    ap.add_argument("--out_csv", type=Path, required=True, help="Output CSV path.")
    ap.add_argument(
        "--modality_glob",
        type=str,
        default="t1*ce*aligned*.nii.gz",
        help="Suffix glob matched after '<CASE>_'. Case-insensitive matching is applied.",
    )
    ap.add_argument(
        "--mask_glob",
        type=str,
        default="mask.nii.gz",  # in argparse defaults
        help="Suffix glob matched after '<CASE>_'.",
    )
    ap.add_argument(
        "--si_thresh_mm",
        type=float,
        default=6.0,
        help="Threshold (mm) to call suprasellar (>+thr) vs sellar (<-thr) relative to volume center along SI axis.",
    )
    ap.add_argument(
        "--midline_tol_mm",
        type=float,
        default=6.0,
        help="Tolerance (mm) to call midline proximity (stalk/duct-like).",
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
    logger.info(f"Case dir: {args.in_dir}")
    logger.info(f"Output CSV: {args.out_csv}")
    logger.info(f"Modality glob: {args.modality_glob}")
    logger.info(f"Mask glob: {args.mask_glob}")
    logger.info(f"SI threshold (mm): {args.si_thresh_mm}")
    logger.info(f"Midline tolerance (mm): {args.midline_tol_mm}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Log file: {args.log_file}")
    logger.info(f"Cases: {sorted(args.in_dir.iterdir())}")

    rows: List[Dict[str, Any]] = []
    cases = [d for d in sorted(args.in_dir.iterdir()) if d.is_dir()]
    if not cases:
        raise RuntimeError(f"No case folders found in {args.in_dir}")

    for c in cases:
        logger.info(f"Processing {c.name}")
        try:
            row = case_stats(
                c,
                args.modality_glob,
                args.mask_glob,
                si_thresh_mm=args.si_thresh_mm,
                midline_tol_mm=args.midline_tol_mm,
            )
            if row is not None:
                rows.append(row)
            else:
                print(f"[WARN] Skipped {c.name}")
        except Exception as e:
            print(f"[FAIL] {c.name}: {e}")

    if rows:
        df = pd.DataFrame(rows).sort_values("case_id")
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[OK] Wrote {args.out_csv} with {len(df)} rows.")
    else:
        print("[WARN] No rows written.")


if __name__ == "__main__":
    main()
