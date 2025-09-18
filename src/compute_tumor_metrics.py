import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import fnmatch
import SimpleITK as sitk
from skimage import measure

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
    files = [p for p in case_dir.iterdir() if p.is_file()]
    patt = suffix_glob.lower()
    if "." not in patt:
        patt = patt + "*"
    pattern = f"*_{patt}"
    hits = [p for p in files if fnmatch.fnmatch(p.name.lower(), pattern)]
    return sorted(hits)[0] if hits else None


# ---------- Shape metrics ----------
def shape_metrics(mask_u8: sitk.Image) -> dict:
    arr = sitk.GetArrayFromImage(mask_u8).astype(bool)
    spacing = np.array(mask_u8.GetSpacing(), float)

    verts, faces, _, _ = measure.marching_cubes(arr, level=0.5, spacing=spacing[::-1])
    surface_area = measure.mesh_surface_area(verts, faces)
    volume = np.sum(arr) * np.prod(spacing)

    props = measure.regionprops(arr.astype(np.uint8))[0]
    if props.inertia_tensor_eigvals is not None:
        eigvals = np.sort(props.inertia_tensor_eigvals)[::-1]
        elongation = np.sqrt(eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else np.nan
    else:
        elongation = np.nan

    sphericity = (
        (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area
        if surface_area > 0
        else np.nan
    )
    compactness = (volume**2) / (surface_area**3) if surface_area > 0 else np.nan

    return {
        "surface_area_mm2": surface_area,
        "sphericity": sphericity,
        "compactness": compactness,
    }


def compute_feret_diameter(mask_u8: sitk.Image) -> float:
    boundary_u8 = sitk.Subtract(mask_u8, sitk.BinaryErode(mask_u8, (1, 1, 1)))
    boundary_arr = sitk.GetArrayFromImage(boundary_u8).astype(bool)
    if not boundary_arr.any():
        return 0.0

    spacing = np.array(mask_u8.GetSpacing(), float)
    coords_zyx = np.argwhere(boundary_arr)
    coords_xyz = coords_zyx[:, ::-1].astype(np.float64)
    pts_mm = coords_xyz * spacing

    M = len(pts_mm)
    if M > 100_000:
        step = int(np.ceil(M / 50_000))
        pts_mm = pts_mm[::step]

    try:
        from scipy.spatial import ConvexHull, distance

        hull = ConvexHull(pts_mm, qhull_options="QJ")
        verts = pts_mm[hull.vertices]
        dmax = (
            float(np.max(distance.pdist(verts, metric="euclidean")))
            if len(verts) > 1
            else 0.0
        )
        return dmax
    except Exception:
        if len(pts_mm) < 2:
            return 0.0
        idx = 0
        a = pts_mm[idx]
        for _ in range(6):
            d2 = np.sum((pts_mm - a) ** 2, axis=1)
            j = int(np.argmax(d2))
            b = pts_mm[j]
            d2b = np.sum((pts_mm - b) ** 2, axis=1)
            i = int(np.argmax(d2b))
            a = pts_mm[i]
        return float(np.linalg.norm(a - b))


def label_stats(mask_u8: sitk.Image, label: int = 1) -> Dict[str, Any]:
    lss = sitk.LabelShapeStatisticsImageFilter()
    lss.Execute(mask_u8)
    labels = list(lss.GetLabels())
    if not labels:
        return {"has_label": False}

    if label in labels:
        lbl = label
    else:
        sizes = {lbl_i: lss.GetNumberOfPixels(lbl_i) for lbl_i in labels}
        lbl = max(sizes, key=sizes.get)

    out: Dict[str, Any] = {"has_label": True, "label_used": int(lbl)}
    out["num_voxels"] = int(lss.GetNumberOfPixels(lbl))
    out["physical_size_mm3"] = float(lss.GetPhysicalSize(lbl))
    out["centroid_world_xyz_mm"] = tuple(map(float, lss.GetCentroid(lbl)))
    bb_index = lss.GetBoundingBox(lbl)
    out["bbox_index_start_xyz"] = tuple(int(v) for v in bb_index[:3])
    out["bbox_index_size_xyz"] = tuple(int(v) for v in bb_index[3:6])

    try:
        out["elongation"] = float(lss.GetElongation(lbl))
    except Exception:
        out["elongation"] = np.nan
    try:
        out["max_diameter_mm"] = compute_feret_diameter(mask_u8)  # renamed
    except Exception:
        out["max_diameter_mm"] = np.nan
    try:
        out["principal_moments"] = tuple(map(float, lss.GetPrincipalMoments(lbl)))
    except Exception:
        out["principal_moments"] = (np.nan, np.nan, np.nan)

    out.update(shape_metrics(mask_u8))
    return out


def basic_geometry(mask_u8: sitk.Image) -> Dict[str, Any]:
    arr = sitk.GetArrayFromImage(mask_u8)
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

    spacing = np.array(mask_u8.GetSpacing(), float)
    coords = np.argwhere(arr > 0)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    size_z, size_y, size_x = (zmax - zmin + 1), (ymax - ymin + 1), (xmax - xmin + 1)
    info["vox_count_total"] = int(coords.shape[0])
    info["extent_index_size_xyz"] = (int(size_x), int(size_y), int(size_z))
    info["extent_mm_size_xyz"] = tuple(
        map(float, np.array([size_x, size_y, size_z]) * spacing)
    )
    return info


# ---------- Helpers (orientation + heuristics) ----------
def index_center_world(
    img: sitk.Image, idx_xyz: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return tuple(
        map(float, img.TransformIndexToPhysicalPoint([int(round(i)) for i in idx_xyz]))
    )


def axis_mapping_from_direction(D: np.ndarray) -> dict:
    """
    Map physical axes (LPS: X=Left-Right, Y=Posterior-Anterior, Z=Inferior-Superior)
    to image axes (0,1,2). Returns dict with keys 'SI', 'AP', 'LR'.
    """
    # rows are physical axes (L, P, S). We want which image axis maximally aligns.
    si_axis = int(np.argmax(np.abs(D[2, :])))  # physical Z (S/I)
    ap_axis = int(np.argmax(np.abs(D[1, :])))  # physical Y (P/A)
    lr_axis = int(np.argmax(np.abs(D[0, :])))  # physical X (L/R)
    return {"SI": si_axis, "AP": ap_axis, "LR": lr_axis}


def ap_displacement_mm(
    vol: sitk.Image, centroid_world_mm: Tuple[float, float, float]
) -> float:
    """
    Signed displacement along physical AP (Y) axis in mm.
    Positive -> posterior; Negative -> anterior.
    """
    D = np.array(vol.GetDirection(), float).reshape(3, 3)
    size = np.array(vol.GetSize(), float)
    center_idx = (size - 1) / 2.0
    center_world = np.array(index_center_world(vol, tuple(center_idx)))
    disp_vec = np.array(centroid_world_mm) - center_world

    ap_img_axis = int(np.argmax(np.abs(D[1, :])))
    ap_dir = D[:, ap_img_axis]
    ap_sign = np.sign(D[1, ap_img_axis])  # +1 if increasing index goes posterior
    return float(np.dot(disp_vec, ap_dir) * ap_sign)


def max_transverse_slice_location(
    mask_u8: sitk.Image, si_img_axis: int, si_center_tol_mm: float
) -> Tuple[str, int]:
    """
    Find slice along SI with maximum cross-sectional area.
    Returns ('inferior'|'superior'|'indeterminate', slice_index)
    """
    arr = sitk.GetArrayFromImage(mask_u8)  # (z,y,x)
    spacing_xyz = np.array(mask_u8.GetSpacing(), float)  # (x,y,z)

    # map image axis -> array axis index (z=0,y=1,x=2)
    imgax_to_arrax = {2: 0, 1: 1, 0: 2}
    si_arr_axis = imgax_to_arrax[si_img_axis]

    arr_si_major = np.moveaxis(arr, si_arr_axis, 0)  # (S, .., ..)

    # pixel area of an in-plane slice (two axes not including SI)
    other_img_axes = [a for a in (0, 1, 2) if a != si_img_axis]
    pix_area = float(spacing_xyz[other_img_axes[0]] * spacing_xyz[other_img_axes[1]])
    areas = (arr_si_major > 0).sum(axis=(1, 2)) * pix_area
    k = int(np.argmax(areas))

    # compare slice position vs volume center
    size = np.array(mask_u8.GetSize(), float)
    center_idx = (size[si_img_axis] - 1) / 2.0
    offset_idx = k - center_idx
    step_mm = float(spacing_xyz[si_img_axis])
    offset_mm = offset_idx * step_mm
    if offset_mm > si_center_tol_mm:
        loc = "superior"
    elif offset_mm < -si_center_tol_mm:
        loc = "inferior"
    else:
        loc = "indeterminate"
    return loc, k


def compute_location_heuristics(
    vol: sitk.Image,
    centroid_world_mm: Tuple[float, float, float],
    midline_tol_mm: float = 6.0,
    si_thresh_mm: float = 6.0,
) -> Dict[str, Any]:
    """
    Heuristic location flags WITHOUT an atlas.
    """
    D = np.array(vol.GetDirection(), float).reshape(3, 3)
    size = np.array(vol.GetSize(), float)

    # Volume center in world
    center_idx = (size - 1) / 2.0
    center_world = np.array(index_center_world(vol, tuple(center_idx)))
    centroid = np.array(centroid_world_mm)

    # SI axis and sign (toward superior is +)
    si_img_axis = int(np.argmax(np.abs(D[2, :])))
    si_dir = D[:, si_img_axis]
    si_sign = np.sign(D[2, si_img_axis])

    disp_vec = centroid - center_world
    disp_si = float(np.dot(disp_vec, si_dir) * si_sign)

    # Midline proximity (x ~ center_world_x)
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
        label.append("Stalk/Duct-like")
    if not label:
        label.append("Indeterminate")

    flags["heuristic_region_label"] = "+".join(label)
    return flags


# ---------- Q/S/T points from available data ----------
def compute_qst_points(
    *,
    ap_diam_cm: float,
    si_diam_cm: float,
    ap_position: str,
    max_transv_loc: str,
    aspect_ratio_gt_1p1: bool,
    pituitary_seen: str,
    ventricles_dilated: str,
    pit_fossa_vol_cm3: Optional[float],
    front_tub_sellae_vol_cm3: Optional[float],
) -> Tuple[int, int, int]:
    Q_pts = 0
    S_pts = 0
    T_pts = 0

    # Q rules (examples from your table subset)
    if pituitary_seen == "not_clear":
        Q_pts += 8
    if np.isfinite(ap_diam_cm) and ap_diam_cm < 3.5:
        Q_pts += 3
    if np.isfinite(si_diam_cm) and si_diam_cm > 3.5:
        Q_pts += 2

    # S rules
    if ventricles_dilated == "no":
        S_pts += 5
    if ap_position == "anterior":
        S_pts += 2
    if max_transv_loc == "inferior":
        S_pts += 2
    if (front_tub_sellae_vol_cm3 is not None) and (front_tub_sellae_vol_cm3 > 0.8):
        S_pts += 3
    if aspect_ratio_gt_1p1:
        S_pts += 4

    # T rules
    if pituitary_seen == "clear":
        T_pts += 6
    if ventricles_dilated == "yes":
        T_pts += 3
    if ap_position == "posterior":
        T_pts += 1
    if max_transv_loc == "superior":
        T_pts += 2
    if (pit_fossa_vol_cm3 is not None) and (pit_fossa_vol_cm3 < 2.1):
        T_pts += 4

    return Q_pts, S_pts, T_pts


# ---------- Case stats ----------
def case_stats(
    case_dir: Path,
    modality_glob: str,
    mask_glob: str,
    *,
    si_thresh_mm: float,
    midline_tol_mm: float,
    ap_thresh_mm: float,
    si_center_tol_mm: float,
    pituitary_seen: str,
    ventricles_dilated: str,
    pit_fossa_vol_cm3: Optional[float],
    front_tub_sellae_vol_cm3: Optional[float],
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
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

    spacing = np.array(vol.GetSpacing(), float)  # (x,y,z)
    voxel_mm3 = float(np.prod(spacing))

    # primary stats via SITK
    logger.info(f"Computing label stats for {case_id}")
    s = label_stats(mask, label=1)
    if not s.get("has_label", False) or s.get("num_voxels", 0) == 0:
        return {"case_id": case_id, "label_present": False}

    # extents
    geo = basic_geometry(mask)
    centroid_world = s["centroid_world_xyz_mm"]

    # axis mapping
    D = np.array(vol.GetDirection(), float).reshape(3, 3)
    axes = axis_mapping_from_direction(D)

    # extent (mm) along x/y/z in the *image* axis sense
    extent_mm_xyz = np.array(
        [
            geo["extent_mm_size_xyz"][0],
            geo["extent_mm_size_xyz"][1],
            geo["extent_mm_size_xyz"][2],
        ],
        dtype=float,
    )

    # diameters in cm: SI (height), AP (depth)
    si_diam_mm = float(extent_mm_xyz[axes["SI"]])
    ap_diam_mm = float(extent_mm_xyz[axes["AP"]])
    si_diam_cm = si_diam_mm / 10.0
    ap_diam_cm = ap_diam_mm / 10.0

    # aspect ratio AP/SI
    aspect_ratio = ap_diam_cm / si_diam_cm if si_diam_cm > 0 else np.nan
    aspect_ratio_gt_1p1 = (
        bool(aspect_ratio > 1.1) if np.isfinite(aspect_ratio) else False
    )

    # AP displacement and categorical call
    disp_ap = ap_displacement_mm(vol, centroid_world)
    if disp_ap < -ap_thresh_mm:
        ap_position = "anterior"
    elif disp_ap > ap_thresh_mm:
        ap_position = "posterior"
    else:
        ap_position = "indeterminate"

    # Max transverse slice location (inferior/superior/indeterminate)
    max_transv_loc, _ = max_transverse_slice_location(
        mask, axes["SI"], si_center_tol_mm
    )

    # volumes
    vol_mm3 = s["physical_size_mm3"]
    vol_cm3 = vol_mm3 / 1000.0

    # Q/S/T points
    Q_pts, S_pts, T_pts = compute_qst_points(
        ap_diam_cm=ap_diam_cm,
        si_diam_cm=si_diam_cm,
        ap_position=ap_position,
        max_transv_loc=max_transv_loc,
        aspect_ratio_gt_1p1=aspect_ratio_gt_1p1,
        pituitary_seen=pituitary_seen,
        ventricles_dilated=ventricles_dilated,
        pit_fossa_vol_cm3=pit_fossa_vol_cm3,
        front_tub_sellae_vol_cm3=front_tub_sellae_vol_cm3,
    )

    # SI/midline heuristics (existing)
    loc = compute_location_heuristics(
        vol,
        centroid_world,
        midline_tol_mm=midline_tol_mm,
        si_thresh_mm=si_thresh_mm,
    )

    # row with cleaned labels
    row: Dict[str, Any] = {
        # IDs & files
        "case_id": case_id,
        "img_file": vol_p.name,
        "mask_file": mask_p.name,
        # voxel / spacing
        # "spacing_mm_x": spacing[0],
        # "spacing_mm_y": spacing[1],
        # "spacing_mm_z": spacing[2],
        # "voxel_mm3": voxel_mm3,
        # label basics
        # "label_present": True,
        # "label": s.get("label_used", 1),
        # "voxel_count": s["num_voxels"],
        # # volumes & area
        # "vol_mm3": vol_mm3,
        # "vol_cm3": vol_cm3,  # explicit cm^3
        "surface_area_mm2": s.get("surface_area_mm2", np.nan),
        # shape descriptors
        "sphericity": s.get("sphericity", np.nan),
        "compactness": s.get("compactness", np.nan),
        # diameters & aspect ratio
        "height_cm": si_diam_cm,  # SI
        "depth_cm": ap_diam_cm,  # AP
        "aspect_ratio_ap_over_si": aspect_ratio,
        "aspect_ratio_gt_1_1": aspect_ratio_gt_1p1,
        # centroid (world, mm)
        # "centroid_mm_x": centroid_world[0],
        # "centroid_mm_y": centroid_world[1],
        # "centroid_mm_z": centroid_world[2],
        # # bounding box (index space)
        # "bbox_start_x": s["bbox_index_start_xyz"][0],
        # "bbox_start_y": s["bbox_index_start_xyz"][1],
        # "bbox_start_z": s["bbox_index_start_xyz"][2],
        # "bbox_size_x": s["bbox_index_size_xyz"][0],
        # "bbox_size_y": s["bbox_index_size_xyz"][1],
        # "bbox_size_z": s["bbox_index_size_xyz"][2],
        # # extents (voxels & mm)
        # "extent_vox_x": geo["extent_index_size_xyz"][0],
        # "extent_vox_y": geo["extent_index_size_xyz"][1],
        # "extent_vox_z": geo["extent_index_size_xyz"][2],
        # "extent_mm_x": geo["extent_mm_size_xyz"][0],
        # "extent_mm_y": geo["extent_mm_size_xyz"][1],
        # "extent_mm_z": geo["extent_mm_size_xyz"][2],
        # # additional shape
        "max_diameter_mm": s.get("max_diameter_mm", np.nan),  # Feret (renamed)
        "elongation": s.get("elongation", np.nan),
        # "pm1": (s.get("principal_moments") or (np.nan, np.nan, np.nan))[0],
        # "pm2": (s.get("principal_moments") or (np.nan, np.nan, np.nan))[1],
        # "pm3": (s.get("principal_moments") or (np.nan, np.nan, np.nan))[2],
        # SI/midline heuristics
        "vertical_shift_mm": loc["si_disp_mm"],  # renamed from si_disp_mm
        "midline_shift_mm": loc["midline_offset_mm"],  # renamed from midline_offset_mm
        "sellar_like": loc["is_sellar_like"],
        "suprasellar_like": loc["is_suprasellar_like"],
        "midline_like": loc["is_midline_like"],
        "region_heuristic": loc["heuristic_region_label"],
        # AP heuristics
        "ap_shift_mm": disp_ap,  # renamed from ap_disp_mm
        "ap_position": ap_position,  # anterior/posterior/indeterminate
        "max_area_slice_loc": max_transv_loc,  # inferior/superior/indeterminate
        # Q/S/T partial points
        "Q_pts": Q_pts,
        "S_pts": S_pts,
        "T_pts": T_pts,
        # inputs echoed back
        "ventricles_dilated": ventricles_dilated,
        "pituitary_seen": pituitary_seen,
        "pit_fossa_cm3": pit_fossa_vol_cm3,
        "front_tub_sellae_cm3": front_tub_sellae_vol_cm3,
    }
    return row


# ---------- CLI / main ----------
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
        help="Suffix glob matched after '<CASE>_'. Case-insensitive.",
    )
    ap.add_argument(
        "--mask_glob",
        type=str,
        default="mask.nii.gz",
        help="Suffix glob matched after '<CASE>_'.",
    )
    # thresholds / heuristics
    ap.add_argument(
        "--si_thresh_mm",
        type=float,
        default=6.0,
        help="SI threshold for sellar/suprasellar call.",
    )
    ap.add_argument(
        "--midline_tol_mm",
        type=float,
        default=6.0,
        help="Tolerance (mm) for midline proximity.",
    )
    ap.add_argument(
        "--ap_thresh_mm",
        type=float,
        default=6.0,
        help="Threshold (mm) for anterior/posterior call.",
    )
    ap.add_argument(
        "--si_center_tol_mm",
        type=float,
        default=6.0,
        help="Tolerance (mm) to call max area slice superior vs inferior.",
    )

    # inputs used in partial Q/S/T scores
    ap.add_argument(
        "--pituitary_seen",
        choices=["clear", "not_clear", "na"],
        default="na",
        help="If the pituitary is clearly seen (for Q/T scoring).",
    )
    ap.add_argument(
        "--ventricles_dilated",
        choices=["yes", "no", "na"],
        default="na",
        help="If ventricles are dilated (for S/T scoring).",
    )
    ap.add_argument(
        "--pit_fossa_vol_cm3",
        type=float,
        default=None,
        help="Tumor volume inside pituitary fossa (cm^3), if available (T scoring).",
    )
    ap.add_argument(
        "--front_tub_sellae_vol_cm3",
        type=float,
        default=None,
        help="Tumor volume anterior to tuberculum sellae (cm^3), if available (S scoring).",
    )

    # logging
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
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)

    logger.info(f"Args: {args}")
    logger.info(f"Case dir: {args.in_dir}")
    logger.info(f"Output CSV: {args.out_csv}")

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
                ap_thresh_mm=args.ap_thresh_mm,
                si_center_tol_mm=args.si_center_tol_mm,
                pituitary_seen=args.pituitary_seen,
                ventricles_dilated=args.ventricles_dilated,
                pit_fossa_vol_cm3=args.pit_fossa_vol_cm3,
                front_tub_sellae_vol_cm3=args.front_tub_sellae_vol_cm3,
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
