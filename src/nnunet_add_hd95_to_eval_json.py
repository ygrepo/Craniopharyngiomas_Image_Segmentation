#!/usr/bin/env python3
"""
add_hd95_to_eval_json.py
- Reads an nnUNet v2 evaluation JSON (e.g., summary.json or per-case list)
- For each case and each class present in 'metrics', computes label HD95 (mm)
  from the predicted segmentation and the reference label volume.
- Writes a new JSON with the extra 'HD95' per class/region.

Requires: SimpleITK
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def load_items(data: Any) -> List[Dict[str, Any]]:
    """Accept list or dict with common keys, return per-case items."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("results", "cases", "items", "per_case"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        if any(k in data for k in ("prediction_file", "reference_file", "metrics")):
            return [data]
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError("Unrecognized JSON structure for nnU-Net evaluation output.")


def binary_contour(img: sitk.Image) -> sitk.Image:
    """Surface voxels of a binary mask (1 on border)."""
    return sitk.BinaryContour(
        img, fullyConnected=True, backgroundValue=0, foregroundValue=1
    )


def hd95_mm_from_binary(pred_bin: sitk.Image, ref_bin: sitk.Image) -> float:
    """
    Symmetric 95% Hausdorff distance in mm between two binary masks (same geometry).
    Uses SimpleITK SignedMaurerDistanceMap with useImageSpacing=True.
    """
    # --- Edge cases (correct foreground check) ---
    sf = sitk.StatisticsImageFilter()
    sf.Execute(pred_bin)
    pred_any = sf.GetSum() > 0
    sf.Execute(ref_bin)
    ref_any = sf.GetSum() > 0

    if not pred_any and not ref_any:
        return 0.0
    if (pred_any and not ref_any) or (ref_any and not pred_any):
        return float("inf")

    # Surfaces
    surf_pred = sitk.BinaryContour(
        pred_bin, fullyConnected=True, backgroundValue=0, foregroundValue=1
    )
    surf_ref = sitk.BinaryContour(
        ref_bin, fullyConnected=True, backgroundValue=0, foregroundValue=1
    )

    # Distance maps (abs, in mm)
    dm_ref = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            ref_bin, squaredDistance=False, useImageSpacing=True
        )
    )
    dm_pred = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            pred_bin, squaredDistance=False, useImageSpacing=True
        )
    )

    # Distances from each surface to the other
    d_pred_to_ref = sitk.GetArrayFromImage(sitk.Mask(dm_ref, surf_pred))
    d_ref_to_pred = sitk.GetArrayFromImage(sitk.Mask(dm_pred, surf_ref))

    a = d_pred_to_ref[d_pred_to_ref > 0]
    b = d_ref_to_pred[d_ref_to_pred > 0]
    if a.size == 0 and b.size == 0:
        return 0.0
    all_d = np.concatenate([a, b]) if a.size and b.size else (a if a.size else b)
    return float(np.percentile(all_d.astype(np.float64), 95))


def load_region_spec(dj_path: Optional[str]) -> Optional[Dict[str, List[int]]]:
    """
    Parse dataset.json to get region definitions.
    Returns mapping like {"whole_tumor":[1,2,3], "tumor_core":[2,3], "enhancing_tumor":[3]}
    or None if dj_path is None.
    """
    if not dj_path:
        return None
    with open(dj_path, "r") as f:
        dj = json.load(f)
    labels = dj.get("labels", {})
    # normalize keys
    region_map = {}
    for k, v in labels.items():
        if k == "background":
            continue
        if isinstance(v, list):
            region_map[k] = [int(x) for x in v]
        elif isinstance(v, int):
            # plain class (not a region)
            continue
    return region_map if region_map else None


def region_masks_from_labels(
    lbl_img: sitk.Image, region_to_classes: Dict[str, List[int]]
) -> Dict[str, sitk.Image]:
    """
    Build region binary masks from a discrete label image using unions of class ids.
    """
    masks = {}
    for rname, cls_list in region_to_classes.items():
        mask = None
        for cid in cls_list:
            m = sitk.Equal(lbl_img, int(cid))
            mask = m if mask is None else sitk.Or(mask, m)
        if mask is None:
            # empty (shouldn't happen), create zeros of same size
            mask = sitk.Equal(lbl_img, -9999)
        masks[rname] = mask
    return masks


def add_hd95_to_item(
    item: Dict[str, Any],
    class_ids: Optional[List[int]] = None,
    region_to_classes: Optional[Dict[str, List[int]]] = None,
) -> None:
    """
    Compute HD95 for label-classes (existing behavior) and, if region_to_classes is given,
    also compute HD95 for regions (WT/TC/ET) from unions of labels.
    Writes to:
      item['metrics'][<class_id>]['HD95']  (labels)
      item['metrics']['regions'][<region_name>]['HD95']  (regions)
    """
    pred_path = (
        item.get("prediction_file") or item.get("prediction") or item.get("pred")
    )
    ref_path = item.get("reference_file") or item.get("gt") or item.get("reference")
    if not (pred_path and ref_path):
        return

    pred_img = sitk.ReadImage(pred_path)
    ref_img = sitk.ReadImage(ref_path)

    metrics = item.get("metrics", {})

    # -------- Label classes (unchanged behavior) --------
    if class_ids is None:
        cls_ids = []
        for k in metrics.keys():
            # only pick numeric keys at this level
            try:
                cid = int(k)
            except Exception:
                continue
            if cid != 0:
                cls_ids.append(cid)
    else:
        cls_ids = class_ids

    for cid in cls_ids:
        pred_bin = sitk.Equal(pred_img, int(cid))
        ref_bin = sitk.Equal(ref_img, int(cid))
        try:
            val = hd95_mm_from_binary(pred_bin, ref_bin)
        except Exception:
            val = float("nan")

        m = metrics.get(str(cid))
        if not isinstance(m, dict):
            m = {}
            metrics[str(cid)] = m
        m["HD95"] = val

    # -------- Regions (WT/TC/ET) --------
    if region_to_classes:
        # ensure subtree exists
        regions_metrics = metrics.get("regions")
        if not isinstance(regions_metrics, dict):
            regions_metrics = {}
            metrics["regions"] = regions_metrics

        pred_regions = region_masks_from_labels(pred_img, region_to_classes)
        ref_regions = region_masks_from_labels(ref_img, region_to_classes)

        for rname, p_mask in pred_regions.items():
            r_mask = ref_regions[rname]
            try:
                val = hd95_mm_from_binary(p_mask, r_mask)
            except Exception:
                val = float("nan")
            m = regions_metrics.get(rname)
            if not isinstance(m, dict):
                m = {}
                regions_metrics[rname] = m
            m["HD95"] = val

    item["metrics"] = metrics  # write-back


def main():
    ap = argparse.ArgumentParser(
        description="Add HD95 (mm) per class/region to an nnUNet v2 evaluation JSON."
    )
    ap.add_argument(
        "-i", "--input", required=True, help="summary.json or per-case JSON"
    )
    ap.add_argument("-o", "--output", default=None, help="Default: *_with_hd95.json")
    ap.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class IDs for label HD95 (default: all nonzero ids present).",
    )
    ap.add_argument(
        "--dataset_json",
        default=None,
        help="Path to dataset.json (enables region HD95 using its 'labels' unions).",
    )
    args = ap.parse_args()
    setup_logging(None, "INFO")

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Classes: {args.classes}")
    logger.info(f"Dataset JSON: {args.dataset_json}")

    class_ids = (
        [int(x.strip()) for x in args.classes.split(",")] if args.classes else None
    )
    region_to_classes = load_region_spec(args.dataset_json)
    logger.info(f"Region to classes: {region_to_classes}")

    with open(args.input, "r") as f:
        data = json.load(f)

    items = load_items(data)
    for it in items:
        add_hd95_to_item(it, class_ids, region_to_classes)

    # preserve top-level shape
    if isinstance(data, list):
        new_data = items
    elif isinstance(data, dict):
        placed = False
        for k in ("results", "cases", "items", "per_case"):
            if isinstance(data.get(k), list):
                data[k] = items
                placed = True
                break
        if not placed and (
            any(k in data for k in ("prediction_file", "reference_file", "metrics"))
        ):
            data = items[0] if items else data
        new_data = data
    else:
        new_data = items

    out_path = args.output or (os.path.splitext(args.input)[0] + "_with_hd95.json")
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)

    logger.info(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
