#!/usr/bin/env python3
"""
add_hd95_to_eval_json.py
- Reads an nnUNet v2 evaluation JSON (e.g., summary.json or per-case list)
- For each case and each class present in 'metrics', computes HD95 (mm)
  from the predicted segmentation and the reference label volume.
- Writes a new JSON with the extra 'HD95' per class.

Requires: SimpleITK
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import SimpleITK as sitk


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
    # Edge cases
    pred_any = int(sitk.StatisticsImageFilter().Execute(pred_bin) or 0)
    ref_any = int(sitk.StatisticsImageFilter().Execute(ref_bin) or 0)
    if not pred_any and not ref_any:
        return 0.0
    if pred_any and not ref_any or ref_any and not pred_any:
        return float("inf")

    # Surfaces
    surf_pred = binary_contour(pred_bin)
    surf_ref = binary_contour(ref_bin)

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

    # Collect positive distances
    a = d_pred_to_ref[d_pred_to_ref > 0]
    b = d_ref_to_pred[d_ref_to_pred > 0]
    if a.size == 0 and b.size == 0:
        return 0.0
    all_d = np.concatenate([a, b]) if a.size and b.size else (a if a.size else b)
    return float(np.percentile(all_d.astype(np.float64), 95))


def add_hd95_to_item(
    item: Dict[str, Any], class_ids: Optional[List[int]] = None
) -> None:
    """
    Compute HD95 per class for a single eval item and insert into item['metrics'][cls]['HD95'].
    """
    pred_path = (
        item.get("prediction_file") or item.get("prediction") or item.get("pred")
    )
    ref_path = item.get("reference_file") or item.get("gt") or item.get("reference")
    if not (pred_path and ref_path):
        return

    # Read images (keep everything in SITK space so spacing/origin/direction are honored)
    pred_img = sitk.ReadImage(pred_path)
    ref_img = sitk.ReadImage(ref_path)

    # classes to process: from metrics keys unless provided
    metrics = item.get("metrics", {})
    if class_ids is None:
        cls_ids = []
        for k in metrics.keys():
            try:
                cid = int(k)
            except Exception:
                continue
            if cid != 0:
                cls_ids.append(cid)
    else:
        cls_ids = class_ids

    for cid in cls_ids:
        # Make binary masks for the class directly in SITK (no numpy re-spacing headaches)
        pred_bin = sitk.Equal(pred_img, int(cid))
        ref_bin = sitk.Equal(ref_img, int(cid))

        try:
            val = hd95_mm_from_binary(pred_bin, ref_bin)
        except Exception as e:
            # Robust fallback: mark as NaN so downstream can spot failures
            val = float("nan")

        # Insert into metrics dict, creating sub-dict if needed
        m = metrics.get(str(cid))
        if not isinstance(m, dict):
            m = {}
            metrics[str(cid)] = m
        m["HD95"] = val

    item["metrics"] = metrics  # ensure write-back


def main():
    ap = argparse.ArgumentParser(
        description="Add HD95 (mm) per class to an nnUNet v2 evaluation JSON."
    )
    ap.add_argument(
        "-i", "--input", required=True, help="Path to eval JSON (e.g., summary.json)"
    )
    ap.add_argument(
        "-o", "--output", default=None, help="Output JSON (default: *_with_hd95.json)"
    )
    ap.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class IDs to compute (default: all non-zero classes present in metrics).",
    )
    args = ap.parse_args()

    out_path = args.output or (os.path.splitext(args.input)[0] + "_with_hd95.json")
    class_ids = None
    if args.classes:
        class_ids = [int(x.strip()) for x in args.classes.split(",") if x.strip()]

    with open(args.input, "r") as f:
        data = json.load(f)

    items = load_items(data)
    for it in items:
        add_hd95_to_item(it, class_ids)

    # If the original top-level was a list, keep it a list; if it was a dict with a key, keep that too
    if isinstance(data, list):
        new_data = items
    elif isinstance(data, dict):
        # Try to put back under the same key if possible
        placed = False
        for k in ("results", "cases", "items", "per_case"):
            if isinstance(data.get(k), list):
                data[k] = items
                placed = True
                break
        if not placed and (
            any(k in data for k in ("prediction_file", "reference_file", "metrics"))
        ):
            # Single item dict
            data = items[0] if items else data
        new_data = data
    else:
        new_data = items

    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
