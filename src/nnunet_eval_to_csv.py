#!/usr/bin/env python3
"""
export_eval_to_csv.py

Read an nnU-Net v2 evaluation JSON (e.g., summary_with_hd95.json),
normalize metric names per class, derive missing metrics when possible
(PPV, NPV, Jaccard from counts / Dice), and export:

  1) <base>_cases.csv   — one row per (case, class)
  2) <base>_summary.csv — one row per class with mean/median/std (scores) and sums (counts)

Notes
-----
- Jaccard == IoU for binary classes. We store it under the canonical name "Jaccard".
- If HD95 was added previously (e.g., by your add_hd95 script), it will be picked up.
- Counts are summed: TP, TN, FP, FN, n_pred, n_ref.
- Scores are averaged: Dice, Jaccard, PPV, NPV, HD95 (optionally labeled HD95_mm).

Usage
-----
python export_eval_to_csv.py \
  -i nnUNet_results/.../summary_with_hd95.json \
  --round 4 --rename-hd95-mm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

# Keys treated as counts (per case, per class/region)
COUNT_KEYS = {"tp", "tn", "fp", "fn", "n_pred", "n_ref"}

# Aliases recognized in incoming JSON
ALIASES = {
    "dice": {"Dice", "DICE", "dice"},
    "iou": {"IoU", "IOU", "Jaccard", "jaccard", "iou"},
    "ppv": {"PPV", "Precision", "precision", "Positive Predictive Value"},
    "npv": {"NPV", "Negative Predictive Value"},
    "hd95": {"HD95", "hd95", "Hausdorff95", "Hausdorff_95", "Hausdorff 95"},
    "tp": {"TP", "tp"},
    "tn": {"TN", "tn"},
    "fp": {"FP", "fp"},
    "fn": {"FN", "fn"},
    "n_pred": {"n_pred", "N_pred", "N_Pred"},
    "n_ref": {"n_ref", "N_ref", "N_Ref"},
}
IOU_ALIASES = {x for x in ALIASES["iou"] if x != "Jaccard"}


# -------------------------- helpers -------------------------- #
def _first_present(d: Dict[str, Any], names: Sequence[str]) -> Optional[float]:
    """Return first matching key's float value, else None."""
    for k in d.keys():
        if k in names:
            try:
                return float(d[k])
            except Exception:
                return None
    return None


def _norm_case_metrics(md: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Normalize a single (case, class/region) metrics dict to canonical keys:
      Dice, Jaccard, PPV, NPV, HD95, tp, tn, fp, fn, n_pred, n_ref
    Derive Jaccard (IoU) from counts or Dice; derive PPV/NPV from counts if needed.
    Non-canonical keys are preserved in the row.
    """
    out: Dict[str, Optional[float]] = {}

    # canonical scores
    dice = _first_present(md, ALIASES["dice"])
    iou = _first_present(md, ALIASES["iou"])
    ppv = _first_present(md, ALIASES["ppv"])
    npv = _first_present(md, ALIASES["npv"])
    hd95 = _first_present(md, ALIASES["hd95"])

    # counts
    tp = _first_present(md, ALIASES["tp"])
    tn = _first_present(md, ALIASES["tn"])
    fp = _first_present(md, ALIASES["fp"])
    fn = _first_present(md, ALIASES["fn"])
    n_pred = _first_present(md, ALIASES["n_pred"])
    n_ref = _first_present(md, ALIASES["n_ref"])

    # derive Jaccard if missing
    if iou is None:
        if tp is not None and fp is not None and fn is not None:
            denom = tp + fp + fn
            iou = (tp / denom) if denom and denom > 0 else 0.0
        elif dice is not None and dice < 2.0:
            try:
                iou = dice / (2.0 - dice) if (2.0 - dice) != 0 else 0.0
            except Exception:
                iou = None

    # derive PPV/NPV if missing
    if ppv is None and tp is not None and fp is not None:
        denom = tp + fp
        ppv = (tp / denom) if denom and denom > 0 else 0.0
    if npv is None and tn is not None and fn is not None:
        denom = tn + fn
        npv = (tn / denom) if denom and denom > 0 else 0.0

    out.update(
        {
            "Dice": dice,
            "Jaccard": iou,  # canonical: Jaccard == IoU
            "PPV": ppv,
            "NPV": npv,
            "HD95": hd95,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_pred": n_pred,
            "n_ref": n_ref,
        }
    )

    # Keep any additional custom metrics present in md
    for k, v in md.items():
        if k in ALIASES["iou"]:
            continue  # drop IoU duplicates
        if k not in out:
            out[k] = v

    return out


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _std(vals: List[float], which: str) -> float:
    n = len(vals)
    if n <= 1:
        return 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n if which == "population" else (n - 1))
    return math.sqrt(var)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


# Keep only columns that have at least one numeric value across all rows
def is_numeric_col(col: str, rows: List[Dict[str, Any]]) -> bool:
    for r in rows:
        if to_float(r.get(col)) is not None:
            return True
    return False


def load_items(data: Any) -> List[Dict[str, Any]]:
    """Accept list or dict with common keys and return the list of per-case entries."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("results", "cases", "items", "per_case"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        if any(k in data for k in ("prediction_file", "reference_file", "metrics")):
            return [data]
        for v in data.values():  # last resort: first list of dicts found
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError("Unrecognized JSON structure for nnU-Net evaluation output.")


def _add_label_rows(
    case_rows: List[Dict[str, Any]],
    metrics_per_class: Dict[str, Any],
    case_id: str,
    pred: str,
    ref: str,
    label_map: Optional[Dict[int, str]],
) -> None:
    # numeric keys only (labels); skip 0 by convention
    for cls_k, m in metrics_per_class.items():
        try:
            cls_id = int(cls_k)
            class_type = "label"
            class_name = label_map.get(cls_id, "") if label_map else ""
        except ValueError:
            cls_id = ""
            class_type = "region"
            class_name = cls_k

        row = {
            "case_id": case_id,
            "class_type": class_type,
            "class_id": cls_id,
            "class_name": class_name,
            "pred_path": pred,
            "ref_path": ref,
        }
        if isinstance(m, dict):
            row.update({str(k): v for k, v in _norm_case_metrics(m).items()})
        case_rows.append(row)


def _add_region_rows(
    case_rows: List[Dict[str, Any]],
    regions_block: Any,
    case_id: str,
    pred: str,
    ref: str,
) -> None:
    """
    Regions live under metrics["regions"] as a dict of region_name -> metrics dict.
    We emit rows with class_type='region', class_id='', class_name=region_name.
    """
    if not isinstance(regions_block, dict):
        return
    for rname, m in regions_block.items():
        row = {
            "case_id": case_id,
            "class_type": "region",
            "class_id": "",  # regions are not numeric classes
            "class_name": rname,  # keep region key as name
            "pred_path": pred,
            "ref_path": ref,
        }
        if isinstance(m, dict):
            row.update({str(k): v for k, v in _norm_case_metrics(m).items()})
        case_rows.append(row)


def parse_rows(
    items: List[Dict[str, Any]], label_map: Optional[Dict[int, str]] = None
) -> List[Dict[str, Any]]:
    """
    Parse nnU-Net region-based JSONs that have both label metrics and nested regions.
    """
    rows: List[Dict[str, Any]] = []

    label_combo_map = {
        "(1, 2, 3)": "whole_tumor",
        "(2, 3)": "tumor_core",
        "3": "enhancing_tumor",
    }

    for it in items:
        pred = it.get("prediction_file") or ""
        ref = it.get("reference_file") or ""
        case_id = (
            it.get("case_id")
            or os.path.splitext(os.path.basename(pred or ref or ""))[0]
        )
        metrics_dict = it.get("metrics", {})
        if not isinstance(metrics_dict, dict):
            continue

        # --- label metrics ---
        for cls_key, mdict in metrics_dict.items():
            if cls_key == "regions":
                continue
            if not isinstance(mdict, dict):
                continue

            cname = label_combo_map.get(cls_key, str(cls_key))
            row = {
                "case_id": case_id,
                "class_type": "label",
                "class_id": cls_key,
                "class_name": cname,
                "pred_path": pred,
                "ref_path": ref,
            }
            normed = _norm_case_metrics(mdict)
            # drop IoU entirely
            for k in list(normed.keys()):
                if k.lower() in {"iou", "io u", "iou_score", "iou_mean"}:
                    normed.pop(k, None)
            row.update(normed)
            rows.append(row)

        # --- region metrics ---
        regions = metrics_dict.get("regions", {})
        if isinstance(regions, dict):
            for reg_name, reg_metrics in regions.items():
                row = {
                    "case_id": case_id,
                    "class_type": "region",
                    "class_id": "",
                    "class_name": reg_name,
                    "pred_path": pred,
                    "ref_path": ref,
                }
                if isinstance(reg_metrics, dict):
                    row.update(
                        {
                            k: float(v)
                            for k, v in reg_metrics.items()
                            if isinstance(v, (float, int))
                        }
                    )
                rows.append(row)

    if not rows:
        raise ValueError("No rows parsed from JSON.")
    return rows


def _round_inplace(row: Dict[str, Any], ndigits: Optional[int]) -> None:
    if ndigits is None:
        return
    for k, v in list(row.items()):
        # round floats only
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            row[k] = round(fv, ndigits)


def write_cases_csv(
    rows: List[Dict[str, Any]],
    out_path: Path,
    round_ndigits: Optional[int],
    rename_hd95_mm: bool,
) -> List[str]:
    """Write per-case/per-class-or-region CSV. Returns the list of metric-like columns included."""
    # union header
    header_keys = set()
    for r in rows:
        header_keys.update(r.keys())

    fixed = ["case_id", "class_type", "class_id", "class_name", "pred_path", "ref_path"]
    preferred = [
        "Dice",
        "Jaccard",
        "PPV",
        "NPV",
        "HD95",
        "tp",
        "tn",
        "fp",
        "fn",
        "n_pred",
        "n_ref",
    ]
    others = sorted([k for k in header_keys if k not in (fixed + preferred)])
    header = fixed + [k for k in preferred if k in header_keys] + others

    if rename_hd95_mm and "HD95" in header:
        header = [("HD95_mm" if c == "HD95" else c) for c in header]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            row = dict(r)
            if rename_hd95_mm and "HD95" in row:
                row["HD95_mm"] = row.pop("HD95")
            _round_inplace(row, round_ndigits)
            w.writerow({k: row.get(k, "") for k in header})

    # metric-like columns (everything except fixed identifiers)
    return [k for k in header if k not in fixed]


def _label_name(cid: Any, label_map: Optional[Dict[int, str]]) -> Optional[str]:
    try:
        return label_map.get(int(cid)) if label_map else None
    except Exception:
        return None


def write_summary_csv(
    rows: List[Dict[str, Any]],
    metric_cols: List[str],
    out_path: Path,
    label_map: Optional[Dict[int, str]],
    round_ndigits: Optional[int],
    std_type: str = "population",
    hd95_quantiles: Optional[str] = None,
):
    """
    Aggregate per (class_type, class_id/name):
      - Scores: mean/median/std (+ optional HD95 quantiles)
      - Counts: sums
    """

    # group rows by (class_type, key)
    def class_key(r):
        if r.get("class_type") == "region":
            return ("region", r.get("class_name", ""))
        # label: prefer numeric id
        cid = r.get("class_id")
        try:
            cid = int(cid)
        except Exception:
            pass
        return ("label", cid)

    by_group: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(class_key(r), []).append(r)

    # split columns
    score_cols: List[str] = []
    count_cols: List[str] = []
    skip_cols = {"class_id", "class_name", "class_type"}

    for c in metric_cols:
        if c in skip_cols:
            continue
        if c.lower() in COUNT_KEYS:
            count_cols.append(c)
        elif is_numeric_col(c, rows):
            score_cols.append(c)
        else:
            score_cols.append(c)

    # parse requested quantiles for HD95
    q_vals: List[float] = []
    if hd95_quantiles:
        for tok in hd95_quantiles.split(","):
            tok = tok.strip()
            if tok:
                try:
                    q = float(tok)
                    if 0 < q < 100:
                        q_vals.append(q)
                except Exception:
                    pass
        q_vals = sorted(set(q_vals))

    out_rows: List[Dict[str, Any]] = []

    for key, grp in sorted(by_group.items(), key=lambda kv: kv[0]):
        ctype, cid_or_name = key
        if ctype == "label":
            out: Dict[str, Any] = {
                "class_type": "label",
                "class_id": (cid_or_name if isinstance(cid_or_name, int) else ""),
                "class_name": _label_name(cid_or_name, label_map),  # None if not found
            }
        else:
            out: Dict[str, Any] = {
                "class_type": "region",
                "class_id": "",
                "class_name": cid_or_name,  # region key (e.g., whole_tumor)
            }

        # scores
        for c in score_cols:
            vals = [to_float(r.get(c)) for r in grp]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if vals:
                out[f"{c}_mean"] = sum(vals) / len(vals)
                out[f"{c}_median"] = _median(vals)
                out[f"{c}_std"] = _std(vals, std_type)
                if c.lower() == "hd95" and q_vals:
                    s = sorted(vals)
                    for q in q_vals:
                        k = int(round((q / 100.0) * (len(s) - 1)))
                        out[f"HD95_p{int(q)}"] = s[k]
            else:
                out[f"{c}_mean"] = out[f"{c}_median"] = out[f"{c}_std"] = ""

        # counts totals
        for c in count_cols:
            vals = [to_float(r.get(c)) for r in grp]
            vals = [int(v) for v in vals if v is not None]
            total = sum(vals) if vals else ""
            out[f"{c}_sum"] = total

        _round_inplace(out, round_ndigits)
        out_rows.append(out)

    # header
    score_headers = ["class_type", "class_id", "class_name"]
    for c in score_cols:
        score_headers += [f"{c}_mean", f"{c}_median", f"{c}_std"]
        if c.lower() == "hd95" and q_vals:
            score_headers += [f"HD95_p{int(q)}" for q in q_vals]
    for c in count_cols:
        score_headers.append(f"{c}_sum")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=score_headers)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in score_headers})


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Convert nnU-Net v2 evaluation JSON to CSV "
            "(adds/derives PPV/NPV/Jaccard; preserves counts; supports labels + regions)."
        )
    )
    ap.add_argument(
        "-i",
        "--in_fn",
        required=True,
        help="Path to evaluation JSON (e.g., summary_with_hd95.json or summary.json)",
    )
    ap.add_argument(
        "--out_cases_fn",
        type=Path,
        default=None,
        help="Output CSV for per-case/per-class-or-region (default: <base>_cases.csv)",
    )
    ap.add_argument(
        "--out_summary_fn",
        type=Path,
        default=None,
        help="Output CSV for per-class-or-region aggregates (default: <base>_summary.csv)",
    )
    ap.add_argument(
        "--counts_out_fn",
        type=Path,
        default=None,
        help="Optional separate CSV for counts totals.",
    )
    ap.add_argument(
        "--round",
        type=int,
        default=4,
        help="Round floats to N decimals in CSV outputs (default: 4).",
    )
    ap.add_argument(
        "--rename_hd95_mm",
        action="store_true",
        help="Rename HD95 -> HD95_mm in outputs (to make units explicit).",
    )
    ap.add_argument(
        "--std_type",
        default="population",
        choices=["population", "sample"],
        help="Std type for summary scores (default: population).",
    )
    ap.add_argument(
        "--hd95_quantiles",
        default=None,
        help="Comma-separated quantiles (e.g., '90,95') to add for HD95 as HD95_pXX.",
    )
    ap.add_argument(
        "--label_map",
        default=None,
        choices=["brats", None, "none"],
        help="Add a class_name column for labels (currently only 'brats' supported).",
    )
    ap.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Log file path (in addition to console).",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return ap.parse_args()


# -------------------------- main -------------------------- #
def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")

    # Prepare class-name map for labels if requested
    label_map = None
    if args.label_map and str(args.label_map).lower() in {"brats"}:
        label_map = {1: "necrotic", 2: "edema", 3: "enhancing"}

    in_path = Path(args.in_fn)
    logger.info(f"Input JSON: {in_path}")
    with open(in_path, "r") as f:
        data = json.load(f)

    items = load_items(data)
    rows = parse_rows(items, label_map)

    base = in_path.with_suffix("")  # drop .json
    out_cases = args.out_cases_fn or Path(str(base) + "_cases.csv")
    metric_cols = write_cases_csv(
        rows,
        out_cases,
        round_ndigits=args.round,
        rename_hd95_mm=args.rename_hd95_mm,
    )

    out_summary = args.out_summary_fn or Path(str(base) + "_summary.csv")
    write_summary_csv(
        rows,
        metric_cols,
        out_summary,
        label_map=label_map,
        round_ndigits=args.round,
        std_type=args.std_type,
        hd95_quantiles=args.hd95_quantiles,
    )

    logger.info("Wrote:")
    logger.info(f"  - {out_cases}")
    logger.info(f"  - {out_summary}")


if __name__ == "__main__":
    main()
