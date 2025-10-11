#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import statistics as stats
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

# Keys we treat as counts (case-level, per class)
COUNT_KEYS = {"tp", "tn", "fp", "fn", "n_pred", "n_ref"}

# Aliases we will recognize in the metrics dicts coming from nnUNet eval JSONs
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


def _first_present(d: Dict[str, Any], names: set) -> Optional[float]:
    for k, v in d.items():
        if k in names:
            try:
                return float(v)
            except Exception:
                return None
    return None


def _norm_case_metrics(md: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Normalize a single case-class metric dict to a standard set of keys and
    derive PPV/NPV/Jaccard when possible.
    Returns numeric values (floats) or None when unavailable.
    """
    out: Dict[str, Optional[float]] = {}

    # Pull canonical scores if present
    dice = _first_present(md, ALIASES["dice"])
    iou = _first_present(md, ALIASES["iou"])
    ppv = _first_present(md, ALIASES["ppv"])
    npv = _first_present(md, ALIASES["npv"])
    hd95 = _first_present(md, ALIASES["hd95"])

    # Pull counts if present
    tp = _first_present(md, ALIASES["tp"])
    tn = _first_present(md, ALIASES["tn"])
    fp = _first_present(md, ALIASES["fp"])
    fn = _first_present(md, ALIASES["fn"])

    # Derive Jaccard if missing and counts exist
    if iou is None:
        if tp is not None and fp is not None and fn is not None:
            denom = tp + fp + fn
            iou = (tp / denom) if denom > 0 else 0.0
        elif dice is not None and dice < 2.0:
            # J = D / (2 - D)
            iou = dice / (2.0 - dice)

    # Derive PPV/NPV if missing and counts present
    if ppv is None and tp is not None and fp is not None:
        denom = tp + fp
        ppv = (tp / denom) if denom > 0 else 0.0
    if npv is None and tn is not None and fn is not None:
        denom = tn + fn
        npv = (tn / denom) if denom > 0 else 0.0

    out["Dice"] = dice
    out["Jaccard"] = iou  # canonical name in our CSV
    out["PPV"] = ppv
    out["NPV"] = npv
    out["HD95"] = hd95

    # Also keep raw counts if they were present (so summaries can sum them)
    for canon, names in (
        ("tp", "tp"),
        ("tn", "tn"),
        ("fp", "fp"),
        ("fn", "fn"),
        ("n_pred", "n_pred"),
        ("n_ref", "n_ref"),
    ):
        val = _first_present(md, ALIASES[names])
        if val is not None:
            out[canon] = val

    # Keep original keys too (so you don't lose any custom metrics in your JSONs)
    for k, v in md.items():
        if k not in out:
            out[k] = v
    return out


def load_items(data: Any) -> List[Dict[str, Any]]:
    # Accept list or dict with common keys
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


def parse_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for it in items:
        pred = it.get("prediction_file") or it.get("prediction") or it.get("pred") or ""
        ref = it.get("reference_file") or it.get("gt") or it.get("reference") or ""
        case_id = (
            it.get("case_id")
            or os.path.splitext(os.path.basename(pred or ref or ""))[0]
        )
        metrics_per_class = it.get("metrics", {})
        if not isinstance(metrics_per_class, dict):
            continue
        for cls_k, m in metrics_per_class.items():
            try:
                cls_id = int(cls_k)
            except Exception:
                cls_id = cls_k
            row = {
                "case_id": case_id,
                "class_id": cls_id,
                "pred_path": pred,
                "ref_path": ref,
            }
            if isinstance(m, dict):
                norm = _norm_case_metrics(m)
                row.update({str(k): v for k, v in norm.items()})
            rows.append(row)
    if not rows:
        raise ValueError("No rows parsed from JSON.")
    return rows


def write_cases_csv(rows: List[Dict[str, Any]], out_path: str) -> List[str]:
    # union of all keys across rows to keep every metric column
    header_keys = set()
    for r in rows:
        header_keys.update(r.keys())
    # deterministic order: fixed cols first, then canonical metrics, then the rest sorted
    fixed = ["case_id", "class_id", "pred_path", "ref_path"]
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
    others = sorted([k for k in header_keys if k not in fixed + preferred])
    header = fixed + preferred + others
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    # return only metric-like columns (exclude the fixed identifiers)
    return [k for k in header if k not in fixed]


def write_summary_csv(
    rows: List[Dict[str, Any]], metric_cols: List[str], out_path: str
):
    # group rows by class_id
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_class.setdefault(str(r["class_id"]), []).append(r)

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # split columns: counts vs scores
    score_cols: List[str] = []
    count_cols: List[str] = []
    for c in metric_cols:
        lc = c.lower()
        if lc in COUNT_KEYS:
            count_cols.append(c)
        else:
            score_cols.append(c)

    summ_rows = []
    for cls, cls_rows in sorted(by_class.items(), key=lambda kv: kv[0]):
        out: Dict[str, Any] = {"class_id": cls}
        # scores: mean/median/std (skip None)
        for c in score_cols:
            vals = [to_float(r.get(c)) for r in cls_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                out[f"{c}_mean"] = stats.mean(vals)
                out[f"{c}_median"] = stats.median(vals)
                out[f"{c}_std"] = stats.pstdev(vals) if len(vals) > 1 else 0.0
            else:
                out[f"{c}_mean"] = out[f"{c}_median"] = out[f"{c}_std"] = ""
        # counts: sum
        for c in count_cols:
            vals = [to_float(r.get(c)) for r in cls_rows]
            vals = [int(v) for v in vals if v is not None]
            out[f"{c}_sum"] = sum(vals) if vals else ""
        summ_rows.append(out)

    header = (
        ["class_id"]
        + [f"{c}_mean" for c in score_cols]
        + [f"{c}_median" for c in score_cols]
        + [f"{c}_std" for c in score_cols]
        + [f"{c}_sum" for c in count_cols]
    )
    logger.info(f"Save summary to: {out_path}")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in summ_rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(
        description="Convert nnUNet v2 evaluation JSON to CSV (adds PPV/NPV/HD95/Jaccard; derives from counts/Dice when needed)."
    )
    ap.add_argument(
        "-i",
        "--input",
        default="nnUNet_results/Dataset501_BraTS2017_4ch/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/predictions/validation/summary_with_hd95.json",
        required=False,
        help="Path to evaluation JSON (e.g., summary_with_hd95.json)",
    )
    ap.add_argument(
        "--out-cases", default=None, help="Output CSV path for per-case/per-class rows"
    )
    ap.add_argument(
        "--out-summary", default=None, help="Output CSV path for per-class aggregates"
    )
    args = ap.parse_args()

    setup_logging(None, "INFO")

    base = os.path.splitext(args.input)[0]
    out_cases = args.out_cases or (base + "_cases.csv")
    out_summary = args.out_summary or (base + "_summary.csv")

    with open(args.input, "r") as f:
        data = json.load(f)

    items = load_items(data)
    rows = parse_rows(items)
    metric_cols = write_cases_csv(rows, out_cases)
    write_summary_csv(rows, metric_cols, out_summary)

    logger.info("[ok] wrote:")
    logger.info(f" {out_cases}")
    logger.info(f" {out_summary}")


if __name__ == "__main__":
    main()
