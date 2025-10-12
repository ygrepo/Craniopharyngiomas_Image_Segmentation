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

# Keys treated as counts (per case, per class)
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
    Normalize a single (case, class) metrics dict to canonical keys:
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
            # J = D / (2 - D)
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


def parse_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn the eval items into long-format rows, 1 row per (case, class)."""
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
                row.update({str(k): v for k, v in _norm_case_metrics(m).items()})
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
    label_map: Optional[Dict[int, str]] = None,
) -> List[str]:
    """Write per-case/per-class CSV. Returns the list of metric-like columns included."""
    # union header
    header_keys = set()
    for r in rows:
        header_keys.update(r.keys())

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
    # ensure preferred exist if present in data
    others = sorted([k for k in header_keys if k not in (fixed + preferred)])
    header = fixed + [k for k in preferred if k in header_keys] + others

    # optional rename for HD95 columns
    if rename_hd95_mm and "HD95" in header:
        header = [("HD95_mm" if c == "HD95" else c) for c in header]

    # insert class_name column right after class_id if available
    if label_map:
        if "class_name" not in header:
            idx = header.index("class_id") + 1 if "class_id" in header else 1
            header = header[:idx] + ["class_name"] + header[idx:]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            row = dict(r)
            if rename_hd95_mm and "HD95" in row:
                row["HD95_mm"] = row.pop("HD95")
            if label_map and "class_id" in row:
                try:
                    cid = int(row["class_id"])
                    row["class_name"] = label_map.get(cid, "")
                except Exception:
                    row["class_name"] = ""
            _round_inplace(row, round_ndigits)

            w.writerow({k: row.get(k, "") for k in header})

    # metric-like columns (everything except fixed identifiers)
    return [k for k in header if k not in fixed and k not in ("class_name",)]


def write_summary_csv(
    rows: List[Dict[str, Any]],
    metric_cols: List[str],
    out_path: Path,
    round_ndigits: Optional[int],
    std_type: str = "population",
    hd95_quantiles: Optional[str] = None,
    label_map: Optional[Dict[int, str]] = None,
    counts_out_fn: Optional[Path] = None,
):
    """
    Aggregate per class:
      - Scores: mean/median/std (+ optional HD95 quantiles)
      - Counts: sums
    Optionally emit counts-only CSV (counts_out_fn).
    """

    # group rows by class_id
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_class.setdefault(str(r["class_id"]), []).append(r)

    # split columns
    score_cols: List[str] = []
    count_cols: List[str] = []
    skip_cols = {"class_id", "class_name"}

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

    # build rows
    out_rows: List[Dict[str, Any]] = []
    counts_rows: List[Dict[str, Any]] = []

    for cls, cls_rows in sorted(
        by_class.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0]
    ):
        cid = int(cls) if str(cls).isdigit() else cls
        class_name = label_map.get(cid) if label_map else None

        out: Dict[str, Any] = {"class_id": cid}
        if class_name:
            out["class_name"] = class_name

        # scores
        for c in score_cols:
            vals = [to_float(r.get(c)) for r in cls_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                out[f"{c}_mean"] = sum(vals) / len(vals)
                out[f"{c}_median"] = _median(vals)
                out[f"{c}_std"] = _std(vals, std_type)
                # optional HD95 quantiles
                if c.lower() == "hd95" and q_vals:
                    s = sorted(vals)
                    for q in q_vals:
                        k = int(round((q / 100.0) * (len(s) - 1)))
                        out[f"HD95_p{int(q)}"] = s[k]
            else:
                out[f"{c}_mean"] = out[f"{c}_median"] = out[f"{c}_std"] = ""

        # counts totals (stay in this summary unless counts_out_fn is used)
        count_totals = {"class_id": cid}
        if class_name:
            count_totals["class_name"] = class_name
        for c in count_cols:
            vals = [to_float(r.get(c)) for r in cls_rows]
            vals = [int(v) for v in vals if v is not None]
            total = sum(vals) if vals else ""
            if counts_out_fn:
                count_totals[f"{c}_sum"] = total
            else:
                out[f"{c}_sum"] = total

        _round_inplace(out, round_ndigits)
        out_rows.append(out)
        if counts_out_fn:
            _round_inplace(count_totals, round_ndigits)
            counts_rows.append(count_totals)

    # header for scores summary
    score_headers = ["class_id"]
    if label_map:
        score_headers.append("class_name")
    for c in score_cols:
        if c.lower() == "iou":
            continue
        score_headers += [f"{c}_mean", f"{c}_median", f"{c}_std"]
        if c.lower() == "hd95" and q_vals:
            score_headers += [f"HD95_p{int(q)}" for q in q_vals]
    if not counts_out_fn:
        # include totals inline
        for c in count_cols:
            score_headers.append(f"{c}_sum")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=score_headers)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in score_headers})

    # optional separate counts-only CSV
    if counts_out_fn:
        count_headers = ["class_id"]
        if label_map:
            count_headers.append("class_name")
        count_headers += [f"{c}_sum" for c in count_cols]
        with open(counts_out_fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=count_headers)
            w.writeheader()
            for r in counts_rows:
                w.writerow({k: r.get(k, "") for k in count_headers})


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Convert nnU-Net v2 evaluation JSON to CSV "
            "(adds/derives PPV/NPV/Jaccard; preserves counts; optional rounding)."
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
        help="Output CSV for per-case/per-class",
    )
    ap.add_argument(
        "--out_summary_fn",
        type=Path,
        default=None,
        help="Output CSV for per-class aggregates",
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
        help="Round floats to N decimals in CSV outputs (default: no rounding).",
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
        choices=["brats", None],
        help="Add a class_name column using a known map (e.g., 'brats').",
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

    args = ap.parse_args()
    return args


# -------------------------- main -------------------------- #
def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")
    # Prepare class-name map if requested
    label_map = None
    if args.label_map == "brats":
        label_map = {1: "necrotic", 2: "edema", 3: "enhancing"}

    out_cases = args.out_cases_fn.resolve()
    out_summary = args.out_summary_fn.resolve()

    logger.info(f"Input JSON: {args.in_fn}")
    with open(args.in_fn, "r") as f:
        data = json.load(f)

    items = load_items(data)
    rows = parse_rows(items)

    metric_cols = write_cases_csv(
        rows,
        out_cases,
        round_ndigits=args.round,
        rename_hd95_mm=args.rename_hd95_mm,
        label_map=label_map,
    )

    write_summary_csv(
        rows,
        metric_cols,
        out_summary,
        round_ndigits=args.round,
        std_type=args.std_type,
        hd95_quantiles=args.hd95_quantiles,
        label_map=label_map,
        counts_out_fn=args.counts_out_fn.resolve() if args.counts_out_fn else None,
    )

    logger.info("Wrote:")
    logger.info(f"  - {out_cases}")
    logger.info(f"  - {out_summary}")


if __name__ == "__main__":
    main()
