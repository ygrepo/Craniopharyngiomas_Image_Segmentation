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
  --granularity labels|regions|both \
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

# -------------------------- config -------------------------- #

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

# Keep any additional custom metrics present in md,
# but DROP any alias keys that map to canonical ones.
DROP_KEYS = set()
DROP_KEYS |= ALIASES["dice"] - {"Dice"}
DROP_KEYS |= ALIASES["iou"] - {"Jaccard"}
DROP_KEYS |= ALIASES["ppv"] - {"PPV"}
DROP_KEYS |= ALIASES["npv"] - {"NPV"}
DROP_KEYS |= (
    ALIASES["tp"]
    | ALIASES["tn"]
    | ALIASES["fp"]
    | ALIASES["fn"]
    | ALIASES["n_pred"]
    | ALIASES["n_ref"]
    | ALIASES["hd95"]  # drop 'HD95' (and its aliases) from source rows
)

IDENTIFIER_KEYS = {
    "case_id",
    "class_type",
    "class_id",
    "class_name",
    "pred_path",
    "ref_path",
}

# -------------------------- helpers -------------------------- #


def _first_present(d: Dict[str, Any], names: Sequence[str]) -> Optional[float]:
    """Return first matching key's float value, respecting alias priority order."""
    for name in names:
        if name in d:
            try:
                return float(d[name])
            except Exception:
                return None
    return None


def _norm_case_metrics(md: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Normalize a single (case, class/region) metrics dict to canonical keys:
      Dice, Jaccard, PPV, NPV, HD95_mm, tp, tn, fp, fn, n_pred, n_ref
    Also derive per-case (macro) metrics when counts exist:
      Recall, Specificity, BalancedAccuracy, Prevalence, PredPosRate, VolumetricBias
    """
    logger.debug(f"Normalizing metrics: {md}")
    out: Dict[str, Optional[float]] = {}

    # canonical scores
    dice = _first_present(md, ALIASES["dice"])
    iou = _first_present(md, ALIASES["iou"])
    ppv = _first_present(md, ALIASES["ppv"])
    npv = _first_present(md, ALIASES["npv"])
    hd95 = _first_present(md, ALIASES["hd95"])
    logger.debug(f"Scores: Dice={dice}, IoU={iou}, PPV={ppv}, NPV={npv}, HD95={hd95}")

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

    # ---- per-case (macro) derived metrics from counts ----
    recall = None
    specificity = None
    balanced_acc = None
    prevalence = None
    pred_pos_rate = None
    vol_bias = None
    total = None

    try:
        if all(x is not None for x in (tp, tn, fp, fn)):
            tp_f = float(tp)
            tn_f = float(tn)
            fp_f = float(fp)
            fn_f = float(fn)
            total = tp_f + tn_f + fp_f + fn_f
            if (tp_f + fn_f) > 0:
                recall = tp_f / (tp_f + fn_f)
            if (tn_f + fp_f) > 0:
                specificity = tn_f / (tn_f + fp_f)
            if recall is not None and specificity is not None:
                balanced_acc = 0.5 * (recall + specificity)
            if total > 0:
                prevalence = (tp_f + fn_f) / total
                pred_pos_rate = (tp_f + fp_f) / total
        if n_pred is not None and n_ref is not None:
            n_pred_f = float(n_pred)
            n_ref_f = float(n_ref)
            if n_ref_f > 0:
                vol_bias = (n_pred_f - n_ref_f) / n_ref_f
            elif n_ref_f == 0 and n_pred_f == 0:
                vol_bias = 0.0
            else:
                vol_bias = None
    except Exception:
        pass

    out.update(
        {
            "Dice": dice,
            "Jaccard": iou,  # canonical: Jaccard == IoU
            "PPV": ppv,
            "NPV": npv,
            "HD95_mm": hd95,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_pred": n_pred,
            "n_ref": n_ref,
            # per-case derived
            "Recall": recall,
            "Specificity": specificity,
            "BalancedAccuracy": balanced_acc,
            "Prevalence": prevalence,
            "PredPosRate": pred_pos_rate,
            "VolumetricBias": vol_bias,
        }
    )

    # Keep any additional custom metrics present in md
    for k, v in md.items():
        if k in DROP_KEYS:
            continue
        kl = str(k).lower()
        if kl in {
            "iou",
            "io u",
            "iou_score",
            "iou_mean",
            "hd95",
            "hausdorff95",
            "hausdorff_95",
            "hausdorff 95",
        }:
            continue
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


def is_numeric_col(col: str, rows: List[Dict[str, Any]]) -> bool:
    """Keep only columns that have at least one numeric value across all rows."""
    for r in rows:
        logger.debug(f"Row: {r}")
        if to_float(r.get(col)) is not None:
            return True
        else:
            logger.debug(f"Non-numeric value found in column {col}: {r.get(col)}")

    return False


def load_items(data: Any) -> List[Dict[str, Any]]:
    """Accept list or dict with common keys and return the list of per-case entries."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("results", "cases", "items", "per_case", "metric_per_case"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        if any(k in data for k in ("prediction_file", "reference_file", "metrics")):
            return [data]
        for v in data.values():  # last resort: first list of dicts found
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError("Unrecognized JSON structure for nnU-Net evaluation output.")


def _case_id_from_paths(pred_path: str, ref_path: str) -> str:
    def _stem(p: str) -> str:
        b = os.path.basename(p or "")
        if b.endswith(".nii.gz"):
            return b[:-7]
        return os.path.splitext(b)[0]

    s = _stem(pred_path) or _stem(ref_path)
    return s


# -------------------------- row builders (no mappings) -------------------------- #


def parse_rows_labels(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Emit rows for EVERY key under metrics (except the 'regions' block),
    using the raw key string as class_id; no mapping/renaming applied.
    """
    rows: List[Dict[str, Any]] = []
    for it in items:
        pred = it.get("prediction_file") or it.get("prediction") or it.get("pred") or ""
        ref = it.get("reference_file") or it.get("gt") or it.get("reference") or ""
        case_id = it.get("case_id") or _case_id_from_paths(pred, ref)

        metrics_dict = it.get("metrics", {}) or {}
        if not isinstance(metrics_dict, dict):
            continue

        for cls_key, mdict in metrics_dict.items():
            if cls_key == "regions" or not isinstance(mdict, dict):
                logger.debug(f"Skipping non-dict entry for key: {cls_key}")
                continue
            row = {
                "case_id": case_id,
                "class_type": "label",
                "class_id": str(cls_key),
                "class_name": "",  # intentionally empty — no mappings
                "pred_path": pred,
                "ref_path": ref,
            }
            normed = _norm_case_metrics(mdict)
            # Drop any IoU aliases that slipped through
            for k in list(normed.keys()):
                if k in IOU_ALIASES or str(k).lower() in {
                    "iou",
                    "io u",
                    "iou_score",
                    "iou_mean",
                }:
                    normed.pop(k, None)
            row.update(normed)
            rows.append(row)
    return rows


def parse_rows_regions(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Emit rows from metrics['regions'] ONLY, taking region keys exactly as present.
    """
    rows: List[Dict[str, Any]] = []
    for it in items:
        pred = it.get("prediction_file") or it.get("prediction") or it.get("pred") or ""
        ref = it.get("reference_file") or it.get("gt") or it.get("reference") or ""
        case_id = it.get("case_id") or _case_id_from_paths(pred, ref)

        metrics_dict = it.get("metrics", {}) or {}
        if not isinstance(metrics_dict, dict):
            continue

        reg_block = metrics_dict.get("regions")
        if not isinstance(reg_block, dict):
            continue

        for rname, mdict in reg_block.items():
            if not isinstance(mdict, dict):
                continue
            row = {
                "case_id": case_id,
                "class_type": "region",
                "class_id": "",  # regions have no numeric id here
                "class_name": str(rname),  # use raw key as-is
                "pred_path": pred,
                "ref_path": ref,
            }
            row.update(_norm_case_metrics(mdict))
            rows.append(row)
    return rows


# -------------------------- writers -------------------------- #


def _round_inplace(row: Dict[str, Any], ndigits: Optional[int]) -> None:
    if ndigits is None:
        return
    for k, v in list(row.items()):
        if k in IDENTIFIER_KEYS:
            continue
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
    header_keys = set()
    for r in rows:
        header_keys.update(r.keys())
    logger.info(f"Header keys: {header_keys}")

    fixed = ["case_id", "class_type", "class_id", "class_name", "pred_path", "ref_path"]
    preferred = [
        # scores first
        "Dice",
        "Jaccard",
        "PPV",
        "NPV",
        "Recall",
        "Specificity",
        "BalancedAccuracy",
        "Prevalence",
        "PredPosRate",
        "VolumetricBias",
        "HD95_mm",
        # counts
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

    def _uniq(seq):
        return list(dict.fromkeys(seq))

    header = _uniq(header)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            row = dict(r)
            if rename_hd95_mm and "HD95" in row:
                row["HD95_mm"] = row.pop("HD95")
            _round_inplace(row, round_ndigits)
            logger.debug(f"Row: {row}")
            w.writerow({k: row.get(k, "") for k in header})
    logger.debug(f"Header: {header}")
    metric_cols = _uniq([k for k in header if k not in fixed])
    # metric-like columns (everything except fixed identifiers)
    return metric_cols


# --- stable sort: labels alpha by raw id; regions alpha by name ---
def _group_sort_key(item):
    (ctype, ident) = item[0]
    return (0 if ctype == "label" else 1, str(ident).lower())


# --- group key ---
def class_key(r):
    ctype = r.get("class_type")
    if ctype == "region":
        return ("region", str(r.get("class_name", "")))
    # labels: use raw class_id string as-is
    return ("label", str(r.get("class_id", "")))


def _quantile(vals: List[float], q: float) -> float:
    """Linear interpolation quantile in [0,1]. Assumes vals is non-empty."""
    s = sorted(vals)
    n = len(s)
    if n == 1:
        return s[0]
    # position in [0, n-1]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _mean_ci95(vals: List[float], std_type: str) -> Tuple[float, float, float]:
    """
    Return (mean, ci_lo, ci_hi) using normal approx (z=1.96).
    std_type: 'population' uses population SD, 'sample' uses sample SD.
    """
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(vals) / n
    if n == 1:
        return mean, mean, mean
    # SD consistent with your _std()
    if std_type == "population":
        m = mean
        var = sum((v - m) ** 2 for v in vals) / n
    else:
        m = mean
        var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    z = 1.96
    half = z * (sd / math.sqrt(n)) if n > 0 else float("nan")
    return mean, mean - half, mean + half


def write_summary_csv(
    rows: List[Dict[str, Any]],
    metric_cols: List[str],
    out_path: Path,
    round_ndigits: Optional[int],
    std_type: str = "population",
    hd95_quantiles: Optional[str] = None,
):
    """
    Aggregate per (class_type, class_id/name):

      Macro block (per-case):
        - For each score column: mean / median / std
        - This includes derived per-case metrics (Recall, Specificity, BalancedAccuracy,
          Prevalence, PredPosRate, VolumetricBias) if present in rows.

      Micro block (from sums):
        - micro_dice, micro_jaccard, micro_precision (PPV), micro_npv, micro_recall,
          micro_specificity, micro_balanced_accuracy, micro_prevalence, micro_pred_pos_rate,
          micro_volumetric_bias

      Counts block:
        - tp_sum, tn_sum, fp_sum, fn_sum, n_pred_sum, n_ref_sum
    """
    logger.info(f"Writing summary to {out_path}")
    logger.info(f"Metric columns: {metric_cols}")
    logger.info(f"HD95 quantiles: {hd95_quantiles}")
    logger.info(f"Std type: {std_type}")
    logger.info(f"Rounding: {round_ndigits}")
    logger.info(f"Rows: {len(rows)}")

    by_group: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(class_key(r), []).append(r)

    # --- split columns ---
    score_cols: List[str] = []
    count_cols: List[str] = []
    skip_cols = {"class_id", "class_name", "class_type"}
    metric_cols = list(dict.fromkeys(metric_cols))
    for c in metric_cols:
        if c in skip_cols:
            continue
        if c.lower() in COUNT_KEYS:
            count_cols.append(c)
        elif is_numeric_col(c, rows):
            score_cols.append(c)

    logger.info(f"Score columns: {score_cols}")
    logger.info(f"Count columns: {count_cols}")
    # --- quantiles for HD95 ---
    q_vals: List[float] = []
    if hd95_quantiles:
        parts = []
        for tok in hd95_quantiles.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                q = float(tok)
                if 0 < q < 100:
                    parts.append(q)
            except Exception:
                pass
        q_vals = sorted(set(parts))

    out_rows: List[Dict[str, Any]] = []
    for key, grp in sorted(by_group.items(), key=_group_sort_key):
        ctype, ident = key
        if ctype == "label":
            class_id = ident
            class_name = ""
        else:
            class_id = ""
            class_name = ident

        out: Dict[str, Any] = {
            "class_type": ctype,
            "class_id": class_id,
            "class_name": class_name,
        }

        # ---------- MACRO: per-case stats ----------
        for c in score_cols:
            vals = [to_float(r.get(c)) for r in grp]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if vals:
                out[f"{c}_mean"] = sum(vals) / len(vals)
                out[f"{c}_median"] = _median(vals)
                out[f"{c}_std"] = _std(vals, std_type)
                # IQR (Q1, Q3, IQR)
                q1 = _quantile(vals, 0.25)
                q3 = _quantile(vals, 0.75)
                out[f"{c}_q1"] = q1
                out[f"{c}_q3"] = q3
                out[f"{c}_iqr"] = q3 - q1
                if c.lower() == "hd95" and q_vals:
                    s = sorted(vals)
                    for q in q_vals:
                        k = int(round((q / 100.0) * (len(s) - 1)))
                        out[f"HD95_p{int(q)}"] = s[k]
            else:
                # empty: fill blanks to keep headers consistent
                out[f"{c}_mean"] = out[f"{c}_ci95_lo"] = out[f"{c}_ci95_hi"] = ""
                out[f"{c}_median"] = out[f"{c}_std"] = ""
                out[f"{c}_q1"] = out[f"{c}_q3"] = out[f"{c}_iqr"] = ""

        # ---------- COUNT SUMS ----------
        sums = {}
        for c in ("tp", "tn", "fp", "fn", "n_pred", "n_ref"):
            vals = [to_float(r.get(c)) for r in grp]
            vals = [int(v) for v in vals if v is not None]
            sums[c] = sum(vals) if vals else 0
            out[f"{c}_sum"] = sums[c] if vals else ""

        _round_inplace(out, round_ndigits)
        out_rows.append(out)

    # ---------- HEADER GROUPING ----------
    # identifiers
    headers = ["class_type", "class_id", "class_name"]

    # Order macro metrics (only those present)
    macro_order = [
        "Dice",
        "Jaccard",
        "PPV",
        "NPV",
        "Recall",
        "Specificity",
        "BalancedAccuracy",
        "Prevalence",
        "PredPosRate",
        "VolumetricBias",
        "HD95_mm",
    ]
    macro_present = [m for m in macro_order if m in score_cols]
    # group: each metric -> mean/median/std
    for c in macro_present:
        headers += [
            f"{c}_mean",
            f"{c}_ci95_lo",
            f"{c}_ci95_hi",
            f"{c}_median",
            f"{c}_std",
            f"{c}_q1",
            f"{c}_q3",
            f"{c}_iqr",
        ]
        if c == "HD95" and q_vals:
            headers += [f"HD95_p{int(q)}" for q in q_vals]

    # counts block at the end
    for c in ("tp", "tn", "fp", "fn", "n_pred", "n_ref"):
        col = f"{c}_sum"
        if any(col in r for r in out_rows):
            headers.append(col)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in headers})


def _group_key(r):
    ctype = r.get("class_type")
    if ctype == "region":
        return ("region", str(r.get("class_name", "")))
    return ("label", str(r.get("class_id", "")))


def write_micro_csv(
    rows: List[Dict[str, Any]],
    out_path: Path,
    round_ndigits: Optional[int],
):
    """
    Compute micro metrics (from sums) per (class_type, class_id/name) and write to CSV.

    Columns:
      - identifiers: class_type, class_id, class_name
      - micro metrics: micro_dice, micro_jaccard, micro_precision, micro_recall,
                       micro_specificity, micro_balanced_accuracy, micro_prevalence,
                       micro_pred_pos_rate, micro_npv, micro_volumetric_bias
      - counts: tp_sum, tn_sum, fp_sum, fn_sum, n_pred_sum, n_ref_sum
    """
    # Group rows
    by_group: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(_group_key(r), []).append(r)

    out_rows: List[Dict[str, Any]] = []

    def safe_div(n, d):
        try:
            return (n / d) if d and d > 0 else ""
        except Exception:
            return ""

    for key, grp in sorted(
        by_group.items(),
        key=lambda it: (0 if it[0][0] == "label" else 1, str(it[0][1]).lower()),
    ):
        ctype, ident = key
        if ctype == "label":
            class_id, class_name = ident, ""
        else:
            class_id, class_name = "", ident

        # Sum counts
        sums = {}
        for c in ("tp", "tn", "fp", "fn", "n_pred", "n_ref"):
            vals = [to_float(r.get(c)) for r in grp]
            vals = [int(v) for v in vals if v is not None]
            sums[c] = sum(vals) if vals else 0

        tp_s, tn_s, fp_s, fn_s = (sums["tp"], sums["tn"], sums["fp"], sums["fn"])
        n_pred_s, n_ref_s = (sums["n_pred"], sums["n_ref"])
        total = tp_s + tn_s + fp_s + fn_s

        micro_precision = safe_div(tp_s, tp_s + fp_s)
        micro_npv = safe_div(tn_s, tn_s + fn_s)
        micro_recall = safe_div(tp_s, tp_s + fn_s)
        micro_specificity = safe_div(tn_s, tn_s + fp_s)
        micro_bal_acc = (
            0.5 * (micro_recall + micro_specificity)
            if isinstance(micro_recall, float) and isinstance(micro_specificity, float)
            else ""
        )
        micro_prevalence = safe_div(tp_s + fn_s, total)
        micro_pred_pos_rate = safe_div(tp_s + fp_s, total)
        micro_dice = safe_div(2 * tp_s, (2 * tp_s + fp_s + fn_s))
        micro_jaccard = safe_div(tp_s, (tp_s + fp_s + fn_s))
        micro_vol_bias = safe_div((n_pred_s - n_ref_s), n_ref_s)

        out = {
            "class_type": ctype,
            "class_id": class_id,
            "class_name": class_name,
            "micro_dice": micro_dice,
            "micro_jaccard": micro_jaccard,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_specificity": micro_specificity,
            "micro_balanced_accuracy": micro_bal_acc,
            "micro_prevalence": micro_prevalence,
            "micro_pred_pos_rate": micro_pred_pos_rate,
            "micro_npv": micro_npv,
            "micro_volumetric_bias": micro_vol_bias,
            "tp_sum": tp_s,
            "tn_sum": tn_s,
            "fp_sum": fp_s,
            "fn_sum": fn_s,
            "n_pred_sum": n_pred_s,
            "n_ref_sum": n_ref_s,
        }

        _round_inplace(out, round_ndigits)
        out_rows.append(out)

    headers = [
        "class_type",
        "class_id",
        "class_name",
        "micro_dice",
        "micro_jaccard",
        "micro_precision",
        "micro_recall",
        "micro_specificity",
        "micro_balanced_accuracy",
        "micro_prevalence",
        "micro_pred_pos_rate",
        "micro_npv",
        "micro_volumetric_bias",
        "tp_sum",
        "tn_sum",
        "fp_sum",
        "fn_sum",
        "n_pred_sum",
        "n_ref_sum",
    ]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in headers})


# -------------------------- CLI / main -------------------------- #


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Convert nnU-Net v2 evaluation JSON to CSV "
            "(adds/derives PPV/NPV/Jaccard; preserves counts; supports labels/regions without mappings)."
        )
    )
    ap.add_argument(
        "-i",
        "--in_fn",
        required=True,
        help="Path to evaluation JSON (e.g., summary_with_hd95.json or summary.json)",
    )
    ap.add_argument(
        "--granularity",
        choices=["labels", "regions", "both", "auto"],
        default="auto",
        help="Emit rows from label keys, regions block, both, or auto-detect (regions if present, else labels).",
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
        "--out_micro_fn",
        type=Path,
        default=None,
        help="Output CSV for dataset-level (micro) metrics per class/region (default: <base>_micro.csv)",
    )
    ap.add_argument(
        "--counts_out_fn",
        type=Path,
        default=None,
        help="Optional separate CSV for counts totals (not implemented).",
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


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)
    logger.info(f"Args: {args}")

    in_path = Path(args.in_fn)
    logger.info(f"Input JSON: {in_path}")
    with open(in_path, "r") as f:
        data = json.load(f)

    items = load_items(data)

    # choose rows according to granularity (no mappings)
    if args.granularity == "auto":
        # Prefer regions if any 'regions' blocks are present; otherwise use labels
        has_regions = any(
            isinstance((it.get("metrics", {}) or {}).get("regions"), dict)
            for it in items
        )
        gran = "regions" if has_regions else "labels"
    else:
        gran = args.granularity

    rows: List[Dict[str, Any]] = []
    if gran in ("labels", "both"):
        rows.extend(parse_rows_labels(items))
    if gran in ("regions", "both"):
        rows.extend(parse_rows_regions(items))

    if not rows:
        raise SystemExit(
            "No rows parsed from JSON (check that your file has 'metrics' blocks)."
        )

    base = in_path.with_suffix("")
    out_cases = args.out_cases_fn or Path(str(base) + "_cases.csv")
    metric_cols = write_cases_csv(
        rows, out_cases, round_ndigits=args.round, rename_hd95_mm=args.rename_hd95_mm
    )

    out_summary = args.out_summary_fn or Path(str(base) + "_summary.csv")
    write_summary_csv(
        rows,
        metric_cols,
        out_summary,
        round_ndigits=args.round,
        std_type=args.std_type,
        hd95_quantiles=args.hd95_quantiles,
    )

    out_micro = args.out_micro_fn or Path(str(base) + "_micro.csv")
    write_micro_csv(
        rows,
        out_micro,
        round_ndigits=args.round,
    )

    logger.info("Wrote:")
    logger.info(f"  - {out_cases}")
    logger.info(f"  - {out_summary}")
    logger.info(f"  - {out_micro}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
