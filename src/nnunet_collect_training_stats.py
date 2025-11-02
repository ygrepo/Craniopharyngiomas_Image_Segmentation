#!/usr/bin/env python3
"""
nnunet_collect_training_stats.py

Parse nnU-Net v2 training_log_*.txt files and export per-epoch metrics to CSV.

Collected per-epoch fields:
- epoch
- lr
- train_loss
- val_loss
- Dynamic DICE classes (auto-detected)
- epoch_time_sec (if present)
- src_file (log file that last updated the epoch)

Usage:
  python nnunet_collect_training_stats.py -i /path/to/training_log_2025_10_6_21_28_23.txt
  python nnunet_collect_training_stats.py -i /path/to/logs/ --class_names Background Tumor Necrosis
"""

import argparse
import csv
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import get_logger, setup_logging

logger = get_logger(__name__)


NUM = r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)"
TS_PREFIX = r"^\s*(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+):\s*"

re_epoch = re.compile(TS_PREFIX + r"Epoch\s+(?P<epoch>\d+)\b", re.IGNORECASE)
re_lr = re.compile(
    TS_PREFIX + r"(?:Current\s+)?learning\s+rate:\s*(?P<lr>" + NUM + ")", re.IGNORECASE
)
re_trainloss = re.compile(
    TS_PREFIX + r"train_loss\s+(?P<loss>" + NUM + ")", re.IGNORECASE
)
re_valloss = re.compile(TS_PREFIX + r"val_loss\s+(?P<loss>" + NUM + ")", re.IGNORECASE)
re_dice_line = re.compile(
    TS_PREFIX + r"Pseudo\s+dice\s*\[(?P<body>.+?)\]", re.IGNORECASE
)
re_epoch_time = re.compile(
    TS_PREFIX + r"Epoch\s+time:\s*(?P<sec>" + NUM + ")", re.IGNORECASE
)
re_ema = re.compile(
    TS_PREFIX + r".*EMA\s+pseudo\s+Dice:\s*(?P<ema>" + NUM + ")", re.IGNORECASE
)

# Fallback (no timestamp)
re_epoch_nt = re.compile(r"^\s*Epoch\s+(?P<epoch>\d+)\b", re.IGNORECASE)
re_lr_nt = re.compile(
    r"(?:Current\s+)?learning\s+rate:\s*(?P<lr>" + NUM + ")", re.IGNORECASE
)
re_trainloss_nt = re.compile(r"train_loss\s+(?P<loss>" + NUM + ")", re.IGNORECASE)
re_valloss_nt = re.compile(r"val_loss\s+(?P<loss>" + NUM + ")", re.IGNORECASE)
re_dice_line_nt = re.compile(r"Pseudo\s+dice\s*\[(?P<body>.+?)\]", re.IGNORECASE)
re_epoch_time_nt = re.compile(r"Epoch\s+time:\s*(?P<sec>" + NUM + ")", re.IGNORECASE)
re_ema_nt = re.compile(r"EMA\s+pseudo\s+Dice:\s*(?P<ema>" + NUM + ")", re.IGNORECASE)

re_dtype_num = re.compile(r"np\.\w+\(\s*" + NUM + r"\s*\)", re.IGNORECASE)
re_tensor_num = re.compile(r"tensor\(\s*" + NUM + r"\s*\)", re.IGNORECASE)
re_num_generic = re.compile(NUM)


def _extract_dices_dynamic(body: str) -> List[str]:
    """
    Extract all dice values from a string, handling various formats:
    - Raw numbers: "0.123, 0.456, 0.789"
    - numpy format: "np.float64(0.123), np.float32(0.456), 0.789"
    - tensor format: "tensor(0.123), tensor(0.456)"
    - Mixed formats
    """
    dice_values = []

    # First try to extract from structured formats (numpy, tensor)
    for match in re_dtype_num.finditer(body):
        dice_values.append(match.group(1))

    for match in re_tensor_num.finditer(body):
        dice_values.append(match.group(1))

    # If we didn't find enough structured values, fall back to generic number extraction
    if len(dice_values) == 0:
        for match in re_num_generic.finditer(body):
            dice_values.append(match.group(1))

    return dice_values


def _get_default_class_names(num_classes: int) -> List[str]:
    """Generate default class names based on common segmentation tasks."""
    if num_classes == 2:
        return ["Background", "Foreground"]
    elif num_classes == 3:
        return ["Necrotic", "Edema", "Enhancing"]  # Original nnU-Net BraTS style
    elif num_classes == 4:
        return ["Background", "Necrotic", "Edema", "Enhancing"]
    elif num_classes == 5:
        return ["Background", "Necrotic", "Edema", "Enhancing", "Tumor"]
    else:
        # For larger numbers, use generic naming
        names = ["Background"] if num_classes > 2 else []
        for i in range(1 if num_classes > 2 else 0, num_classes):
            names.append(f"Class_{i}")
        return names


def _update_dice_values(
    row_dict: Dict[str, Any],
    dice_values: List[str],
    class_names: List[str] = None,
    detected_classes: Set[str] = None,
):
    """Update row dictionary with dice values for detected classes."""
    if not dice_values:
        return

    num_classes = len(dice_values)

    if class_names and len(class_names) >= num_classes:
        # Use provided class names
        current_class_names = class_names[:num_classes]
    else:
        # Generate default class names
        current_class_names = _get_default_class_names(num_classes)
        if class_names:
            # Override with provided names where available
            for i, name in enumerate(class_names):
                if i < len(current_class_names):
                    current_class_names[i] = name

    # Update row with dice values
    for i, (class_name, dice_val) in enumerate(zip(current_class_names, dice_values)):
        field_name = f"{class_name}_Dice"
        row_dict[field_name] = dice_val
        if detected_classes is not None:
            detected_classes.add(field_name)


def _normalize_class_fields(
    rows: Dict[int, Dict[str, Any]], detected_classes: Set[str]
):
    """Ensure all epochs have the same class fields, filling missing ones with empty strings."""
    if not rows or not detected_classes:
        return

    # Add missing fields to all rows
    for row in rows.values():
        for field in detected_classes:
            if field not in row:
                row[field] = ""


def parse_one_file(
    path: str,
    rows: Dict[int, Dict[str, Any]],
    class_names: List[str] = None,
    detected_classes: Set[str] = None,
):
    """
    Robust parser that:
      - Tracks the current epoch explicitly (no ep=max(rows) heuristic).
      - Works with/without timestamp prefixes.
      - Skips blank/spacer lines safely.
      - Extracts Dice values dynamically for any number of classes.
      - Updates ts_last/src_file consistently on every matched line.
    """
    current_epoch = None
    if detected_classes is None:
        detected_classes = set()

    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    # skip empty lines
                    continue

                # --- Epoch header (with or without timestamp) ---
                m = re_epoch.search(line) or re_epoch_nt.search(s)
                if m:
                    current_epoch = int(m.group("epoch"))
                    ts = m.groupdict().get("ts", "")
                    r = rows.get(current_epoch)
                    if r is None:
                        rows[current_epoch] = {
                            "epoch": current_epoch,
                            "ts_first": ts,
                            "ts_last": ts,
                            "lr": "",
                            "train_loss": "",
                            "val_loss": "",
                            "epoch_time_sec": "",
                            "EMA_DICE": "",
                            "src_file": os.path.basename(path),
                        }
                    else:
                        # epoch already seen in another file pass; update last seen info
                        if not r.get("ts_first") and ts:
                            r["ts_first"] = ts
                        if ts:
                            r["ts_last"] = ts
                        r["src_file"] = os.path.basename(path)
                    continue

                # Ignore any metric lines before the first epoch
                if current_epoch is None:
                    continue

                r = rows[current_epoch]

                # --- Learning rate ---
                m = re_lr.search(line) or re_lr_nt.search(s)
                if m:
                    r["lr"] = m.group("lr")
                    ts = m.groupdict().get("ts", r["ts_last"])
                    if ts:
                        r["ts_last"] = ts
                    r["src_file"] = os.path.basename(path)
                    continue

                # --- Train loss ---
                m = re_trainloss.search(line) or re_trainloss_nt.search(s)
                if m:
                    r["train_loss"] = m.group("loss")
                    ts = m.groupdict().get("ts", r["ts_last"])
                    if ts:
                        r["ts_last"] = ts
                    r["src_file"] = os.path.basename(path)
                    continue

                # --- Val loss ---
                m = re_valloss.search(line) or re_valloss_nt.search(s)
                if m:
                    r["val_loss"] = m.group("loss")
                    ts = m.groupdict().get("ts", r["ts_last"])
                    if ts:
                        r["ts_last"] = ts
                    r["src_file"] = os.path.basename(path)
                    continue

                # --- Dynamic Dice parsing ---
                m = re_dice_line.search(line) or re_dice_line_nt.search(s)
                if m:
                    dice_values = _extract_dices_dynamic(m.group("body"))
                    _update_dice_values(r, dice_values, class_names, detected_classes)
                    ts = m.groupdict().get("ts", r["ts_last"])
                    if ts:
                        r["ts_last"] = ts
                    r["src_file"] = os.path.basename(path)
                    continue

                # --- Epoch time ---
                m = re_epoch_time.search(line) or re_epoch_time_nt.search(s)
                if m:
                    r["epoch_time_sec"] = m.group("sec")
                    ts = m.groupdict().get("ts", r["ts_last"])
                    if ts:
                        r["ts_last"] = ts
                    r["src_file"] = os.path.basename(path)
                    continue

                # --- EMA pseudo Dice (with/without "Yayy! New best ...") ---
                m = re_ema.search(line) or re_ema_nt.search(s)
                if m:
                    r["EMA_DICE"] = m.group("ema")
                    ts = m.groupdict().get("ts", r["ts_last"])
                    if ts:
                        r["ts_last"] = ts
                    r["src_file"] = os.path.basename(path)
                    continue

    except FileNotFoundError:
        logger.warning(f"[warn] not found: {path}")

    return detected_classes


def main():
    ap = argparse.ArgumentParser(description="Export nnU-Net v2 logs to CSV.")
    ap.add_argument("-i", "--input_dir", required=True, help="Log file or directory.")
    ap.add_argument("-o", "--output_fn", default=None)
    ap.add_argument(
        "--class_names",
        nargs="*",
        default=None,
        help="Custom class names (e.g., --class_names Background Tumor Necrosis Edema)",
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
    ap.add_argument(
        "--pattern", default="training_log*.txt", help="Glob if input is a directory."
    )
    args = ap.parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)

    logger.info(f"Args: {args}")

    in_path = args.input_dir
    if os.path.isdir(in_path):
        files = sorted(glob(os.path.join(in_path, args.pattern)))
        default_out = os.path.join(in_path, "training_metrics.csv")
    else:
        files = [in_path]
        default_out = os.path.splitext(in_path)[0] + "_metrics.csv"

    out_csv = args.output_fn or default_out
    if not files:
        logger.error("[error] no log files found.")
        sys.exit(1)

    rows = {}
    detected_classes = set()

    for fp in files:
        logger.info(f"Parsing {fp}...")
        detected_classes.update(
            parse_one_file(fp, rows, args.class_names, detected_classes)
        )

    if not rows:
        logger.error("[error] no epochs parsed.")
        sys.exit(1)

    # Normalize all rows to have the same fields
    _normalize_class_fields(rows, detected_classes)

    # Build dynamic fieldnames
    base_fieldnames = [
        "epoch",
        "epoch_time_sec",
        "ts_first",
        "ts_last",
        "lr",
        "train_loss",
        "val_loss",
    ]

    # Preserve order based on class_names if provided, otherwise use detected order
    if args.class_names:
        # Use the order from command line arguments
        dice_fieldnames = []
        for class_name in args.class_names:
            field_name = f"{class_name}_Dice"
            if field_name in detected_classes:
                dice_fieldnames.append(field_name)

        # Add any detected classes not in the provided list (sorted)
        remaining_classes = sorted(
            [
                f
                for f in detected_classes
                if f.endswith("_Dice") and f not in dice_fieldnames
            ]
        )
        dice_fieldnames.extend(remaining_classes)
    else:
        # If no class names provided, try to preserve natural order from first occurrence
        # For now, fall back to sorted order (you could enhance this to track first occurrence order)
        dice_fieldnames = sorted([f for f in detected_classes if f.endswith("_Dice")])

    end_fieldnames = [
        "EMA_DICE",
        "src_file",
    ]

    fieldnames = base_fieldnames + dice_fieldnames + end_fieldnames

    logger.info(f"Detected {len(dice_fieldnames)} dice classes: {dice_fieldnames}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ep in sorted(rows):
            w.writerow({k: rows[ep].get(k, "") for k in fieldnames})

    logger.info(
        f"Wrote {out_csv} with {len(rows)} epochs and {len(dice_fieldnames)} dice classes."
    )


if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# export_training_logs_to_csv.py

# Parse nnU-Net v2 training_log_*.txt files and export per-epoch metrics to CSV.

# Collected per-epoch fields:
# - epoch
# - lr
# - train_loss
# - val_loss
# - DICE_1, DICE_2, DICE_3
# - epoch_time_sec (if present)
# - src_file (log file that last updated the epoch)

# Usage:
#   python export_training_logs_to_csv.py -i /path/to/training_log_2025_10_6_21_28_23.txt
# """

# import argparse
# import csv
# import os
# import re
# import sys
# from glob import glob
# from pathlib import Path
# from typing import Any, Dict


# REPO_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(REPO_ROOT))
# from src.util import get_logger, setup_logging

# logger = get_logger(__name__)


# NUM = r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)"
# TS_PREFIX = r"^\s*(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+):\s*"

# re_epoch = re.compile(TS_PREFIX + r"Epoch\s+(?P<epoch>\d+)\b", re.IGNORECASE)
# re_lr = re.compile(
#     TS_PREFIX + r"(?:Current\s+)?learning\s+rate:\s*(?P<lr>" + NUM + ")", re.IGNORECASE
# )
# re_trainloss = re.compile(
#     TS_PREFIX + r"train_loss\s+(?P<loss>" + NUM + ")", re.IGNORECASE
# )
# re_valloss = re.compile(TS_PREFIX + r"val_loss\s+(?P<loss>" + NUM + ")", re.IGNORECASE)
# re_dice_line = re.compile(
#     TS_PREFIX + r"Pseudo\s+dice\s*\[(?P<body>.+?)\]", re.IGNORECASE
# )
# re_epoch_time = re.compile(
#     TS_PREFIX + r"Epoch\s+time:\s*(?P<sec>" + NUM + ")", re.IGNORECASE
# )
# re_ema = re.compile(
#     TS_PREFIX + r".*EMA\s+pseudo\s+Dice:\s*(?P<ema>" + NUM + ")", re.IGNORECASE
# )

# # Fallback (no timestamp)
# re_epoch_nt = re.compile(r"^\s*Epoch\s+(?P<epoch>\d+)\b", re.IGNORECASE)
# re_lr_nt = re.compile(
#     r"(?:Current\s+)?learning\s+rate:\s*(?P<lr>" + NUM + ")", re.IGNORECASE
# )
# re_trainloss_nt = re.compile(r"train_loss\s+(?P<loss>" + NUM + ")", re.IGNORECASE)
# re_valloss_nt = re.compile(r"val_loss\s+(?P<loss>" + NUM + ")", re.IGNORECASE)
# re_dice_line_nt = re.compile(r"Pseudo\s+dice\s*\[(?P<body>.+?)\]", re.IGNORECASE)
# re_epoch_time_nt = re.compile(r"Epoch\s+time:\s*(?P<sec>" + NUM + ")", re.IGNORECASE)
# re_ema_nt = re.compile(r"EMA\s+pseudo\s+Dice:\s*(?P<ema>" + NUM + ")", re.IGNORECASE)

# re_dtype_num = re.compile(r"np\.\w+\(\s*" + NUM + r"\s*\)", re.IGNORECASE)
# re_num_generic = re.compile(NUM)


# def _extract_dices(body: str):
#     # Try strict pattern first
#     vals = [m.group(1) for m in re_dtype_num.finditer(body)]
#     if len(vals) < 3:
#         vals = [m.group(1) for m in re_num_generic.finditer(body)]
#     vals = (vals + ["", "", ""])[:3]
#     return vals[0], vals[1], vals[2]


# def parse_one_file(path: str, rows: Dict[int, Dict[str, Any]]):
#     """
#     Robust parser that:
#       - Tracks the current epoch explicitly (no ep=max(rows) heuristic).
#       - Works with/without timestamp prefixes.
#       - Skips blank/spacer lines safely.
#       - Extracts Dice triplets from np.floatXX(...) or raw numbers.
#       - Updates ts_last/src_file consistently on every matched line.
#     """
#     current_epoch = None
#     try:
#         with open(path, "r", errors="replace") as f:
#             for line in f:
#                 s = line.strip()
#                 if not s:
#                     # skip empty lines
#                     continue

#                 # --- Epoch header (with or without timestamp) ---
#                 m = re_epoch.search(line) or re_epoch_nt.search(s)
#                 if m:
#                     current_epoch = int(m.group("epoch"))
#                     ts = m.groupdict().get("ts", "")
#                     r = rows.get(current_epoch)
#                     if r is None:
#                         rows[current_epoch] = {
#                             "epoch": current_epoch,
#                             "ts_first": ts,
#                             "ts_last": ts,
#                             "lr": "",
#                             "train_loss": "",
#                             "val_loss": "",
#                             "epoch_time_sec": "",
#                             "Necrotic_Dice": "",
#                             "Edema_Dice": "",
#                             "Enhancing_Dice": "",
#                             "EMA_DICE": "",
#                             "src_file": os.path.basename(path),
#                         }
#                     else:
#                         # epoch already seen in another file pass; update last seen info
#                         if not r.get("ts_first") and ts:
#                             r["ts_first"] = ts
#                         if ts:
#                             r["ts_last"] = ts
#                         r["src_file"] = os.path.basename(path)
#                     continue

#                 # Ignore any metric lines before the first epoch
#                 if current_epoch is None:
#                     continue

#                 r = rows[current_epoch]

#                 # --- Learning rate ---
#                 m = re_lr.search(line) or re_lr_nt.search(s)
#                 if m:
#                     r["lr"] = m.group("lr")
#                     ts = m.groupdict().get("ts", r["ts_last"])
#                     if ts:
#                         r["ts_last"] = ts
#                     r["src_file"] = os.path.basename(path)
#                     continue

#                 # --- Train loss ---
#                 m = re_trainloss.search(line) or re_trainloss_nt.search(s)
#                 if m:
#                     r["train_loss"] = m.group("loss")
#                     ts = m.groupdict().get("ts", r["ts_last"])
#                     if ts:
#                         r["ts_last"] = ts
#                     r["src_file"] = os.path.basename(path)
#                     continue

#                 # --- Val loss ---
#                 m = re_valloss.search(line) or re_valloss_nt.search(s)
#                 if m:
#                     r["val_loss"] = m.group("loss")
#                     ts = m.groupdict().get("ts", r["ts_last"])
#                     if ts:
#                         r["ts_last"] = ts
#                     r["src_file"] = os.path.basename(path)
#                     continue

#                 # --- Dice triplet ---
#                 m = re_dice_line.search(line) or re_dice_line_nt.search(s)
#                 if m:
#                     d1, d2, d3 = _extract_dices(m.group("body"))
#                     r["Necrotic_Dice"] = d1
#                     r["Edema_Dice"] = d2
#                     r["Enhancing_Dice"] = d3
#                     ts = m.groupdict().get("ts", r["ts_last"])
#                     if ts:
#                         r["ts_last"] = ts
#                     r["src_file"] = os.path.basename(path)
#                     continue

#                 # --- Epoch time ---
#                 m = re_epoch_time.search(line) or re_epoch_time_nt.search(s)
#                 if m:
#                     r["epoch_time_sec"] = m.group("sec")
#                     ts = m.groupdict().get("ts", r["ts_last"])
#                     if ts:
#                         r["ts_last"] = ts
#                     r["src_file"] = os.path.basename(path)
#                     continue

#                 # --- EMA pseudo Dice (with/without "Yayy! New best ...") ---
#                 m = re_ema.search(line) or re_ema_nt.search(s)
#                 if m:
#                     r["EMA_DICE"] = m.group("ema")
#                     ts = m.groupdict().get("ts", r["ts_last"])
#                     if ts:
#                         r["ts_last"] = ts
#                     r["src_file"] = os.path.basename(path)
#                     continue
#     except FileNotFoundError:
#         logger.warning(f"[warn] not found: {path}")


# def main():
#     ap = argparse.ArgumentParser(description="Export nnU-Net v2 logs to CSV.")
#     ap.add_argument("-i", "--input_dir", required=True, help="Log file or directory.")
#     ap.add_argument("-o", "--output_fn", default=None)
#     ap.add_argument(
#         "--log_level",
#         choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
#         default="INFO",
#         help="Logging level.",
#     )
#     ap.add_argument(
#         "--log_file",
#         type=Path,
#         default=None,
#         help="Log file path (in addition to console).",
#     )
#     ap.add_argument(
#         "--pattern", default="training_log*.txt", help="Glob if input is a directory."
#     )
#     args = ap.parse_args()
#     setup_logging(Path(args.log_file) if args.log_file else None, args.log_level)

#     logger.info(f"Args: {args}")

#     in_path = args.input_dir
#     if os.path.isdir(in_path):
#         files = sorted(glob(os.path.join(in_path, args.pattern)))
#         default_out = os.path.join(in_path, "training_metrics.csv")
#     else:
#         files = [in_path]
#         default_out = os.path.splitext(in_path)[0] + "_metrics.csv"

#     out_csv = args.output_fn or default_out
#     if not files:
#         logger.error("[error] no log files found.", file=sys.stderr)
#         sys.exit(1)

#     rows = {}
#     for fp in files:
#         logger.info(f"Parsing {fp}...")
#         parse_one_file(fp, rows)

#     if not rows:
#         logger.error("[error] no epochs parsed.", file=sys.stderr)
#         sys.exit(1)

#     fieldnames = [
#         "epoch",
#         "epoch_time_sec",
#         "ts_first",
#         "ts_last",
#         "lr",
#         "train_loss",
#         "val_loss",
#         "Necrotic_Dice",
#         "Edema_Dice",
#         "Enhancing_Dice",
#         "EMA_DICE",
#         "src_file",
#     ]
#     with open(out_csv, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for ep in sorted(rows):
#             w.writerow({k: rows[ep].get(k, "") for k in fieldnames})
#     logger.info(f"Wrote {out_csv} with {len(rows)} epochs.")


# if __name__ == "__main__":
#     main()
