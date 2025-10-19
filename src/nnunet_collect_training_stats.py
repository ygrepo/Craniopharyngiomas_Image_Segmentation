#!/usr/bin/env python3
"""
export_training_logs_to_csv.py

Parse nnU-Net v2 training_log_*.txt files and export per-epoch metrics to CSV.

Collected per-epoch fields:
- epoch
- lr
- train_loss
- val_loss
- DICE_1, DICE_2, DICE_3
- epoch_time_sec (if present)
- src_file (log file that last updated the epoch)

Usage:
  python export_training_logs_to_csv.py -i /path/to/training_log_2025_10_6_21_28_23.txt
"""

import argparse
import csv
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict


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
re_num_generic = re.compile(NUM)


def _extract_dices(body: str):
    # Try strict pattern first
    vals = [m.group(1) for m in re_dtype_num.finditer(body)]
    if len(vals) < 3:
        vals = [m.group(1) for m in re_num_generic.finditer(body)]
    vals = (vals + ["", "", ""])[:3]
    return vals[0], vals[1], vals[2]


def parse_one_file(path: str, rows: Dict[int, Dict[str, Any]]):
    """
    Robust parser that:
      - Tracks the current epoch explicitly (no ep=max(rows) heuristic).
      - Works with/without timestamp prefixes.
      - Skips blank/spacer lines safely.
      - Extracts Dice triplets from np.floatXX(...) or raw numbers.
      - Updates ts_last/src_file consistently on every matched line.
    """
    current_epoch = None
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
                            "Necrotic_Dice": "",
                            "Edema_Dice": "",
                            "Enhancing_Dice": "",
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

                # --- Dice triplet ---
                m = re_dice_line.search(line) or re_dice_line_nt.search(s)
                if m:
                    d1, d2, d3 = _extract_dices(m.group("body"))
                    r["Necrotic_Dice"] = d1
                    r["Edema_Dice"] = d2
                    r["Enhancing_Dice"] = d3
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


# def parse_one_file(path: str, rows: Dict[int, Dict[str, Any]]):
#     with open(path, "r", errors="replace") as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue

#             # epoch start
#             m = re_epoch.search(line) or re_epoch_nt.search(s)
#             if m:
#                 epoch = int(m.group("epoch"))
#                 ts = m.groupdict().get("ts", "")
#                 rows.setdefault(
#                     epoch,
#                     {
#                         "epoch": epoch,
#                         "ts_first": ts,
#                         "ts_last": ts,
#                         "lr": "",
#                         "train_loss": "",
#                         "val_loss": "",
#                         "epoch_time_sec": "",
#                         "Necrotic_Dice": "",
#                         "Edema_Dice": "",
#                         "Enhancing_Dice": "",
#                         "EMA_DICE": "",
#                         "src_file": os.path.basename(path),
#                     },
#                 )
#                 continue

#             if not rows:
#                 continue  # ignore lines before first epoch

#             ep = max(rows.keys())
#             r = rows[ep]

#             for regex, key in [
#                 (re_lr, "lr"),
#                 (re_lr_nt, "lr"),
#                 (re_trainloss, "train_loss"),
#                 (re_trainloss_nt, "train_loss"),
#                 (re_valloss, "val_loss"),
#                 (re_valloss_nt, "val_loss"),
#             ]:
#                 m = regex.search(line)
#                 if m:
#                     r[key] = m.group(list(m.groupdict().keys())[-1])
#                     r["ts_last"] = m.groupdict().get("ts", r["ts_last"])
#                     r["src_file"] = os.path.basename(path)
#                     break

#             m = re_dice_line.search(line) or re_dice_line_nt.search(s)
#             if m:
#                 d1, d2, d3 = _extract_dices(m.group("body"))
#                 r.update(
#                     {
#                         "Necrotic_Dice": d1,
#                         "Edema_Dice": d2,
#                         "Enhancing_Dice": d3,
#                         "ts_last": m.groupdict().get("ts", r["ts_last"]),
#                         "src_file": os.path.basename(path),
#                     }
#                 )
#                 continue

#             m = re_epoch_time.search(line) or re_epoch_time_nt.search(s)
#             if m:
#                 r["epoch_time_sec"] = m.group("sec")
#                 r["ts_last"] = m.groupdict().get("ts", r["ts_last"])
#                 r["src_file"] = os.path.basename(path)
#                 continue

#             m = re_ema.search(line) or re_ema_nt.search(s)
#             if m:
#                 r["EMA_DICE"] = m.group("ema")
#                 r["ts_last"] = m.groupdict().get("ts", r["ts_last"])
#                 r["src_file"] = os.path.basename(path)
#                 continue


def main():
    ap = argparse.ArgumentParser(description="Export nnU-Net v2 logs to CSV.")
    ap.add_argument("-i", "--input_dir", required=True, help="Log file or directory.")
    ap.add_argument("-o", "--output_fn", default=None)
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
        logger.error("[error] no log files found.", file=sys.stderr)
        sys.exit(1)

    rows = {}
    for fp in files:
        logger.info(f"Parsing {fp}...")
        parse_one_file(fp, rows)

    if not rows:
        logger.error("[error] no epochs parsed.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "epoch",
        "epoch_time_sec",
        "ts_first",
        "ts_last",
        "lr",
        "train_loss",
        "val_loss",
        "Necrotic_Dice",
        "Edema_Dice",
        "Enhancing_Dice",
        "EMA_DICE",
        "src_file",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ep in sorted(rows):
            w.writerow({k: rows[ep].get(k, "") for k in fieldnames})
    logger.info(f"Wrote {out_csv} with {len(rows)} epochs.")


if __name__ == "__main__":
    main()
