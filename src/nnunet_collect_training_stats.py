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
- ts_first, ts_last (timestamps of first/last line seen for that epoch)
- src_file (log file that last updated the epoch)

Usage:
  python export_training_logs_to_csv.py -i /path/to/fold_0/ -o /path/to/metrics.csv
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

TS_PREFIX = r"^\s*(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+):\s*"
NUM = r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)"

re_epoch = re.compile(TS_PREFIX + r"Epoch\s+(?P<epoch>\d+)\s*$", re.IGNORECASE)
re_lr = re.compile(
    TS_PREFIX + r"(?:Current\s+)?learning\s+rate:\s*(?P<lr>" + NUM + r")\s*$",
    re.IGNORECASE,
)
re_trainloss = re.compile(
    TS_PREFIX + r"train_loss\s+(?P<loss>" + NUM + r")\s*$", re.IGNORECASE
)
re_valloss = re.compile(
    TS_PREFIX + r"val_loss\s+(?P<loss>" + NUM + r")\s*$", re.IGNORECASE
)
re_dice_line = re.compile(
    TS_PREFIX + r"Pseudo\s+dice\s*\[(?P<body>.+?)\]\s*$", re.IGNORECASE
)
re_epoch_time = re.compile(
    TS_PREFIX + r"Epoch\s+time:\s*(?P<sec>" + NUM + r")\s*s\b", re.IGNORECASE
)

# Fallbacks without timestamp prefix
re_epoch_nt = re.compile(r"^\s*Epoch\s+(?P<epoch>\d+)\s*$", re.IGNORECASE)
re_lr_nt = re.compile(
    r"^\s*(?:Current\s+)?learning\s+rate:\s*(?P<lr>" + NUM + r")\s*$", re.IGNORECASE
)
re_trainloss_nt = re.compile(
    r"^\s*train_loss\s+(?P<loss>" + NUM + r")\s*$", re.IGNORECASE
)
re_valloss_nt = re.compile(r"^\s*val_loss\s+(?P<loss>" + NUM + r")\s*$", re.IGNORECASE)
re_dice_line_nt = re.compile(
    r"^\s*Pseudo\s+dice\s*\[(?P<body>.+?)\]\s*$", re.IGNORECASE
)
re_epoch_time_nt = re.compile(
    r"^\s*Epoch\s+time:\s*(?P<sec>" + NUM + r")\s*s\b", re.IGNORECASE
)

re_any_num = re.compile(NUM)


def parse_one_file(path: str, rows: Dict[int, Dict[str, Any]]):
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()

                m = re_epoch.search(line) or re_epoch_nt.search(s)
                if m:
                    epoch = int(m.group("epoch"))
                    ts = m.groupdict().get("ts") if "ts" in m.groupdict() else ""
                    r = rows.setdefault(
                        epoch,
                        {
                            "epoch": epoch,
                            "lr": "",
                            "train_loss": "",
                            "val_loss": "",
                            "DICE_1": "",
                            "DICE_2": "",
                            "DICE_3": "",
                            "epoch_time_sec": "",
                            "ts_first": ts or "",
                            "ts_last": ts or "",
                            "src_file": os.path.basename(path),
                        },
                    )
                    if not r["ts_first"] and ts:
                        r["ts_first"] = ts
                    if ts:
                        r["ts_last"] = ts
                    r["src_file"] = os.path.basename(path)
                    continue

                m = re_lr.search(line) or re_lr_nt.search(s)
                if m and rows:
                    ts = m.groupdict().get("ts") if "ts" in m.groupdict() else ""
                    ep = max(rows.keys())
                    rows[ep]["lr"] = m.group("lr")
                    if ts:
                        rows[ep]["ts_last"] = ts
                        rows[ep]["src_file"] = os.path.basename(path)
                    continue

                m = re_trainloss.search(line) or re_trainloss_nt.search(s)
                if m and rows:
                    ts = m.groupdict().get("ts") if "ts" in m.groupdict() else ""
                    ep = max(rows.keys())
                    rows[ep]["train_loss"] = m.group("loss")
                    if ts:
                        rows[ep]["ts_last"] = ts
                        rows[ep]["src_file"] = os.path.basename(path)
                    continue

                m = re_valloss.search(line) or re_valloss_nt.search(s)
                if m and rows:
                    ts = m.groupdict().get("ts") if "ts" in m.groupdict() else ""
                    ep = max(rows.keys())
                    rows[ep]["val_loss"] = m.group("loss")
                    if ts:
                        rows[ep]["ts_last"] = ts
                        rows[ep]["src_file"] = os.path.basename(path)
                    continue

                m = re_dice_line.search(line) or re_dice_line_nt.search(s)
                if m and rows:
                    body = m.group("body")
                    nums = re_any_num.findall(body)
                    d1 = nums[0] if len(nums) >= 1 else ""
                    d2 = nums[1] if len(nums) >= 2 else ""
                    d3 = nums[2] if len(nums) >= 3 else ""
                    ep = max(rows.keys())
                    rows[ep]["DICE_1"] = d1
                    rows[ep]["DICE_2"] = d2
                    rows[ep]["DICE_3"] = d3
                    ts = m.groupdict().get("ts") if "ts" in m.groupdict() else ""
                    if ts:
                        rows[ep]["ts_last"] = ts
                        rows[ep]["src_file"] = os.path.basename(path)
                    continue

                m = re_epoch_time.search(line) or re_epoch_time_nt.search(s)
                if m and rows:
                    ts = m.groupdict().get("ts") if "ts" in m.groupdict() else ""
                    ep = max(rows.keys())
                    rows[ep]["epoch_time_sec"] = m.group("sec")
                    if ts:
                        rows[ep]["ts_last"] = ts
                        rows[ep]["src_file"] = os.path.basename(path)
                    continue
    except FileNotFoundError:
        logger.warning(f"[warn] not found: {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate nnU-Net training logs into CSV (per-epoch metrics)."
    )
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="Directory (reads training_log*.txt) or single log file.",
    )
    ap.add_argument("-o", "--output", default=None, help="Output CSV path.")
    ap.add_argument(
        "--pattern", default="training_log*.txt", help="Glob if input is a directory."
    )
    args = ap.parse_args()

    setup_logging(None, "INFO")
    logger.info(f"Args: {args}")
    in_path = args.input
    if os.path.isdir(in_path):
        files = sorted(glob(os.path.join(in_path, args.pattern)))
        default_out = os.path.join(in_path, "training_metrics.csv")
    else:
        files = [in_path]
        default_out = os.path.splitext(in_path)[0] + "_metrics.csv"

    out_csv = args.output or default_out
    if not files:
        logger.error("[error] no log files found.", file=sys.stderr)
        sys.exit(1)

    rows = {}
    for fp in files:
        logger.info(f"Parsing {fp}...")
        parse_one_file(fp, rows)

    if not rows:
        logger.error("[error] parsed 0 epochs.", file=sys.stderr)
        sys.exit(2)

    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "DICE_1",
        "DICE_2",
        "DICE_3",
        "epoch_time_sec",
        "ts_first",
        "ts_last",
        "src_file",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ep in sorted(rows.keys()):
            r = rows[ep]
            w.writerow({k: r.get(k, "") for k in fieldnames})

    logger.info(f"Wrote {out_csv} with {len(rows)} epochs.")


if __name__ == "__main__":
    main()
