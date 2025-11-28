import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
)

logger = get_logger(__name__)


def save_image(img: np.ndarray, path: Path):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    logger.info(f"Saving to {path}")
    plt.savefig(path, dpi=150)
    plt.close()


#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def get_args():
    ap = argparse.ArgumentParser(
        description="Plot CV AUC/F1 vs C from binary LASSO CV metrics CSV."
    )
    ap.add_argument("--csv_path", type=Path, required=True)
    ap.add_argument(
        "--output_png",
        type=Path,
        default=None,
    )
    return ap.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.csv_path)

    if args.output_png is None:
        args.output_png = args.csv_path.with_suffix("_plot.png")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xscale("log")

    # Plot F1
    ax1.plot(df["C"], df["mean_f1"], marker="o", label="Mean F1")
    ax1.fill_between(
        df["C"],
        df["mean_f1"] - df["std_f1"],
        df["mean_f1"] + df["std_f1"],
        alpha=0.2,
    )
    ax1.set_xlabel("C (log scale)")
    ax1.set_ylabel("F1")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # Plot AUC on second axis
    ax2 = ax1.twinx()
    ax2.plot(df["C"], df["mean_auc"], marker="s", linestyle="--", label="Mean AUC")
    ax2.fill_between(
        df["C"],
        df["mean_auc"] - df["std_auc"],
        df["mean_auc"] + df["std_auc"],
        alpha=0.2,
    )
    ax2.set_ylabel("AUC")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Binary LASSO: 5-fold CV results")
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
