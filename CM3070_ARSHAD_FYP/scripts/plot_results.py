#!/usr/bin/env python3
"""Generate publication-ready plots from experiment CSVs.

Usage:
  python scripts/plot_results.py            # pick latest CSV in runs/
  python scripts/plot_results.py --input runs/experiments_2025-09-19.csv
  python scripts/plot_results.py --dry-run  # verify script environment without data

This script prefers pandas, matplotlib, and seaborn. If they are missing it
prints a helpful message and exits cleanly.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path


def check_imports():
    missing = []
    try:
        pass  # type: ignore
    except Exception:
        missing.append("pandas")
    try:
        pass  # type: ignore
    except Exception:
        missing.append("matplotlib")
    try:
        pass  # type: ignore
    except Exception:
        missing.append("seaborn")
    return missing


def find_latest_csv(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


def make_plots(csv_path: Path, out_dir: Path):
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    df = pd.read_csv(csv_path)

    # Basic sanity checks
    if df.empty:
        print(f"CSV {csv_path} is empty — nothing to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure columns exist
    for col in ("rescued", "ttr_median"):
        if col not in df.columns:
            raise RuntimeError(f"Required column '{col}' not found in CSV")

    # Boxplot: ttr_median grouped by rescued (0/1)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["rescued"].astype(str), y=df["ttr_median"], palette="Set2")
    plt.xlabel("rescued")
    plt.ylabel("ttr_median (steps)")
    plt.title("TTR distribution grouped by rescued")
    f1 = out_dir / "box_ttr_by_rescued.png"
    plt.tight_layout()
    plt.savefig(f1, dpi=200)
    plt.close()

    # Bar: mean ± std for rescued proportion and ttr_median
    stats = {
        "rescued_prop": (df["rescued"].mean(), df["rescued"].std()),
        "ttr_median": (df["ttr_median"].mean(), df["ttr_median"].std()),
    }

    labels = ["rescued_prop", "ttr_median"]
    means = [stats[name][0] for name in labels]
    errs = [stats[name][1] for name in labels]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.bar(labels, means, yerr=errs, capsize=6, color=["#4c72b0", "#55a868"]) 
    ax.set_ylabel("Value")
    ax.set_title("Mean ± std: rescued proportion and ttr_median")
    f2 = out_dir / "mean_std_stats.png"
    plt.tight_layout()
    plt.savefig(f2, dpi=200)
    plt.close()

    # Histogram of ttr_median (only for runs where rescue happened)
    rescued_df = df[df["rescued"] == 1]
    if not rescued_df.empty:
        plt.figure(figsize=(6, 4))
        sns.histplot(rescued_df["ttr_median"].dropna(), bins=20, kde=True, color="#7a5195")
        plt.xlabel("ttr_median")
        plt.title("Histogram of ttr_median (rescued runs)")
        f3 = out_dir / "hist_ttr_rescued.png"
        plt.tight_layout()
        plt.savefig(f3, dpi=200)
        plt.close()

    # Save a small JSON summary
    summary = {
        "n_runs": int(len(df)),
        "rescued_mean": float(df["rescued"].mean()),
        "ttr_median_mean": float(df["ttr_median"].mean()),
        "ttr_median_std": float(df["ttr_median"].std()),
        "csv_source": str(csv_path.name),
    }
    import json

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved plots to {out_dir} and summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", help="Path to experiments CSV. If omitted, picks latest in runs/.")
    ap.add_argument("--out", "-o", help="Output directory (default: runs/plots_<ts>)")
    ap.add_argument("--dry-run", action="store_true", help="Check environment and exit without data processing")
    args = ap.parse_args()

    missing = check_imports()
    if missing:
        print("Missing required libraries:", ", ".join(missing))
        print("Install them with: pip install pandas matplotlib seaborn")
        if args.dry_run:
            print("Dry-run: exiting (OK)")
            return 0
        return 2

    if args.dry_run:
        print("All plotting libs present. Dry-run OK.")
        return 0

    input_path = None
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Specified file {input_path} does not exist")
            return 3
    else:
        # Look for aggregated CSVs inside runs/
        patterns = ["runs/experiments_*.csv", "runs/*.csv"]
        input_path = find_latest_csv(patterns)
        if input_path is None:
            print("No experiments CSV found in runs/. Run experiments first or pass --input")
            return 4

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(f"runs/plots_{ts}")

    make_plots(input_path, out_dir)
    return 0


if __name__ == "__main__":
    rc = main()
    if isinstance(rc, int):
        sys.exit(rc)
    sys.exit(0)
