"""Aggregate JSON metrics from the runs/ directory into a CSV and produce a simple plot.

Usage:
  python scripts\aggregate_runs.py

Outputs:
- runs/metrics_aggregated.csv
- runs/metrics_summary.png  (if matplotlib is available)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import csv

RUNS_DIR = Path("runs")
OUT_CSV = RUNS_DIR / "metrics_aggregated.csv"
OUT_PNG = RUNS_DIR / "metrics_summary.png"


def find_metrics_files(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_file() and p.name.startswith("metrics_") and p.suffix == ".json")


def load_metrics(fp: Path) -> Dict[str, Any]:
    try:
        with fp.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"__error__": True, "__path__": str(fp)}


def aggregate(runs_dir: Path) -> List[Dict[str, Any]]:
    files = find_metrics_files(runs_dir)
    rows: List[Dict[str, Any]] = []
    for f in files:
        m = load_metrics(f)
        m["__file__"] = f.name
        rows.append(m)
    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        print("No metrics files found in", out_csv.parent)
        return
    # Compute union of all keys for header
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = sorted(keys)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
    print("Wrote CSV:", out_csv)


def try_plot(rows: List[Dict[str, Any]], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print("matplotlib not available or failed to import; skipping plot:", e)
        return

    # Pick a few numeric columns if present
    numeric_keys = [k for k in ["rescued", "ttr_median", "steps_done", "total_xy_disp"] if any(isinstance(r.get(k), (int, float)) for r in rows)]
    if not numeric_keys:
        print("No numeric keys found for plotting; skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(rows)))
    for k in numeric_keys:
        y = [float(r.get(k)) if (r.get(k) is not None and r.get(k) != "") else float('nan') for r in rows]
        ax.plot(x, y, marker='o', label=k)
    ax.set_xlabel('run index')
    ax.set_ylabel('value')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    print("Wrote plot:", out_png)


if __name__ == "__main__":
    rows = aggregate(RUNS_DIR)
    write_csv(rows, OUT_CSV)
    try_plot(rows, OUT_PNG)
