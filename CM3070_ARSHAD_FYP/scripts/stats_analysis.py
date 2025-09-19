"""Simple statistical analysis for aggregated experiment metrics.

Usage:
  python scripts\stats_analysis.py

Reads `runs/metrics_aggregated.csv` (created by run_experiments or aggregate_runs)
and computes summary statistics, bootstrap confidence intervals for median TTR,
and writes `runs/metrics_stats.csv` and `runs/metrics_stats.png`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any
import statistics
import random

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore

RUNS_DIR = Path('runs')
AGG_CSV = RUNS_DIR / 'metrics_aggregated.csv'
OUT_SUMMARY = RUNS_DIR / 'metrics_stats.csv'
OUT_PNG = RUNS_DIR / 'metrics_stats.png'


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open('r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # Convert numeric fields when possible
            for k in ['rescued', 'ttr_median', 'steps_done', 'total_xy_disp', 'run_seed']:
                v = r.get(k)
                if v is None or v == '':
                    r[k] = None
                else:
                    try:
                        r[k] = float(v)
                    except Exception:
                        r[k] = None
            rows.append(r)
    return rows


def bootstrap_median(data: List[float], n_iter: int = 2000, alpha: float = 0.05) -> (float, float, float):
    if not data:
        return (float('nan'), float('nan'), float('nan'))
    med = statistics.median(data)
    boot = []
    n = len(data)
    for _ in range(n_iter):
        sample = [random.choice(data) for _ in range(n)]
        boot.append(statistics.median(sample))
    boot.sort()
    lo = boot[int((alpha/2.0)*n_iter)]
    hi = boot[int((1-alpha/2.0)*n_iter) - 1]
    return (med, lo, hi)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Group by run_config if available, else treat all together
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        key = r.get('run_config') or '__all__'
        groups.setdefault(key, []).append(r)
    out = []
    for key, group in groups.items():
        rescued = [float(r['rescued']) for r in group if r.get('rescued') is not None]
        ttrs = [float(r['ttr_median']) for r in group if r.get('ttr_median') is not None]
        steps = [float(r['steps_done']) for r in group if r.get('steps_done') is not None]
        disp = [float(r['total_xy_disp']) for r in group if r.get('total_xy_disp') is not None]
        row = {
            'run_config': key,
            'n_runs': len(group),
            'rescued_mean': statistics.mean(rescued) if rescued else None,
            'rescued_median': statistics.median(rescued) if rescued else None,
            'ttr_median_median': statistics.median(ttrs) if ttrs else None,
            'ttr_median_ci_lo': None,
            'ttr_median_ci_hi': None,
            'steps_mean': statistics.mean(steps) if steps else None,
            'total_xy_disp_mean': statistics.mean(disp) if disp else None,
        }
        if ttrs:
            med, lo, hi = bootstrap_median(ttrs)
            row['ttr_median_ci_lo'] = lo
            row['ttr_median_ci_hi'] = hi
        out.append(row)
    return out


def write_summary(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        print('no rows to summarize')
        return
    keys = list(rows[0].keys())
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print('Wrote summary CSV:', out_csv)


def plot_summary(summary_rows: List[Dict[str, Any]], out_png: Path) -> None:
    if plt is None:
        print('matplotlib not available; skipping plot')
        return
    # Plot rescued_mean and ttr_median_median per group
    labels = [r['run_config'] for r in summary_rows]
    rescued = [r['rescued_mean'] or 0 for r in summary_rows]
    ttr = [r['ttr_median_median'] or float('nan') for r in summary_rows]
    x = range(len(labels))
    fig, ax1 = plt.subplots()
    ax1.bar(x, rescued, color='tab:blue', alpha=0.6)
    ax1.set_ylabel('rescued_mean', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(x, ttr, color='tab:orange', marker='o')
    ax2.set_ylabel('ttr_median_median', color='tab:orange')
    plt.xticks(x, labels, rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(out_png)
    print('Wrote plot:', out_png)


if __name__ == '__main__':
    rows = load_rows(AGG_CSV)
    summary = summarize(rows)
    write_summary(summary, OUT_SUMMARY)
    plot_summary(summary, OUT_PNG)
