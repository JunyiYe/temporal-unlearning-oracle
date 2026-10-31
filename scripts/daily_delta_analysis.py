"""Analyze the per-day deletion effect (mechanism probe) and draw the figure.

Pre-registered criterion (recorded before the data came in): the
template-ACTIVATION hypothesis predicts the daily deletion gain
concentrates on the test year's stressed days and is flat in the calm
fold. Verdict from the data: REFUTED. The gain is flat across stress
quintiles in every fold under both stress proxies, and the oracle beats
the reference on 73-99% of individual days. What varies with the regime
is the level of the daily gain, not its day pattern.

Inputs:  outputs/daily_delta/*_strict_*.json  (from daily_delta.py)
Outputs: prints the quintile table; writes fig_daily.pdf/png into both
         paper trees' figs/ directories.
"""
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = (_REPO)
PANEL = str(_REPO / "data" / "sp500_kaggle_returns.csv")
ARCHS = ["transformer", "tsmixer", "segrnn"]
FOLDS = ["2022", "2025", "2023", "2024"]
VOL = {"2022": 24, "2025": 18, "2023": 14, "2024": 12}
FCOL = {"2022": "#7a1f1f", "2025": "#b05c22", "2023": "#4a6fa5", "2024": "#7d7d7d"}

day = defaultdict(list)
with open(PANEL) as f:
    r = csv.reader(f)
    next(r)
    for d, s, v in r:
        day[d].append(float(v))
stress = {d: abs(sum(v) / len(v)) for d, v in day.items()}


def mean(v):
    return sum(v) / len(v)


def sem(v):
    if len(v) < 2:
        return 0.0
    mu = mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) \
        / math.sqrt(len(v))


series = {}
for fold in FOLDS:
    per = defaultdict(list)
    for a in ARCHS:
        d = json.load(open(ROOT / "outputs" / "daily_delta" /
                           f"{a}_sp500_vol_era2020_strict_fold{fold}_daily.json"))
        for i, dt in enumerate(d["dates"]):
            per[dt].append(mean([d["seeds"][s]["ora"][i] - d["seeds"][s]["ref"][i]
                                 for s in d["seeds"]]))
    ds = sorted((stress[dt], mean(v)) for dt, v in per.items() if dt in stress)
    n = len(ds)
    q = n // 5
    rows = []
    for k in range(5):
        chunk = [x[1] for x in ds[k * q:(k + 1) * q if k < 4 else n]]
        rows.append((mean(chunk), sem(chunk)))
    pos = sum(1 for _, v in ds if v > 0) / n
    series[fold] = {"q": rows, "pos": pos, "n": n}
    print(f"fold{fold}: " + "  ".join(f"Q{k+1} {m:+.4f}±{e:.4f}"
                                      for k, (m, e) in enumerate(rows))
          + f"   positive days {pos:.0%}")

fig, ax = plt.subplots(figsize=(3.35, 2.2))
ax.axhline(0, color="#444444", linewidth=0.8, zorder=1)
for fold in FOLDS:
    qs = series[fold]["q"]
    ax.errorbar(range(1, 6), [m for m, _ in qs], yerr=[e for _, e in qs],
                color=FCOL[fold], linewidth=1.2, marker="o", markersize=3.0,
                elinewidth=0.7, capsize=1.6,
                label=f"{fold} fold ({VOL[fold]}%), "
                      f"{series[fold]['pos']:.0%} of days positive")
ax.set_xticks(range(1, 6))
ax.set_xticklabels(["calmest\n20%", "Q2", "Q3", "Q4", "most\nstressed"],
                   fontsize=6.6)
ax.set_xlabel("test days grouped by that day's market stress")
ax.set_ylabel(r"daily deletion effect $\Delta$ (IC)")
ax.legend(frameon=False, loc="upper right", handletextpad=0.4,
          borderpad=0.2, labelspacing=0.25, fontsize=6.0)
fig.tight_layout(pad=0.4)
for out in [ROOT / "paper" / "figs", ROOT / "paper_v3" / "figs"]:
    fig.savefig(out / "fig_daily.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / "fig_daily.png", dpi=200, bbox_inches="tight",
                pad_inches=0.01)
print("wrote fig_daily")
