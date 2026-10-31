"""Per-year market conditions quoted in the paper (tab:years and axis labels).

For every calendar year of the evaluation panel, computes on the
equal-weighted market (the unweighted mean of all names present each day
in data/sp500_kaggle_returns.csv, i.e. the same panel every model trains
on, before the min-history filter):

  - annualized realized volatility: sample sd of the daily equal-weight
    return x sqrt(252), in per cent
  - maximum drawdown: worst peak-to-trough decline of the compounded
    equal-weight cumulative return within the year, in per cent

Run from the fin-ts-ptq repo root:
  python scripts/year_market_stats.py

Validation note (2026-08-08): reproduces the previously recorded values
exactly for 2017-2024; corrects 2016's drawdown to -11.3% (the older
hand-recorded -10.1% could not be reproduced) and adds 2025.
"""
import csv
import math
from collections import defaultdict

PANEL = "data/sp500_kaggle_returns.csv"

day = defaultdict(list)
with open(PANEL) as f:
    r = csv.reader(f)
    next(r)
    for d, s, v in r:
        day[d].append(float(v))

years = defaultdict(list)
for d in sorted(day):
    years[d[:4]].append(sum(day[d]) / len(day[d]))

print(f"{'year':6s}{'ann.vol%':>10s}{'maxDD%':>9s}{'days':>6s}")
for y in sorted(years):
    if y < "2016":
        continue
    rets = years[y]
    mu = sum(rets) / len(rets)
    vol = math.sqrt(sum((x - mu) ** 2 for x in rets) / (len(rets) - 1)) \
        * math.sqrt(252) * 100
    cum = peak = 1.0
    dd = 0.0
    for x in rets:
        cum *= 1 + x
        peak = max(peak, cum)
        dd = min(dd, cum / peak - 1)
    print(f"{y:6s}{vol:>10.1f}{dd * 100:>9.1f}{len(rets):>6d}")
