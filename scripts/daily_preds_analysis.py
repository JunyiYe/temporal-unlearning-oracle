"""Direction of the distortion: what does training on 2020 make the
reference over-predict, relative to its own oracle?

For every test window, diff = pred_ref - pred_ora (mean over 10 seeds;
predictions are of the day-standardized log forward volatility, so
diff > 0 means the reference ranks that stock as relatively more
volatile than the oracle does). Per test day we compute the
cross-sectional Spearman correlation of diff with the stock's RAW
trailing 20-day return volatility (recomputed from the panel, since
network inputs are window-normalized), and with the realized target y.

Hypothesis ("persistence inflation"): training on 2020 teaches an
exaggerated volatility persistence, so diff should correlate
POSITIVELY with trailing volatility, stably across days, archs, folds.

Inputs:  outputs/daily_preds/*_preds.npz  (from daily_preds.py)
"""
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = (_REPO)
PANEL = str(_REPO / "data" / "sp500_kaggle_returns.csv")
ARCHS = ["transformer", "tsmixer", "segrnn"]
FOLDS = ["2022", "2023", "2024", "2025"]
SEEDS = list(range(4, 14))

# ---- raw trailing 20-day volatility per (asset, date), from the panel
by_asset = defaultdict(list)
with open(PANEL) as f:
    r = csv.reader(f)
    next(r)
    for d, s, v in r:
        by_asset[s].append((d, float(v)))
trailvol = {}
for s, rows in by_asset.items():
    rows.sort()
    rets = [v for _, v in rows]
    for i, (d, _) in enumerate(rows):
        if i >= 19:
            w = rets[i - 19:i + 1]
            mu = sum(w) / 20
            trailvol[(s, d)] = math.sqrt(sum((x - mu) ** 2 for x in w) / 19)


def daily_spearman(dates, assets, a, b):
    """mean +- sem over days of the cross-sectional Spearman corr(a, b)."""
    idx = defaultdict(list)
    for i, d in enumerate(dates):
        idx[d].append(i)
    cors = []
    for d, ii in idx.items():
        if len(ii) < 30:
            continue
        x = a[ii]
        y = b[ii]
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        if denom > 0:
            cors.append(float((rx * ry).sum() / denom))
    m = sum(cors) / len(cors)
    se = (sum((c - m) ** 2 for c in cors) / (len(cors) - 1)) ** 0.5 \
        / math.sqrt(len(cors))
    return m, se, len(cors)


print(f"{'arch':13s}{'fold':>6s}{'corr(diff, trail vol20)':>26s}"
      f"{'corr(diff, realized y)':>25s}{'days':>6s}")
for a in ARCHS:
    for fold in FOLDS:
        z = np.load(ROOT / "outputs" / "daily_preds" /
                    f"{a}_sp500_vol_era2020_strict_fold{fold}_preds.npz",
                    allow_pickle=False)
        dates = z["date"]
        assets = z["asset"]
        diff = np.mean([z[f"pred_ref_seed{s}"] - z[f"pred_ora_seed{s}"]
                        for s in SEEDS], axis=0).ravel()
        tv = np.array([trailvol.get((str(s), str(d)), np.nan)
                       for s, d in zip(assets, dates)])
        ok = ~np.isnan(tv)
        if a == ARCHS[0] and fold == FOLDS[0]:
            print(f"[join] {ok.sum()}/{len(ok)} windows matched")
        m1, e1, nd = daily_spearman(dates[ok], assets[ok], diff[ok], tv[ok])
        m2, e2, _ = daily_spearman(dates, assets, diff, z["y"].ravel())
        print(f"{a:13s}{fold:>6s}{m1:+21.3f}±{e1:.3f}"
              f"{m2:+20.3f}±{e2:.3f}{nd:>6d}")
