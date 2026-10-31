"""Attribute the stress-fold p99-vs-absmax daily damage gap to contaminated
(envelope-break-exposed) vs clean days.

Inputs: outputs/daily_attrib/<model>_fold<Y>_seed<S>_daily_ic.csv (from
rescue_check.py --dump-daily) and data/sp500_kaggle_returns.csv (for the
sigma_t envelope). A test day is *contaminated* if its trailing 64-day
context contains a day whose cross-sectional dispersion exceeds the
calibration year's maximum (the calibration envelope of Fig. regime).

For each model x fold: tax_t = IC_absmax(t) - IC_p99(t)  (the daily cost of
percentile calibration relative to abs-max; positive = p99 worse that day).
Reported per cell, then aggregated across seeds: mean tax on contaminated
vs clean days, share of total tax mass on contaminated days, paired t.

    python scripts/daily_attrib_analysis.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = {2018: 2017, 2020: 2019}


def contaminated_days(disp: pd.Series, test_year: int, cal_year: int) -> set:
    all_days = list(disp.index)
    pos = {d: i for i, d in enumerate(all_days)}
    cal_max = disp[disp.index.year == cal_year].max()
    exc = [d for d in all_days if d.year == test_year and disp[d] > cal_max]
    cont: set = set()
    for d in exc:
        cont |= set(all_days[pos[d]:pos[d] + 64])
    return {d for d in cont if d.year == test_year}


def main() -> None:
    ret = pd.read_csv("data/sp500_kaggle_returns.csv", parse_dates=["date"])
    col = "value" if "value" in ret.columns else "return"
    disp = ret.groupby("date")[col].std()

    results: dict[str, dict] = {}
    for fold, cal in FOLDS.items():
        cont = contaminated_days(disp, fold, cal)
        files = sorted(glob.glob(f"outputs/daily_attrib/*_fold{fold}_seed*_daily_ic.csv"))
        by_model: dict[str, list[dict]] = {}
        for f in files:
            m = re.match(rf".*/(\w+)_fold{fold}_seed(\d+)_daily_ic\.csv", f)
            model, seed = m.group(1), int(m.group(2))
            df = pd.read_csv(f, index_col=0)
            df.index = pd.to_datetime(df.index)
            p99col = next(c for c in df.columns if c.startswith("w4a4_p0.99") and not c.startswith("w4a4_p0.999"))
            tax = df["w4a4_absmax"] - df[p99col]  # daily IC advantage of absmax over p99
            is_c = tax.index.isin(cont)
            total = tax.sum()
            by_model.setdefault(model, []).append({
                "seed": seed,
                "tax_mean_cont": float(tax[is_c].mean()) if is_c.any() else float("nan"),
                "tax_mean_clean": float(tax[~is_c].mean()),
                "tax_share_cont": float(tax[is_c].sum() / total) if total != 0 else float("nan"),
                "tax_annual_mean": float(tax.mean()),
                "n_cont": int(is_c.sum()), "n_days": int(len(tax)),
            })
        for model, cells in by_model.items():
            cdf = pd.DataFrame(cells)
            diff = cdf["tax_mean_cont"] - cdf["tax_mean_clean"]
            n = len(diff.dropna())
            t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n))) if n > 1 and diff.std(ddof=1) > 0 else float("nan")
            results[f"{model}_fold{fold}"] = {
                "cells": n,
                "day_share_cont": round(cells[0]["n_cont"] / cells[0]["n_days"], 3),
                "tax_annual_mean": round(float(cdf["tax_annual_mean"].mean()), 4),
                "tax_mean_cont": round(float(cdf["tax_mean_cont"].mean()), 4),
                "tax_mean_clean": round(float(cdf["tax_mean_clean"].mean()), 4),
                "tax_share_cont_mean": round(float(cdf["tax_share_cont"].mean()), 3),
                "t_cont_vs_clean_across_seeds": round(t, 2),
            }

    out = Path("outputs/daily_attrib_summary.json")
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    hdr = f"{'model_fold':<26} {'annual':>8} {'cont':>8} {'clean':>8} {'share_cont':>10} {'t':>6}"
    print(hdr)
    for k, v in sorted(results.items()):
        print(f"{k:<26} {v['tax_annual_mean']:>8.4f} {v['tax_mean_cont']:>8.4f} "
              f"{v['tax_mean_clean']:>8.4f} {v['tax_share_cont_mean']:>10.3f} "
              f"{v['t_cont_vs_clean_across_seeds']:>6.2f}")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
