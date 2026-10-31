"""Minimum-detectable-effect analysis for the evaluation-protocol section.

A: test days needed (t=2) just to confirm each task's base signal exists.
B: days needed to detect a 10% relative W4A4 degradation, paired-by-day vs
   unpaired — quantifying why the paired protocol matters.
Reads existing significance logs and noise-equivalence daily IC series.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd


def main() -> None:
    print("=== A. days needed to detect the base signal (t=2) ===")
    for f in sorted(glob.glob("outputs/significance_sig_*.log")):
        try:
            txt = open(f).read()
            d = json.loads(txt[txt.index("{"):])
        except Exception:
            continue
        ic, n = d["daily_ic_mean"], d["n_test_days"]
        t_naive = d.get("daily_ic_tstat_naive")
        if not t_naive or not ic:
            continue
        sigma = abs(ic) * np.sqrt(n) / abs(t_naive)
        need = (2 * sigma / abs(ic)) ** 2
        name = d["config"].split("/")[-1].replace("_sp500", "").replace("_opt.yaml", "")
        print(f"{name:38s} IC={ic:+.4f} sigma={sigma:.3f} have={n} need={need:.0f}")

    print("\n=== B. days to detect 10% relative degradation: paired vs unpaired ===")
    for m in ["tsmixer", "transformer", "itransformer", "patchtst", "timemixer", "dlinear"]:
        try:
            df = pd.read_csv(f"outputs/{m}_sp500_vol_opt_noise_equiv.daily_ic.csv")
        except FileNotFoundError:
            continue
        ic = df["fp32"].mean()
        if abs(ic) < 0.02:
            continue
        sd_delta = (df["fp32"] - df["static_int4_recent"]).std()
        sd_ic = df["fp32"].std()
        target = 0.10 * abs(ic)
        n_paired = (2 * sd_delta / target) ** 2
        n_unpaired = (2 * np.sqrt(2) * sd_ic / target) ** 2
        print(f"{m:14s} IC={ic:+.3f} paired={n_paired:.0f}d unpaired={n_unpaired:.0f}d "
              f"advantage={n_unpaired / max(n_paired, 1e-9):.0f}x")


if __name__ == "__main__":
    main()
