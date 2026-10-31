"""Does the deletion effect survive in a model with no capacity to memorize?

Every architecture in the study is a neural forecaster, so a sceptic can read
the exposure result as a statement about deep models rather than about the
data. HAR-RV is the standard econometric benchmark: four parameters, ordinary
least squares, no capacity to store an era. Running the same deletion protocol
on it separates the two readings.

  reference   HAR fit on the fold's full five-year training window
  oracle      HAR refit on the same window minus era E (L1 semantics: any row
              whose information date or target date falls inside E is dropped)
  Delta       IC_test(oracle) - IC_test(reference), paired by day with HAC

If HAR gains from deleting the crisis year too, the effect lives in the data.
If HAR is inert while the neural models move, it lives in model capacity.

    python scripts/har_deletion.py --fold 2022 --eras 2016,2017,2018,2019,2020
"""
from __future__ import annotations
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(_REPO / "ptq_frozen" / "src"))
from fin_ts_ptq.data.panel import PanelSpec, load_csv_panel          # noqa: E402
from fin_ts_ptq.metrics.significance import daily_ic_frame, newey_west_mean_test  # noqa: E402

EPS = 1e-6
CONTEXT = 64   # the neural models' context length, for the L1 halo
FWD = 5


def build_har_table(panel: pd.DataFrame, fwd: int = FWD) -> pd.DataFrame:
    rows = []
    for asset, g in panel.sort_values(["asset", "date"]).groupby("asset", sort=False):
        r = g["value"].reset_index(drop=True)
        y = r.rolling(fwd).std()
        rv1 = r.abs().shift(fwd)
        rv5 = r.rolling(5).std().shift(fwd)
        rv22 = r.rolling(22).std().shift(fwd)
        rows.append(pd.DataFrame({
            "date": g["date"].to_numpy(), "asset": asset,
            "y": np.log(y + EPS), "x1": np.log(rv1 + EPS),
            "x5": np.log(rv5 + EPS), "x22": np.log(rv22 + EPS),
        }))
    return pd.concat(rows, ignore_index=True).dropna()


def fit(train: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(train)), train[["x1", "x5", "x22"]].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)
    return beta


def predict(beta: np.ndarray, tbl: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(tbl)), tbl[["x1", "x5", "x22"]].to_numpy()])
    return X @ beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_REPO / "data" / "sp500_kaggle_returns.csv"))
    ap.add_argument("--fold", type=int, required=True, help="test year")
    ap.add_argument("--eras", required=True)
    ap.add_argument("--train-years", type=int, default=5)
    ap.add_argument("--modes", default="strict,target",
                    help="strict = L1 (context halo), target = L0 (target date only)")
    ap.add_argument("--boot", type=int, default=10,
                    help="bootstrap refits standing in for the neural models' seed population")
    ap.add_argument("--out", default=str(_REPO / "results" / "har_deletion.json"))
    args = ap.parse_args()

    panel = load_csv_panel(PanelSpec(path=args.data, min_history=1000))
    tbl = build_har_table(panel)

    train_hi = pd.Timestamp(f"{args.fold - 2}-12-31")
    train_lo = pd.Timestamp(f"{args.fold - 2 - args.train_years}-12-31")
    test_lo, test_hi = pd.Timestamp(f"{args.fold - 1}-12-31"), pd.Timestamp(f"{args.fold}-12-31")
    train = tbl[(tbl["date"] > train_lo) & (tbl["date"] <= train_hi)]
    test = tbl[(tbl["date"] > test_lo) & (tbl["date"] <= test_hi)]

    ref_pred = predict(fit(train), test)
    base = daily_ic_frame(test[["date"]], test["y"].to_numpy(), {"ref": ref_pred})
    ref_ic, ref_t, _, _ = newey_west_mean_test(base["ref"].to_numpy(), lag=FWD - 1)
    print(f"fold {args.fold}: HAR reference IC = {ref_ic:+.4f} (t={ref_t:.1f}), "
          f"train rows={len(train)}, test days={len(base)}")

    ref_beta = fit(train)
    rng = np.random.default_rng(0)
    out = {"fold": args.fold, "train_years": args.train_years,
           "reference_ic": ref_ic, "n_train": int(len(train)), "cells": {}}

    def era_mask(t, era, mode):
        """Rows the deletion removes. L1 also drops the 64-day context halo."""
        lo = pd.Timestamp(f"{era}-01-01")
        hi = pd.Timestamp(f"{era}-12-31") + pd.Timedelta(days=FWD * 2)
        if mode == "strict":
            lo = lo - pd.Timedelta(days=int(CONTEXT * 1.5))
        return (t["date"] >= lo) & (t["date"] <= hi)

    for mode in [m for m in args.modes.split(",") if m]:
        for era in [int(e) for e in args.eras.split(",") if e]:
            m = era_mask(train, era, mode)
            forget, keep = train[m], train[~m]
            ora_beta = fit(keep)
            ora_pred = predict(ora_beta, test)
            frame = daily_ic_frame(test[["date"]], test["y"].to_numpy(),
                                   {"ref": ref_pred, "ora": ora_pred})
            dm, dt, dp, _ = newey_west_mean_test(
                frame["ora"].to_numpy() - frame["ref"].to_numpy(), lag=FWD - 1)
            ora_ic, _, _, _ = newey_west_mean_test(frame["ora"].to_numpy(), lag=FWD - 1)

            # memorization gap on the deleted era, and a size-matched retain control
            yf = forget["y"].to_numpy()
            g = float(((predict(ora_beta, forget) - yf) ** 2).mean()
                      - ((predict(ref_beta, forget) - yf) ** 2).mean())
            idx = rng.choice(len(keep), size=min(len(forget), len(keep)), replace=False)
            ctrl = keep.iloc[idx]; yc = ctrl["y"].to_numpy()
            g_ctrl = float(((predict(ora_beta, ctrl) - yc) ** 2).mean()
                           - ((predict(ref_beta, ctrl) - yc) ** 2).mean())

            # bootstrap population: the deterministic analogue of seed variation
            ref_l, ora_l = [], []
            for _ in range(args.boot):
                bi = rng.choice(len(train), size=len(train), replace=True)
                ref_l.append(float(((predict(fit(train.iloc[bi]), forget) - yf) ** 2).mean()))
                bk = rng.choice(len(keep), size=len(keep), replace=True)
                ora_l.append(float(((predict(fit(keep.iloc[bk]), forget) - yf) ** 2).mean()))
            pooled = np.sqrt((np.var(ref_l, ddof=1) + np.var(ora_l, ddof=1)) / 2)
            dprime = float((np.mean(ora_l) - np.mean(ref_l)) / pooled) if pooled > 0 else float("nan")

            key = f"{era}_{mode}"
            out["cells"][key] = {"oracle_ic": ora_ic, "delta": dm, "tstat": dt, "pvalue": dp,
                                 "G": g, "G_retain_control": g_ctrl, "dprime_bootstrap": dprime,
                                 "rows_dropped": 1 - len(keep) / len(train)}
            sig = "***" if abs(dt) > 3.5 else ("*" if abs(dt) > 2.3 else "n.s.")
            print(f"  [{mode:6s}] delete {era}: D={dm:+.4f} (t{dt:6.1f}) {sig:4s} "
                  f"G={g:+.5f} (retain ctrl {g_ctrl:+.5f})  d'boot={dprime:5.1f}  "
                  f"dropped {1 - len(keep)/len(train):.1%}")

    # placebo: drop the same number of rows at random, no era structure
    n_drop = int(round(np.mean([c["rows_dropped"] for c in out["cells"].values()]) * len(train)))
    pi = rng.choice(len(train), size=n_drop, replace=False)
    pl_keep = train.drop(train.index[pi])
    pl_frame = daily_ic_frame(test[["date"]], test["y"].to_numpy(),
                              {"ref": ref_pred, "ora": predict(fit(pl_keep), test)})
    pm, pt, _, _ = newey_west_mean_test(
        pl_frame["ora"].to_numpy() - pl_frame["ref"].to_numpy(), lag=FWD - 1)
    out["placebo"] = {"delta": pm, "tstat": pt, "rows_dropped": n_drop / len(train)}
    print(f"  [placebo] random {n_drop/len(train):.1%} of rows: D={pm:+.4f} (t{pt:5.1f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    prev[str(args.fold)] = out
    Path(args.out).write_text(json.dumps(prev, indent=2, sort_keys=True))
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
