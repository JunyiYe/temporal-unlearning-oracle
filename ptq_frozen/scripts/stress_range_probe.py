"""Do crisis-year activations actually leave the calibrated range?

For a given fold, compute per-layer calibration scales (abs-max / p99 /
p999 from the validation year, policy=recent, as in the main pipeline) and
compare them against the fp32 per-window activation amax on the test year.
Writes outputs/mechanism/<label>_rangecheck.json with per-layer overall and
worst-day exceedance rates, plus a per-day CSV for the worst layer.

    python scripts/stress_range_probe.py --config configs/tsmixer_sp500_vol_tuned.yaml \
        --test-year 2020 --seed 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_calibration_loader, build_datasets
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.quantization.profiling import profile_fp32
from fin_ts_ptq.quantization.rtn import collect_linear_input_amax
from fin_ts_ptq.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-range exceedance on a test year.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep-dir", default="outputs/sweep")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-years", type=int, default=None,
                        help="Fixed training-window length; must match the sweep (norm stats).")
    parser.add_argument("--output-dir", default="outputs/mechanism")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["train_end"] = f"{args.test_year - 2}-12-31"
    if args.train_years is not None:
        cfg["data"]["train_start"] = f"{args.test_year - 2 - args.train_years}-12-31"
    cfg["data"]["val_end"] = f"{args.test_year - 1}-12-31"
    cfg["data"]["test_end"] = f"{args.test_year}-12-31"
    cfg["seed"] = int(args.seed)
    set_seed(int(cfg["seed"]))

    stem = Path(args.config).stem
    label = f"{stem}_fold{args.test_year}_seed{args.seed}"
    checkpoint = Path(args.sweep_dir) / f"{label}_model.pt"

    data = build_datasets(cfg, splits=("val", "test"))
    test_ds = data.splits["test"]
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
    cal_loader = build_calibration_loader(
        data.splits["val"], policy="recent", calibration_cfg=calibration_cfg,
        seed=int(cfg["seed"]), batch_size=int(args.batch_size),
    )
    cal_inputs = [b[0] for b in cal_loader]
    scales = {
        "absmax": collect_linear_input_amax(model, cal_inputs),
        "p99": collect_linear_input_amax(model, cal_inputs, percentile=0.99),
        "p999": collect_linear_input_amax(model, cal_inputs, percentile=0.999),
    }

    _, window_amax, _ = profile_fp32(model, loader, device)
    dates = pd.to_datetime(test_ds.meta["date"].to_numpy())

    report: dict = {"config": str(args.config), "test_year": args.test_year,
                    "seed": args.seed, "layers": {}}
    worst = (None, -1.0)  # (layer, overall p99 exceedance)
    for lname, amax in window_amax.items():
        entry: dict = {}
        for sname, sdict in scales.items():
            if lname not in sdict:
                continue
            thr = float(sdict[lname])
            exceed = amax > thr
            daily = pd.DataFrame({"d": dates[: len(amax)], "e": exceed,
                                  "r": amax / max(thr, 1e-12)})
            by_day = daily.groupby("d").agg(exceed=("e", "mean"), ratio=("r", "max"))
            wd = by_day["exceed"].idxmax()
            entry[sname] = {
                "scale": thr,
                "overall_exceed_pct": float(100 * exceed.mean()),
                "worst_day": str(wd.date()),
                "worst_day_exceed_pct": float(100 * by_day.loc[wd, "exceed"]),
                "max_amax_over_scale": float(by_day["ratio"].max()),
            }
        report["layers"][lname] = entry
        if "p99" in entry and entry["p99"]["overall_exceed_pct"] > worst[1]:
            worst = (lname, entry["p99"]["overall_exceed_pct"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if worst[0] is not None:
        lname = worst[0]
        amax = window_amax[lname]
        daily = pd.DataFrame({"date": dates[: len(amax)], "amax": amax})
        for sname in ("absmax", "p99", "p999"):
            if lname in scales[sname]:
                daily[f"exceed_{sname}"] = amax > float(scales[sname][lname])
        by_day = daily.groupby("date").mean()
        by_day.to_csv(out_dir / f"{label}_rangecheck_daily.csv")
        report["worst_layer"] = lname

    (out_dir / f"{label}_rangecheck.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] Wrote {out_dir / f'{label}_rangecheck.json'}")


if __name__ == "__main__":
    main()
