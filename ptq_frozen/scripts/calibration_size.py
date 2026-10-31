"""Calibration-set size study: how few windows are enough, and does the
statistic (abs-max vs p99) change the answer?

Grid: sizes x {absmax, p99} x (fold2024 x seeds), paired daily deltas vs fp32.
Abs-max should degrade as size shrinks (max of fewer samples underestimates
the range -> clipping) while p99 should be far more size-stable — if so, the
percentile recipe also wins on calibration-data efficiency.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_calibration_loader, build_datasets
from fin_ts_ptq.metrics.significance import daily_ic_frame, paired_delta_tests
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.quantization.rtn import fake_static_int8
from fin_ts_ptq.training.loop import predict
from fin_ts_ptq.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-set size study.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=2024)
    parser.add_argument("--seeds", default="7,8,9,10,11")
    parser.add_argument("--sizes", default="16,64,256,1024,4096")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-years", type=int, default=None,
                        help="Fixed training-window length; must match the sweep (norm stats).")
    parser.add_argument("--sweep-dir", default="outputs/sweep")
    parser.add_argument("--output-dir", default="outputs/mechanism")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    model_key = stem
    for suffix in ("_sp500v2_vol_tuned", "_sp500v2_vol_opt", "_sp500_vol_tuned", "_sp500_vol_opt"):
        model_key = model_key.replace(suffix, "")
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["train_end"] = f"{args.fold - 2}-12-31"
    cfg["data"]["val_end"] = f"{args.fold - 1}-12-31"
    cfg["data"]["test_end"] = f"{args.fold}-12-31"
    if args.train_years is not None:
        cfg["data"]["train_start"] = f"{args.fold - 2 - args.train_years}-12-31"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hac_lag = max(0, int(cfg["task"].get("forward_period", 1)) - 1)
    data = build_datasets(cfg, splits=("val", "test"))
    test_ds = data.splits["test"]
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    per_cell: dict[str, list[float]] = {}
    for seed in [int(s) for s in str(args.seeds).split(",") if s]:
        ckpt = Path(args.sweep_dir) / f"{stem}_fold{args.fold}_seed{seed}_model.pt"
        if not ckpt.exists():
            print(f"[SKIP] {ckpt}", flush=True)
            continue
        set_seed(seed)
        model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

        preds: dict[str, np.ndarray] = {}
        target_true, preds["fp32"] = predict(model, loader, device)
        for size in [int(v) for v in str(args.sizes).split(",") if v]:
            calibration_cfg = dict(cfg.get("ptq", {}).get("calibration", {}))
            calibration_cfg["num_windows"] = size
            cal_inputs = [b[0] for b in build_calibration_loader(
                data.splits["val"], policy="recent", calibration_cfg=calibration_cfg,
                seed=seed, batch_size=int(args.batch_size))]
            for pct, tag in ((None, "absmax"), (0.99, "p99")):
                _, preds[f"n{size}_{tag}"] = predict(
                    fake_static_int8(model, cal_inputs, num_bits=4, percentile=pct), loader, device)
        frame = daily_ic_frame(test_ds.meta, target_true, preds)
        tests = paired_delta_tests(frame, baseline="fp32", lag=hac_lag)
        for k, t in tests.items():
            per_cell.setdefault(k, []).append(t["delta_ic_mean"])
        print(f"[CELL] seed{seed}: " + "  ".join(f"{k}={t['delta_ic_mean']:+.4f}" for k, t in sorted(tests.items())),
              flush=True)

    summary = {
        "config": str(args.config), "fold": args.fold,
        "cells": len(next(iter(per_cell.values()), [])),
        "delta_ic": {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                     for k, v in per_cell.items()},
    }
    out = Path(args.output_dir) / f"{model_key}_calibration_size.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["delta_ic"], indent=2, sort_keys=True))
    print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    main()
