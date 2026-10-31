"""SmoothQuant rescue attempt for SegRNN across fold x seed checkpoints.

Evaluates, paired by day against fp32:
  - W4A4 abs-max (reference), W4A4 p99 (best calibration-only rescue),
  - smoothed + W4A4 abs-max, smoothed + W4A4 p99,
plus an fp32-equivalence sanity check for the smoothing itself.
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
from fin_ts_ptq.quantization.smoothing import smooth_segrnn_gru
from fin_ts_ptq.training.loop import predict
from fin_ts_ptq.utils.seed import set_seed


def fold_cfg(base: dict, test_year: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["data"]["train_end"] = f"{test_year - 2}-12-31"
    cfg["data"]["val_end"] = f"{test_year - 1}-12-31"
    cfg["data"]["test_end"] = f"{test_year}-12-31"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="SmoothQuant x-path rescue for SegRNN.")
    parser.add_argument("--config", default="configs/segrnn_sp500_vol_tuned.yaml")
    parser.add_argument("--folds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--seeds", default="7,8,9,10,11")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hac_lag = max(0, int(base_cfg["task"].get("forward_period", 1)) - 1)
    per_cell: dict[str, list[float]] = {}
    fp32_equiv_worst = 0.0

    for fold in [int(f) for f in str(args.folds).split(",") if f]:
        cfg = fold_cfg(base_cfg, fold)
        data = build_datasets(cfg, splits=("val", "test"))
        test_ds = data.splits["test"]
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
        for seed in [int(s) for s in str(args.seeds).split(",") if s]:
            ckpt = Path("outputs/sweep") / f"{stem}_fold{fold}_seed{seed}_model.pt"
            if not ckpt.exists():
                print(f"[SKIP] {ckpt}", flush=True)
                continue
            set_seed(seed)
            model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            cal_inputs = [b[0] for b in build_calibration_loader(
                data.splits["val"], policy="recent", calibration_cfg=calibration_cfg,
                seed=seed, batch_size=int(args.batch_size))]

            smoothed = smooth_segrnn_gru(model, cal_inputs, alpha=args.alpha)
            with torch.no_grad():
                xb = cal_inputs[0][:64]
                equiv = float((model.cpu()(xb) - smoothed(xb)).abs().max())
            fp32_equiv_worst = max(fp32_equiv_worst, equiv)

            preds: dict[str, np.ndarray] = {}
            target_true, preds["fp32"] = predict(model, loader, device)
            variants = {
                "w4a4_absmax": (model, None),
                "w4a4_p99": (model, 0.99),
                "smooth_absmax": (smoothed, None),
                "smooth_p99": (smoothed, 0.99),
            }
            for name, (m, pct) in variants.items():
                _, preds[name] = predict(
                    fake_static_int8(m, cal_inputs, num_bits=4, percentile=pct), loader, device)

            frame = daily_ic_frame(test_ds.meta, target_true, preds)
            tests = paired_delta_tests(frame, baseline="fp32", lag=hac_lag)
            for k, t in tests.items():
                per_cell.setdefault(k, []).append(t["delta_ic_mean"])
            print(f"[CELL] fold{fold} seed{seed} equiv={equiv:.2e}: "
                  + "  ".join(f"{k}={t['delta_ic_mean']:+.4f}" for k, t in tests.items()), flush=True)

    summary = {
        "config": str(args.config),
        "alpha": args.alpha,
        "fp32_equiv_worst": fp32_equiv_worst,
        "cells": len(next(iter(per_cell.values()), [])),
        "delta_ic": {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                     for k, v in per_cell.items()},
    }
    out = Path("outputs/mechanism") / "segrnn_smoothquant.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["delta_ic"], indent=2, sort_keys=True))
    print(f"[OK] Wrote {out} (fp32 equivalence worst diff {fp32_equiv_worst:.2e})")


if __name__ == "__main__":
    main()
