"""Bet matrix (calibration-year x test-year), the E2 experiment of the v2 design.

Anchor-fold checkpoints are FROZEN (train window fixed by --anchor-fold and
--train-years); the calibration set is drawn from year C and the quantized
model is evaluated on every test year S in the span. Train is held fixed, so
each (C, S) cell isolates the calibration<->test relationship. Cells with
C >= S are look-ahead diagnostics (C == S is the oracle diagonal); the
deployable region is C < S.

Per (checkpoint, C, method) the model is quantized once and evaluated over
the whole 2018-2025 span in a single pass; daily ICs are split by year
afterwards. Output: long-format CSV of per-cell paired deltas + a per-seed
daily-IC dump for day-level analysis.

    python scripts/bet_matrix.py --config configs/tsmixer_sp500v2_vol_tuned.yaml \
        --anchor-fold 2018 --train-years 7 --seeds 1,2,3,4,5 \
        --calib-years 2017,2019,2020,2021,2024 --sweep-dir outputs/v2/sweep \
        --output-dir outputs/v2/bet_matrix
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_calibration_loader, build_datasets
from fin_ts_ptq.metrics.significance import daily_ic_frame
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.quantization.rtn import fake_static_int8
from fin_ts_ptq.training.loop import predict
from fin_ts_ptq.utils.seed import set_seed


def parse_list(v: str, cast=str) -> list:
    return [cast(p) for p in str(v).split(",") if p != ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-year x test-year bet matrix on frozen anchor-fold checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-fold", type=int, default=2018)
    parser.add_argument("--train-years", type=int, default=7)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--calib-years", default="2017,2019,2020,2021,2024")
    parser.add_argument("--test-span-end", type=int, default=2025)
    parser.add_argument("--percentiles", default="0.99,0.999",
                        help="Percentile scales evaluated alongside abs-max.")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--policy", default="random")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sweep-dir", default="outputs/v2/sweep")
    parser.add_argument("--output-dir", default="outputs/v2/bet_matrix")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    model_key = stem
    for suffix in ("_sp500v2_vol_tuned", "_sp500v2_vol_opt", "_sp500_vol_tuned", "_sp500_vol_opt"):
        model_key = model_key.replace(suffix, "")

    F = int(args.anchor_fold)
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["train_start"] = f"{F - 2 - args.train_years}-12-31"
    cfg["data"]["train_end"] = f"{F - 2}-12-31"
    cfg["data"]["val_end"] = f"{F - 1}-12-31"
    cfg["data"]["test_end"] = f"{args.test_span_end}-12-31"  # span, not one fold
    anchor_val_year = F - 1

    calib_years = parse_list(args.calib_years, int)
    percentiles = parse_list(args.percentiles, float)
    seeds = parse_list(args.seeds, int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_datasets(cfg, splits=("val", "test"))
    test_ds = data.splits["test"]
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
    print(f"[DATA] span windows: {len(test_ds)} (test years {F}..{args.test_span_end}), "
          f"val windows: {len(data.splits['val'])}", flush=True)

    def year_ds(year: int):
        if year == anchor_val_year:
            return data.splits["val"]
        return test_ds.subset_by_date(f"{year - 1}-12-31", f"{year}-12-31")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        ckpt = Path(args.sweep_dir) / f"{stem}_fold{F}_seed{seed}_model.pt"
        if not ckpt.exists():
            print(f"[SKIP] missing {ckpt}", flush=True)
            continue
        set_seed(seed)
        cfg["seed"] = seed
        model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

        preds: dict[str, np.ndarray] = {}
        target_true, preds["fp32"] = predict(model, loader, device)
        print(f"[SEED {seed}] fp32 done", flush=True)
        for C in calib_years:
            cal_ds = year_ds(C)
            cal_inputs = [b[0] for b in build_calibration_loader(
                cal_ds, policy=args.policy, calibration_cfg=calibration_cfg,
                seed=seed, batch_size=int(args.batch_size))]
            preds[f"c{C}::absmax"] = predict(
                fake_static_int8(model, cal_inputs, num_bits=args.bits), loader, device)[1]
            for p in percentiles:
                preds[f"c{C}::p{p:g}"] = predict(
                    fake_static_int8(model, cal_inputs, num_bits=args.bits, percentile=p), loader, device)[1]
            print(f"[SEED {seed}] calib {C} done", flush=True)

        frame = daily_ic_frame(test_ds.meta, target_true, preds)
        frame.to_csv(out_dir / f"{model_key}_anchor{F}_seed{seed}_daily_ic.csv")
        years = pd.to_datetime(frame.index).year
        for S in range(F, args.test_span_end + 1):
            sub = frame[years == S]
            if not len(sub):
                continue
            for col in frame.columns:
                if col == "fp32":
                    continue
                C, method = col.split("::")
                rows.append({
                    "model": model_key, "anchor_fold": F, "seed": seed,
                    "calib_year": int(C[1:]), "test_year": S, "method": method,
                    "fp32_daily_ic": float(sub["fp32"].mean()),
                    "delta_daily_ic": float((sub["fp32"] - sub[col]).mean()),
                    "n_days": int(len(sub)),
                })

    out = out_dir / f"{model_key}_anchor{F}_bet_matrix.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] Wrote {out} ({len(rows)} cells)", flush=True)


if __name__ == "__main__":
    main()
