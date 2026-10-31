"""Mechanism probe: activation outliers, clip rates, and their regime structure.

For one config + checkpoint:
  - fp32 pass: per-layer activation stats (kurtosis/amax) + per-window amax
  - quantized pass(es): per-window clip fraction per FakeQuant layer + preds
Everything is saved per window with dates/assets so daily aggregation, ΔIC
pairing, and market-state regressions happen offline.

    python scripts/mechanism_probe.py --config configs/tsmixer_sp500_vol_tuned.yaml \
        --checkpoint outputs/sweep/tsmixer_sp500_vol_tuned_fold2024_seed7_model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_calibration_loader, build_datasets
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.quantization.profiling import profile_clipping, profile_fp32
from fin_ts_ptq.quantization.ptq import quantize_model
from fin_ts_ptq.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation/clip mechanism probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Default: outputs/<config stem>_model.pt")
    parser.add_argument("--methods", default="static_int4_recent,static_int8_recent")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--train-years", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/mechanism")
    parser.add_argument("--label", default=None, help="Output label; default = checkpoint stem")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.test_year is not None:
        cfg["data"]["train_end"] = f"{args.test_year - 2}-12-31"
        cfg["data"]["val_end"] = f"{args.test_year - 1}-12-31"
        cfg["data"]["test_end"] = f"{args.test_year}-12-31"
        if args.train_years is not None:
            cfg["data"]["train_start"] = f"{args.test_year - 2 - args.train_years}-12-31"
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    set_seed(int(cfg.get("seed", 7)))
    stem = Path(args.config).stem
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path("outputs") / f"{stem}_model.pt"
    label = args.label or checkpoint.stem.replace("_model", "")

    data = build_datasets(cfg, splits=("val", "test"))
    test_ds = data.splits["test"]
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[PROBE] fp32 pass ({len(test_ds)} windows)", flush=True)
    layer_stats, window_amax, fp32_preds = profile_fp32(model, loader, device)

    arrays: dict[str, np.ndarray] = {
        "target_true": test_ds.y.astype(np.float32),
        "raw_y": test_ds.raw_y.astype(np.float32),
        "date_ns": test_ds.meta["date"].to_numpy().astype("datetime64[ns]").astype(np.int64),
        "asset": test_ds.meta["asset"].to_numpy().astype(str),
        "pred::fp32": fp32_preds.astype(np.float32),
    }
    for lname, arr in window_amax.items():
        arrays[f"amax::{lname}"] = arr.astype(np.float32)

    calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
    for mlabel in [m for m in str(args.methods).split(",") if m]:
        method = next((p for p in ("static_int8", "static_int4") if mlabel.startswith(p + "_")), None)
        if method is None:
            raise ValueError(f"Only static methods supported, got {mlabel}")
        policy = mlabel[len(method) + 1 :]
        cal_loader = build_calibration_loader(
            data.splits["val"], policy=policy, calibration_cfg=calibration_cfg,
            seed=int(cfg.get("seed", 7)), batch_size=int(args.batch_size),
        )
        qmodel = quantize_model(model, method, cal_loader)
        print(f"[PROBE] clip pass: {mlabel}", flush=True)
        clip_arrays, qpreds = profile_clipping(qmodel, loader, device)
        arrays[f"pred::{mlabel}"] = qpreds.astype(np.float32)
        for lname, arr in clip_arrays.items():
            arrays[f"clip::{mlabel}::{lname}"] = arr.astype(np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{label}_probe.npz"
    np.savez_compressed(npz_path, **arrays)
    summary = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "n_windows": int(len(test_ds)),
        "fp32_layer_stats": layer_stats,
    }
    json_path = out_dir / f"{label}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: {"excess_kurtosis": round(v.get("excess_kurtosis", 0), 2),
                          "amax_over_std": round(v.get("amax_over_std", 0), 1)}
                      for k, v in layer_stats.items()}, indent=2))
    print(f"[OK] Wrote {npz_path} and {json_path}")


if __name__ == "__main__":
    main()
