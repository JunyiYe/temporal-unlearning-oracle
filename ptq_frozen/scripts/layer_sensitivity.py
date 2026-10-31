"""Per-layer W4A4 sensitivity: which layer carries the damage?

For one config + checkpoint, evaluates daily IC for:
  - fp32, full W4A4,
  - only-one: exactly one layer quantized (per quantizable layer),
  - leave-one-out: all layers quantized except one.
Deltas are paired by day against fp32 (Newey-West HAC). If damage is
"importance-weighted resolution loss", only-one should localize it to a few
layers, and leaving those out should recover most of the IC.

    python scripts/layer_sensitivity.py --config configs/tsmixer_sp500_vol_tuned.yaml \
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
from fin_ts_ptq.metrics.significance import daily_ic_frame, paired_delta_tests
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.quantization.profiling import quantizable_input_points
from fin_ts_ptq.quantization.rtn import fake_static_int8
from fin_ts_ptq.training.loop import predict
from fin_ts_ptq.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Only-one / leave-one-out layer sensitivity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--policy", default="recent")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--label", default=None)
    parser.add_argument("--test-year", type=int, default=None,
                        help="Walk-forward fold override: train<=Y-2, val=Y-1, test=Y (same as run_sweep).")
    parser.add_argument("--train-years", type=int, default=None,
                        help="Fixed training-window length; must match the sweep that trained the checkpoint (norm stats).")
    parser.add_argument("--output-dir", default="outputs/mechanism")
    parser.add_argument("--seed", type=int, default=None)
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

    layers = [name for name, _ in quantizable_input_points(model)]
    print(f"[LAYERS] {len(layers)}: {layers}", flush=True)

    calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
    cal_loader = build_calibration_loader(
        data.splits["val"], policy=args.policy, calibration_cfg=calibration_cfg,
        seed=int(cfg.get("seed", 7)), batch_size=int(args.batch_size),
    )
    cal_batches = [b for b in cal_loader]
    cal_inputs = [b[0] for b in cal_batches]

    preds: dict[str, np.ndarray] = {}
    target_true, preds["fp32"] = predict(model, loader, device)
    _, preds["full"] = predict(
        fake_static_int8(model, cal_inputs, num_bits=args.bits), loader, device
    )
    print("[EVAL] fp32 + full done", flush=True)
    for i, layer in enumerate(layers):
        qm_only = fake_static_int8(model, cal_inputs, num_bits=args.bits, layer_filter=lambda n, L=layer: n == L)
        _, preds[f"oo::{layer}"] = predict(qm_only, loader, device)
        qm_loo = fake_static_int8(model, cal_inputs, num_bits=args.bits, layer_filter=lambda n, L=layer: n != L)
        _, preds[f"loo::{layer}"] = predict(qm_loo, loader, device)
        print(f"[EVAL] {i+1}/{len(layers)} {layer}", flush=True)

    frame = daily_ic_frame(test_ds.meta, target_true, preds)
    hac_lag = max(0, int(cfg["task"].get("forward_period", 1)) - 1)
    tests = paired_delta_tests(frame, baseline="fp32", lag=hac_lag)

    report = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "bits": int(args.bits),
        "policy": args.policy,
        "fp32_ic_mean": float(frame["fp32"].mean()),
        "full_delta": tests["full"],
        "layers": {
            layer: {"only_one": tests[f"oo::{layer}"], "leave_one_out": tests[f"loo::{layer}"]}
            for layer in layers
        },
    }
    out = Path(args.output_dir) / f"{label}_layer_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    ranked = sorted(layers, key=lambda l: -tests[f"oo::{l}"]["delta_ic_mean"])
    print(f"fp32 IC={report['fp32_ic_mean']:+.4f}  full W4A4 delta={tests['full']['delta_ic_mean']:+.4f}")
    for l in ranked[:5]:
        print(f"  only-one {l}: {tests[f'oo::{l}']['delta_ic_mean']:+.4f}   "
              f"loo: {tests[f'loo::{l}']['delta_ic_mean']:+.4f}")
    print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    main()
